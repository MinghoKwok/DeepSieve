"""
DeepSieve RAG-only pipeline

This script implements a modular RAG pipeline with the following features:
- LLM-based query decomposition (optional)
- Per-subquestion routing to local/global knowledge sources
- Light-weight vector RAG with top-k retrieval
- Structured LLM prompting for reasoning + JSON parsing
- Reflection mechanism to reroute or rephrase failed queries
- Final answer fusion based on reasoning trace
- Comprehensive logging of retrieval, token usage, and accuracy metrics

Usage:
    Configure flags like `decompose`, `use_routing`, `use_reflection` to ablate components.
    Example:
        python script.py --decompose True --use_routing True --use_reflection True

Output:
    Each query's full trace is saved in outputs/{dataset_name}/query_{i}_results.jsonl
    Aggregated metrics saved in overall_results.json

Compatible datasets:
    - hotpot_qa
    - 2wikimultihopqa
    - musique
"""


import os
import json
import requests
import numpy as np
from typing import List, Dict, Union, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import tiktoken
import networkx as nx
from collections import defaultdict
import re
from rag.naive_rag import NaiveRAG
from rag.graph_rag import GraphRAG_Improved

# LLM 调用

def call_openai_chat(prompt: str, api_key: str, model: str, base_url: str, max_retries: int = 3) -> str:
    """Call OpenAI Chat API with retry mechanism

    Args:
        prompt: Prompt text
        api_key: API key
        model: Model name
        base_url: API base URL
        max_retries: Max retry times

    Returns:
        str: API response content
    """
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    
    session = requests.Session()
    retry_strategy = requests.adapters.Retry(
        total=max_retries,
        backoff_factor=2,  # Increase backoff time
        status_forcelist=[500, 502, 503, 504, 408, 429],  # Add timeout and rate limit status codes
        allowed_methods=["POST"]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy, pool_maxsize=100)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    try:
        response = session.post(url, headers=headers, json=payload, timeout=60)  # 增加超时时间到60秒
        response.raise_for_status()  # 检查响应状态
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"🔴 API request error: {str(e)}")
        if isinstance(e, (requests.exceptions.ChunkedEncodingError, requests.exceptions.ReadTimeout)):
            print("Detected connection error, retrying...")
            # Special handling for connection errors
            for i in range(max_retries):
                try:
                    print(f"Retry #{i+1}...")
                    time.sleep(2 ** i)  # 指数退避
                    response = session.post(url, headers=headers, json=payload, timeout=60)
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
                except requests.exceptions.RequestException as retry_e:
                    print(f"Retry #{i+1} failed: {str(retry_e)}")
                    if i == max_retries - 1:  # 如果是最后一次重试
                        print("All retries failed")
                        return ""
    except Exception as e:
        print(f"🔴 Other error: {str(e)}")
        return ""
    finally:
        session.close()


def plan_subqueries_with_llm(query: str, api_key: str, model: str, base_url: str) -> dict:
    prompt = f"""You are a reasoning planner. Your task is to decompose a multi-hop question into a sequence of dependent sub-questions.
For each sub-question, you should:
1. Identify any variables that need to be filled from previous sub-questions' answers
2. Specify the dependency relationship between sub-questions
3. Use consistent variable names in square brackets (e.g. [birthplace]) to show dependencies

Question: {query}

Please output in JSON format as follows:
{{
    "subqueries": [
        {{
            "id": "q1",
            "query": "First sub-question without dependencies",
            "depends_on": [],
            "variables": []
        }},
        {{
            "id": "q2",
            "query": "Second sub-question that may contain [variable_from_q1]",
            "depends_on": ["q1"],
            "variables": [
                {{
                    "name": "variable_from_q1",
                    "source_query": "q1"
                }}
            ]
        }},
        ...
    ]
}}
Only output valid JSON. Do not add any explanation or markdown code block markers."""

    response = call_openai_chat(prompt, api_key, model, base_url)
    try:
        # Remove possible markdown code block markers from response
        cleaned_response = str(response).strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        result = json.loads(cleaned_response)
        if "subqueries" not in result:
            print("⚠️ Missing subqueries field in response:")
            print(result)
            return {"subqueries": []}
        return result
    except json.JSONDecodeError as e:
        print("⚠️ Failed to parse JSON from LLM response:")
        print(response)
        print(f"Error: {str(e)}")
        return {"subqueries": []}


def substitute_variables(query: str, variable_values: dict) -> str:
    """
    Replace variables in the query with their actual values.
    For example: replace "What country is [birthplace] in?" with the actual value of [birthplace].
    """
    result = query
    for var_name, value in variable_values.items():
        result = result.replace(f"[{var_name}]", value)
    return result


def route_query_with_llm(query: str, local_profile: str, global_profile: str,
                         api_key: str, model: str, base_url: str, fail_history: str) -> str:
    """Route the query to the appropriate knowledge base

    Args:
        query: Query text
        local_profile: Local knowledge base description
        global_profile: Global knowledge base description
        api_key: API key
        model: Model name
        base_url: API base URL
        fail_history: Failure history

    Returns:
        str: Routing result ("local" or "global")
    """
    prompt = f"""You are a routing assistant. Your task is to decide whether a query should be answered using LOCAL knowledge or GLOBAL knowledge.

LOCAL PROFILE:
{local_profile}

GLOBAL PROFILE:
{global_profile}

QUERY:
{query}

{fail_history}

Please output only one word: \"local\" or \"global\" based on which profile is more relevant to the query.
Do not add any explanation or extra words."""

    try:
        response = call_openai_chat(prompt, api_key, model, base_url)
        if not response:  # 如果响应为空
            print("⚠️ Routing response is empty, defaulting to local routing")
            return "local"
        
        route = response.strip().lower()
        if route not in {"local", "global"}:
            print(f"⚠️ Unexpected routing output: {route}, defaulting to local routing")
            return "local"
        return route
    except Exception as e:
        print(f"⚠️ Routing error: {str(e)}, defaulting to local routing")
        return "local"


# 主程序

def normalize_answer(s: str) -> str:
    """
    Normalize answer string for comparison
    1. Lowercase
    2. Remove punctuation and extra spaces
    3. Remove stopwords like a, an, the
    """
    import re
    from string import punctuation
    
    # 转换为小写
    s = s.lower()
    
    # 移除标点符号
    s = s.translate(str.maketrans("", "", punctuation))
    
    # 移除多余空格
    s = " ".join(s.split())
    
    # 移除停用词
    stop_words = {"a", "an", "the", "is", "are", "was", "were"}
    s = " ".join([w for w in s.split() if w not in stop_words])
    
    return s

def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute Exact Match score
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    # 计算重叠的token数量
    common = set(prediction_tokens) & set(ground_truth_tokens)
    
    if not common:
        return 0.0
    
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(ground_truth_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def evaluate_answer(prediction: str, ground_truth: str) -> dict:
    """
    Evaluate the quality of the predicted answer
    """
    return {
        "exact_match": compute_exact_match(prediction, ground_truth),
        "f1": compute_f1(prediction, ground_truth)
    }

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # fallback
    return len(enc.encode(text))

def calculate_overall_metrics(all_metrics):
    """
    Calculate average performance metrics for all queries
    """
    total_queries = len(all_metrics)
    if total_queries == 0:
        return {}
    
    overall = {
        "avg_exact_match": sum(m["evaluation_metrics"]["exact_match"] for m in all_metrics) / total_queries,
        "avg_f1": sum(m["evaluation_metrics"]["f1"] for m in all_metrics) / total_queries,
        "avg_retrieval_time": sum(m["total_retrieval_time"] for m in all_metrics) / total_queries,
        "avg_docs_searched": sum(m["total_docs_searched"] for m in all_metrics) / total_queries,
        "avg_similarity": sum(m["avg_similarity"] for m in all_metrics) / total_queries,
        "avg_prompt_tokens_per_subquery": sum(m["avg_prompt_tokens"] for m in all_metrics) / total_queries,
        "avg_total_tokens_per_query": sum(m["total_prompt_tokens"] for m in all_metrics) / total_queries
    }
    return overall


def convert_to_query_format(df, dataset_name):
    """Convert different dataset DataFrames to a unified query format"""
    if dataset_name in ["hotpot_qa", "trivia_qa", "gsm8k", "sotu_qa"]:
        # These datasets already use question/answer format
        return [{"query": row["question"], "ground_truth": row["answer"]} 
                for _, row in df.iterrows()]
    elif dataset_name in ["physics_question", "sports_understanding", "disfl_qa", "strategy_qa"]:
        # These datasets use input/target format
        return [{"query": row["input"], "ground_truth": row["target"]} 
                for _, row in df.iterrows()]
    elif dataset_name == "fever":
        # FEVER dataset uses claim/label format
        return [{"query": row["claim"], "ground_truth": row["label"]} 
                for _, row in df.iterrows()]
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")


def get_fused_final_answer(original_question: str, subquery_results: List[Dict], api_key: str, model: str, base_url: str) -> tuple:
    prompt = f"""You are a multi-hop reasoning assistant. Your task is to generate the final answer to a multi-hop question based on the following reasoning steps.

Original Question: {original_question}

Subquestion Reasoning Steps:
"""
    for r in subquery_results:
        prompt += f"{r['subquery_id']}: {r['actual_query']} → {r['answer']}\n"
        prompt += f"Reason: {r['reason']}\n\n"

    prompt += """\nBased on the above reasoning steps, what is the final answer to the original question?

Please respond in JSON format:
{
  "answer": "final_answer",
  "reason": "final_reasoning"
}
Only output valid JSON. Do not add any explanation or markdown code block markers."""

    token_count = count_tokens(prompt, model)
    print(f"🧠 Fusion Prompt Token Count: {token_count}")

    response = call_openai_chat(prompt, api_key, model, base_url)
    try:
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        parsed = json.loads(cleaned_response)
        answer = parsed.get("answer", "").strip()
        reason = parsed.get("reason", "").strip()
        print(f"✅ Final fused answer: {answer}")
        print(f"🔎 Final reasoning: {reason}")
        return answer, reason, token_count, prompt
    except Exception as e:
        print(f"⚠️ Failed to parse fused answer: {e}")
        return "", "", token_count, prompt


def main(decompose: bool = True, use_routing: bool = True, use_reflection: bool = True, max_reflexion_times: int = 2, dataset: str = "hotpot_qa", sample_size: int = 100, openai_model: str = "deepseek-chat", openai_api_key: str = None, openai_base_url: str = None, rag_type: str = "naive"):
    """
    Main function
    Args:
        decompose: Whether to decompose the query
        use_routing: Whether to use routing
        use_reflection: Whether to use reflection mechanism
        max_reflexion_times: Max reflection times
        dataset: Dataset name
        sample_size: Sample size
        openai_model: OpenAI model name
        openai_api_key: OpenAI API key
        openai_base_url: OpenAI API base URL
        rag_type: RAG type, can be "naive" or "graph"
    """
    # Define multiple queries and corresponding ground truth
    # Load data
    with open(f"data/rag/{dataset}.json", "r") as f:
        data = json.load(f)

    queries_and_truth = [
        {
            "query": item["question"],
            "ground_truth": item["answer"]
        }
        for item in data[:sample_size]
    ]

    save_dir = "outputs/"
    save_dir += rag_type
    save_dir += "_"
    save_dir += dataset
    save_dir += "_"
    if not use_routing:
        save_dir += "_no_routing"
    if not decompose:
        save_dir += "_no_decompose"
    if not use_reflection:
        save_dir += "_no_reflection"
    os.makedirs(save_dir, exist_ok=True)

    openai_model = openai_model
    openai_api_key = openai_api_key
    openai_base_url = openai_base_url
    if not openai_api_key:
        raise ValueError("❌ Please set your OPENAI_API_KEY environment variable.")

    # Prepare knowledge base documents
    with open(f"data/rag/{dataset}_corpus_local.json", "r") as f:
        data = json.load(f)
    local_docs = [f"{item['title']}. {item['text']}" for item in data]
    print(f"✅ Loaded {len(local_docs)} documents into local_docs.")

    with open(f"data/rag/{dataset}_corpus_global.json", "r") as f:
        data = json.load(f)
    global_docs = [f"{item['title']}. {item['text']}" for item in data]
    print(f"✅ Loaded {len(global_docs)} documents into global_docs.")

    # Read profiles.json
    with open(f"data/rag/{dataset}_corpus_profiles.json", "r") as f:
        profiles = json.load(f)
    local_profile = profiles["local_profile"]
    global_profile = profiles["global_profile"]
    print(f"✅ Loaded local_profile and global_profile.")

    merged_docs = local_docs + global_docs

    # Initialize RAG system
    if use_routing:
        if rag_type == "naive":
            local_rag = NaiveRAG(local_docs)
            global_rag = NaiveRAG(global_docs)
        elif rag_type == "graph":
            local_rag = GraphRAG_Improved(local_docs)
            global_rag = GraphRAG_Improved(global_docs)
        else:
            raise ValueError(f"Unsupported RAG type: {rag_type}")
        print(f"🔍 Using routing mode: initialized local and global knowledge bases, RAG type: {rag_type}")
    else:
        # Merge datasets
        if rag_type == "naive":
            merged_rag = NaiveRAG(merged_docs)
        elif rag_type == "graph":
            merged_rag = GraphRAG_Improved(merged_docs)
        else:
            raise ValueError(f"Unsupported RAG type: {rag_type}")
        print(f"🔍 Using no-routing mode: merged local and global knowledge bases, RAG type: {rag_type}")

    all_metrics = []  # Store all query performance metrics

    # Process each query
    for idx, query_info in enumerate(queries_and_truth, 1):
        multi_hop_query = query_info["query"]
        ground_truth = query_info["ground_truth"]
        
        print(f"\n📝 Processing query {idx}/{len(queries_and_truth)}:")
        print(f"Query: {multi_hop_query}")
        print(f"Ground Truth: {ground_truth}")
        
        # Initialize variable_values dict
        variable_values = {}
        
        if decompose:
            # Get subquery plan with dependencies
            query_plan = plan_subqueries_with_llm(multi_hop_query, openai_api_key, openai_model, openai_base_url)
            if not query_plan or not query_plan["subqueries"]:
                print("❌ Subquery planning failed, skipping current query.")
                continue
        else:
            # Do not decompose, treat as a single question
            query_plan = {
                "subqueries": [{
                    "id": "q1",
                    "query": multi_hop_query,
                    "depends_on": [],
                    "variables": []
                }]
            }

        results, fused_answer_texts = [], []
        performance_metrics = {
            "total_retrieval_time": 0,
            "total_docs_searched": 0,
            "avg_similarity_scores": [],
            "max_similarity_scores": [],
            "subquery_metrics": [],
            "total_prompt_tokens": 0,
            "prompt_token_counts": [],
            "evaluation_metrics": {
                "exact_match": 0,
                "f1": 0,
                "final_answer": "",
                "ground_truth": ground_truth
            },
            "evaluation_metrics_fallback": {
                "exact_match": 0,
                "f1": 0,
                "ground_truth": ground_truth
            }
        }

        # Process subqueries in order, handle dependencies
        for subquery_info in query_plan["subqueries"]:
            subquery_start_time = time.time()
            subquery_id = subquery_info["id"]
            original_query = subquery_info["query"]
            
            # Check and wait for all dependencies to complete
            if subquery_info["depends_on"]:
                print(f"\n⏳ Processing dependencies for query {subquery_id}: {subquery_info['depends_on']}")
                
            # Replace variables in the query
            current_variables = {}
            for var in subquery_info.get("variables", []):
                var_name = var["name"]
                source_query = var["source_query"]
                if source_query not in variable_values:
                    print(f"❌ Error: Query {subquery_id} depends on an incomplete query {source_query}")
                    continue
                current_variables[var_name] = variable_values[source_query]
            
            # Actual query after variable substitution
            actual_query = substitute_variables(original_query, current_variables)
            print(f"\n🔍 Processing query {subquery_id}: {actual_query}")
            print(f"Original query: {original_query}")
            if current_variables:
                print(f"Variable substitution: {current_variables}")

            # Loop for reflection
            fail_history = ""
            left_reflexion_times = max_reflexion_times
            while True and left_reflexion_times > 0:
                left_reflexion_times -= 1
                success = 0  # 初始化success为0
                if use_routing:
                    route = route_query_with_llm(actual_query, local_profile, global_profile,
                                            api_key=openai_api_key, model=openai_model, base_url=openai_base_url, fail_history=fail_history)
                    rag = local_rag if route == "local" else global_rag
                    print(f"Routing to {route.upper()} DB")
                else:
                    rag = merged_rag
                    route = "merged"
                    print(f"Using merged DB")

                try:
                    retrieved = rag.rag_qa(actual_query, k=5)
                    
                    # Collect performance metrics
                    metrics = retrieved["metrics"]
                    performance_metrics["total_retrieval_time"] += metrics["retrieval_time"]
                    performance_metrics["total_docs_searched"] += metrics["total_docs_searched"]
                    performance_metrics["avg_similarity_scores"].append(metrics["avg_similarity"])
                    performance_metrics["max_similarity_scores"].append(metrics["max_similarity"])
                    
                    subquery_metrics = {
                        "subquery_id": subquery_id,
                        "retrieval_time": metrics["retrieval_time"],
                        "docs_searched": metrics["total_docs_searched"],
                        "avg_similarity": metrics["avg_similarity"],
                        "max_similarity": metrics["max_similarity"]
                    }
                    performance_metrics["subquery_metrics"].append(subquery_metrics)
                    
                    prompt = f"""Answer the following question based on the provided documents.

    Please respond strictly in JSON format with the following fields:
    - answer: the direct, concise answer (just the value/entity/fact, no explanation). Leave it empty ("") if the answer is not found.
    - reason: a brief explanation of how you arrived at this answer.
    - success: 1 if the answer is confidently found from the documents, 0 otherwise.

    Format:
    {{
        "answer": "...",
        "reason": "...",
        "success": 1
    }}

    If the answer is not mentioned or cannot be inferred from the documents, return:
    {{
        "answer": "",
        "reason": "no relevant information found",
        "success": 0
    }}

    Question: {actual_query}

    Documents:
    """
                    for d in retrieved["docs"]:
                        prompt += f"- {d}\n"
                    prompt += "\nOnly output valid JSON. Do not add any explanation or markdown code block markers."

                    token_count = count_tokens(prompt, openai_model)
                    print(f"🧮 Prompt Token Count: {token_count}")
                    
                    response = call_openai_chat(prompt, openai_api_key, openai_model, openai_base_url)
                    try:
                        # 清理响应中可能存在的markdown代码块标记
                        cleaned_response = response.strip()
                        if cleaned_response.startswith("```json"):
                            cleaned_response = cleaned_response[7:]
                        if cleaned_response.endswith("```"):
                            cleaned_response = cleaned_response[:-3]
                        cleaned_response = cleaned_response.strip()
                        
                        parsed_response = json.loads(cleaned_response)
                        answer = parsed_response["answer"].strip()
                        reason = parsed_response["reason"].strip()
                        success = int(parsed_response["success"])
                        
                        # Store answer for subsequent queries
                        if success == 1:
                            variable_values[subquery_id] = answer
                            print(f"Extracted answer: {answer}")
                            print(f"Reasoning: {reason}")
                            print(f"Success: {success}")
                        else:
                            variable_values[subquery_id] = ""
                            fail_history += f"Fail History: Last routing failed because {reason}. Last routing result is {route}. So please try another routing choice, don't choose {route} again."
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"⚠️ Failed to parse answer: {str(e)}")
                        print(f"Raw response: {response}")
                        answer, reason = f"Error: {str(e)}", ""
                        success = 0
                    
                    # Update token statistics
                    performance_metrics["total_prompt_tokens"] += token_count
                    performance_metrics["prompt_token_counts"].append(token_count)
                    
                    results.append({
                        "subquery_id": subquery_id,
                        "original_query": original_query,
                        "actual_query": actual_query,
                        "routing": route,
                        "answer": answer,
                        "reason": reason,
                        "docs": retrieved["docs"],
                        "doc_scores": retrieved["doc_scores"],
                        "variables_used": current_variables,
                        "metrics": subquery_metrics,
                        "prompt_token_count": token_count
                    })
                    fused_answer_texts.append(f"{subquery_id}: {actual_query} → {answer} (reason: {reason})")
                    
                except Exception as e:
                    print(f"⚠️ Error occurred while processing query: {str(e)}")
                    answer, retrieved = f"Error: {str(e)}", {"docs": [], "doc_scores": []}
                    success = 0  # 设置success为0

                if success == 1 or use_routing == False or use_reflection == False or left_reflexion_times <= 0:
                    break   # 如果成功或者不使用路由或者不使用反射或者反射次数用完，则跳出循环

        # Compute summary metrics
        performance_metrics["avg_retrieval_time"] = performance_metrics["total_retrieval_time"] / len(query_plan["subqueries"])
        performance_metrics["avg_similarity"] = np.mean(performance_metrics["avg_similarity_scores"])
        performance_metrics["max_similarity"] = np.max(performance_metrics["max_similarity_scores"])
        
        # Compute token statistics
        performance_metrics["avg_prompt_tokens"] = performance_metrics["total_prompt_tokens"] / len(query_plan["subqueries"])
        performance_metrics["max_prompt_tokens"] = max(performance_metrics["prompt_token_counts"], default=0)
        performance_metrics["min_prompt_tokens"] = min(performance_metrics["prompt_token_counts"], default=0)

        # Get fallback (last hop) answer (for comparison)
        fallback_answer = results[-1]["answer"] if results else ""
        performance_metrics["evaluation_metrics"]["fallback_answer"] = fallback_answer


        # Get final answer (fusing all reasoning steps)
        final_answer, final_reason, fusion_token_count, fusion_prompt = get_fused_final_answer(
            multi_hop_query, results,
            api_key=openai_api_key,
            model=openai_model,
            base_url=openai_base_url
        )
        performance_metrics["evaluation_metrics"]["final_answer"] = final_answer
        performance_metrics["evaluation_metrics"]["final_reason"] = final_reason
        performance_metrics["fusion_prompt_tokens"] = fusion_token_count

        performance_metrics["total_prompt_tokens"] += fusion_token_count
        performance_metrics["prompt_token_counts"].append(fusion_token_count)

        
        # Compute evaluation metrics
        eval_results = evaluate_answer(final_answer, ground_truth)
        eval_results_fallback = evaluate_answer(fallback_answer, ground_truth)
        performance_metrics["evaluation_metrics"].update(eval_results)
        performance_metrics["evaluation_metrics_fallback"].update(eval_results_fallback)
        
        # Save current query results
        query_results_path = os.path.join(save_dir, f"query_{idx}_results.jsonl")
        with open(query_results_path, "w") as f:
            # 1st: Query metadata
            f.write(json.dumps({
                "type": "query_info",
                "query": multi_hop_query,
                "ground_truth": ground_truth
            }) + "\n")

            # 2nd: Final answer summary
            f.write(json.dumps({
                "type": "final_answer",
                "final_answer": final_answer,
                "final_reason": final_reason,
                "fusion_prompt_tokens": fusion_token_count,
                "fallback_answer": fallback_answer,
                "fusion_equals_fallback": fallback_answer.strip().lower() == final_answer.strip().lower()
            }) + "\n")

            # 3rd: Evaluation metrics
            f.write(json.dumps({
                "type": "evaluation_metrics",
                "fusion": {
                    "exact_match": eval_results["exact_match"],
                    "f1": eval_results["f1"]
                },
                "fallback": {
                    "exact_match": eval_results_fallback["exact_match"],
                    "f1": eval_results_fallback["f1"]
                }
            }) + "\n")

            # 4th: Performance metrics
            f.write(json.dumps({
                "type": "performance_metrics",
                "total_retrieval_time": performance_metrics["total_retrieval_time"],
                "avg_retrieval_time": performance_metrics["avg_retrieval_time"],
                "total_docs_searched": performance_metrics["total_docs_searched"],
                "avg_similarity": performance_metrics["avg_similarity"],
                "max_similarity": performance_metrics["max_similarity"],
                "token_cost": {
                    "total_prompt_tokens": performance_metrics["total_prompt_tokens"],
                    "avg_prompt_tokens": performance_metrics["avg_prompt_tokens"],
                    "max_prompt_tokens": performance_metrics["max_prompt_tokens"],
                    "min_prompt_tokens": performance_metrics["min_prompt_tokens"]
                }
            }) + "\n")

            # 5th: Subquery performance details
            for metrics in performance_metrics["subquery_metrics"]:
                f.write(json.dumps({
                    "type": "subquery_metric",
                    "subquery_id": metrics["subquery_id"],
                    "retrieval_time": metrics["retrieval_time"],
                    "docs_searched": metrics["docs_searched"],
                    "avg_similarity": metrics["avg_similarity"],
                    "max_similarity": metrics["max_similarity"]
                }) + "\n")

            # 6th: Execution results (per subquery)
            for r in results:
                f.write(json.dumps({
                    "type": "execution_result",
                    "subquery_id": r["subquery_id"],
                    "original_query": r["original_query"],
                    "actual_query": r["actual_query"],
                    "variables_used": r.get("variables_used", None),
                    "routing": r["routing"],
                    "answer": r["answer"],
                    "reason": r["reason"],
                    "docs": [
                        {
                            "text": doc,
                            "score": r["doc_scores"][i]
                        }
                        for i, doc in enumerate(r["docs"])
                    ]
                }) + "\n")

            # 7th: Answer chain
            for step in fused_answer_texts:
                f.write(json.dumps({
                    "type": "fused_answer_step",
                    "text": step
                }) + "\n")        
        # Save fusion prompt
        fusion_prompt_path = os.path.join(save_dir, f"query_{idx}_fusion_prompt.txt")
        with open(fusion_prompt_path, "w") as f_prompt:
            f_prompt.write(fusion_prompt)


        all_metrics.append(performance_metrics)

    # Compute and save overall performance metrics
    overall_metrics = calculate_overall_metrics(all_metrics)
    
    # Save overall results
    overall_txt_path = os.path.join(save_dir, "overall_results.txt")
    with open(overall_txt_path, "w") as f:
        f.write("📊 Overall Performance Summary:\n")
        f.write(f"- Average Exact Match: {overall_metrics['avg_exact_match']:.4f}\n")
        f.write(f"- Average F1 Score: {overall_metrics['avg_f1']:.4f}\n")
        f.write(f"- Average Retrieval Time: {overall_metrics['avg_retrieval_time']:.4f}s\n")
        f.write(f"- Average Documents Searched: {overall_metrics['avg_docs_searched']:.1f}\n")
        f.write(f"- Average Similarity Score: {overall_metrics['avg_similarity']:.4f}\n")
        f.write(f"- Average Prompt Tokens per Subquery: {overall_metrics['avg_prompt_tokens_per_subquery']:.2f}\n")
        f.write(f"- Average Total Tokens per Query: {overall_metrics['avg_total_tokens_per_query']:.2f}\n")

        

    overall_json_path = os.path.join(save_dir, "overall_results.json")
    with open(overall_json_path, "w") as f:
        json.dump({
            "queries": queries_and_truth,
            "overall_metrics": overall_metrics,
            "all_query_metrics": all_metrics
        }, f, indent=2)

    # Update console output
    print("\n📊 Overall Performance Summary:")
    print(f"- Average Exact Match: {overall_metrics['avg_exact_match']:.4f}")
    print(f"- Average F1 Score: {overall_metrics['avg_f1']:.4f}")
    print(f"- Average Retrieval Time: {overall_metrics['avg_retrieval_time']:.4f}s")
    print(f"- Average Documents Searched: {overall_metrics['avg_docs_searched']:.1f}")
    print(f"- Average Similarity Score: {overall_metrics['avg_similarity']:.4f}")
    print(f"- Average Prompt Tokens per Subquery: {overall_metrics['avg_prompt_tokens_per_subquery']:.2f}")
    print(f"- Average Total Tokens per Query: {overall_metrics['avg_total_tokens_per_query']:.2f}")

    print("\n✅ Results saved to:")
    print(f"   - {overall_txt_path}")
    print(f"   - {overall_json_path}")
    for i in range(len(queries_and_truth)):
        print(f"   - {os.path.join(save_dir, f'query_{i+1}_results.txt')}")


if __name__ == "__main__":
    # Default to decompose and routing mode
    main(decompose=True, 
         use_routing=True, 
         use_reflection=True, 
         max_reflexion_times=1, 
         dataset="hotpot_qa", 
         sample_size=5, 
         openai_model=os.environ.get("OPENAI_MODEL"),  
         openai_api_key=os.environ.get("OPENAI_API_KEY"), 
         openai_base_url=os.environ.get("OPENAI_API_BASE"),
         rag_type=os.environ.get("RAG_TYPE", "graph"))  # naive or graph
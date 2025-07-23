"""
pipeline/subquery_executor.py

This module implements the subquery executor for DeepSieve.
"""

import time
import json
from utils.llm_call import call_openai_chat
from utils.metrics import count_tokens
from pipeline.reasoning_pipeline import route_query_with_llm, substitute_variables



def execute_subquery(
    subquery_info: dict, 
    variable_values: dict, 
    local_rag,  
    global_rag, 
    merged_rag, 
    use_routing, 
    use_reflection, 
    max_reflexion_times, 
    local_profile, 
    global_profile, 
    openai_api_key, 
    openai_model, 
    openai_base_url
) -> dict:
    """
    Execute a subquery and return the results.
    """

    # Ensure all return variables are initialized
    answer = ""
    reason = ""
    success = 0
    retrieved = {"docs": [], "doc_scores": []}
    subquery_metrics = {
        "subquery_id": subquery_info.get("id", ""),
        "retrieval_time": 0,
        "docs_searched": 0,
        "avg_similarity": 0,
        "max_similarity": 0
    }
    token_count = 0
    current_variables = {}
    performance_metrics = {
        "total_retrieval_time": 0,
        "total_docs_searched": 0,
        "avg_similarity_scores": [],
        "max_similarity_scores": [],
        "subquery_metrics": [],
        "total_prompt_tokens": 0,
        "prompt_token_counts": [],
    }
    results = []
    fused_answer_texts = []

    subquery_start_time = time.time()
    subquery_id = subquery_info["id"]
    original_query = subquery_info["query"]
    
    # Check and wait for all dependencies to complete
    if subquery_info["depends_on"]:
        print(f"\n⏳ Processing dependencies for query {subquery_id}: {subquery_info['depends_on']}")
        
    # Replace variables in the query
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
                answer = f"Error: {str(e)}"
                reason = ""
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
            answer = f"Error: {str(e)}"
            reason = ""
            success = 0
            retrieved = {"docs": [], "doc_scores": []}
            subquery_metrics = {
                "subquery_id": subquery_id,
                "retrieval_time": 0,
                "docs_searched": 0,
                "avg_similarity": 0,
                "max_similarity": 0
            }
            token_count = 0

        if success == 1 or use_routing == False or use_reflection == False or left_reflexion_times <= 0:
            break   

    return {
        "performance_metrics": performance_metrics,
        "answer": answer,
        "reason": reason,
        "success": success,
        "docs": retrieved["docs"],
        "doc_scores": retrieved["doc_scores"],
        "variables_used": current_variables,
        "metrics": subquery_metrics,
        "prompt_token_count": token_count,
        "subquery_id": subquery_id,
        "original_query": original_query,
        "actual_query": actual_query,
        "routing": route,
        # for main aggregation:
        "retrieval_time": subquery_metrics["retrieval_time"],
        "docs_searched": subquery_metrics["docs_searched"],
        "avg_similarity": subquery_metrics["avg_similarity"],
        "max_similarity": subquery_metrics["max_similarity"]
    }

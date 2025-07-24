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
from rag.initializer import initialize_rag_system
from pipeline.reasoning_pipeline import plan_subqueries_with_llm, route_query_with_llm, get_fused_final_answer, substitute_variables
from pipeline.subquery_executor import execute_subquery
from utils.data_load import load_queries, load_corpus_and_profiles
from utils.llm_call import call_openai_chat
from utils.metrics import count_tokens, evaluate_answer, calculate_overall_metrics

def get_save_dir(decompose: bool, use_routing: bool, use_reflection: bool, dataset: str, rag_type: str):
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
    return save_dir

def save_overall_results(save_dir, overall_metrics, queries_and_truth, all_metrics):
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

def save_single_query_results(save_dir, idx, multi_hop_query, ground_truth, final_answer, final_reason, fusion_token_count, fallback_answer, fusion_prompt, eval_results, eval_results_fallback, performance_metrics, results, fused_answer_texts):
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
    return performance_metrics

def process_subqueries(performance_metrics, query_plan, variable_values, local_rag, global_rag, merged_rag, use_routing, use_reflection, max_reflexion_times, local_profile, global_profile, openai_api_key, openai_model, openai_base_url, save_dir, idx, multi_hop_query, ground_truth, results, fused_answer_texts):
    for subquery_info in query_plan["subqueries"]:
        subquery_result = execute_subquery(
            subquery_info,
            variable_values,
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
        )
        results.append(subquery_result)
        fused_answer_texts.append(f"{subquery_result['subquery_id']}: {subquery_result['actual_query']} → {subquery_result['answer']} (reason: {subquery_result['reason']})")
        performance_metrics["total_retrieval_time"] += subquery_result["retrieval_time"]
        performance_metrics["total_docs_searched"] += subquery_result["docs_searched"]
        performance_metrics["avg_similarity_scores"].append(subquery_result["avg_similarity"])
        performance_metrics["max_similarity_scores"].append(subquery_result["max_similarity"])
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
    performance_metrics = save_single_query_results(save_dir, idx, multi_hop_query, ground_truth, final_answer, final_reason, fusion_token_count, fallback_answer, fusion_prompt, eval_results, eval_results_fallback, performance_metrics, results, fused_answer_texts)

    return performance_metrics

def single_query_execution(decompose, all_metrics, queries_and_truth, local_rag, global_rag, merged_rag, use_routing, use_reflection, max_reflexion_times, local_profile, global_profile, openai_api_key, openai_model, openai_base_url, save_dir):
    # Process each query
    for idx, query_info in enumerate(queries_and_truth, 1):
        multi_hop_query = query_info["query"]
        ground_truth = query_info["ground_truth"]
        
        print(f"\n📝 Processing query {idx}/{len(queries_and_truth)}:")
        print(f"Query: {multi_hop_query}")
        print(f"Ground Truth: {ground_truth}")
        
        # Initialize variable_values dict
        variable_values = {}

        query_plan = plan_subqueries_with_llm(decompose, multi_hop_query, openai_api_key, openai_model, openai_base_url)
        if not query_plan or not query_plan["subqueries"]:
            print("❌ Subquery planning failed, skipping current query.")
            continue

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
        performance_metrics = process_subqueries(performance_metrics, query_plan, variable_values, local_rag, global_rag, merged_rag, use_routing, use_reflection, max_reflexion_times, local_profile, global_profile, openai_api_key, openai_model, openai_base_url, save_dir, idx, multi_hop_query, ground_truth, results, fused_answer_texts)
        all_metrics.append(performance_metrics)
    
    return all_metrics

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
    # Load data
    queries_and_truth = load_queries(dataset, sample_size)
    save_dir = get_save_dir(decompose, use_routing, use_reflection, dataset, rag_type)
    os.makedirs(save_dir, exist_ok=True)

    openai_model = openai_model
    openai_api_key = openai_api_key
    openai_base_url = openai_base_url
    if not openai_api_key:
        raise ValueError("❌ Please set your OPENAI_API_KEY environment variable.")

    # Prepare knowledge base documents
    local_docs, global_docs, local_profile, global_profile = load_corpus_and_profiles(dataset)
    print(f"✅ Loaded {len(local_docs)} documents into local_docs.")
    print(f"✅ Loaded {len(global_docs)} documents into global_docs.")
    print(f"✅ Loaded local_profile and global_profile.")

    # Initialize RAG system
    local_rag, global_rag, merged_rag = initialize_rag_system(rag_type, use_routing, local_docs, global_docs)

    all_metrics = []  # Store all query performance metrics
    

    all_metrics = single_query_execution(decompose, all_metrics, queries_and_truth, local_rag, global_rag, merged_rag, use_routing, use_reflection, max_reflexion_times, local_profile, global_profile, openai_api_key, openai_model, openai_base_url, save_dir)
    # Compute and save overall performance metrics
    overall_metrics = calculate_overall_metrics(all_metrics)
    
    # Save overall results
    save_overall_results(save_dir, overall_metrics, queries_and_truth, all_metrics)

if __name__ == "__main__":
    # Default to decompose and routing mode
    main(decompose=True, 
         use_routing=True, 
         use_reflection=True, 
         max_reflexion_times=2, 
         dataset="hotpot_qa", 
         sample_size=1, 
         openai_model=os.environ.get("OPENAI_MODEL"),  
         openai_api_key=os.environ.get("OPENAI_API_KEY"), 
         openai_base_url=os.environ.get("OPENAI_API_BASE"),
         rag_type=os.environ.get("RAG_TYPE", "naive"))  # naive or graph
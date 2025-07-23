"""
pipeline/query_executor.py

This module implements the query executor for DeepSieve.
"""

from pipeline.reasoning_pipeline import (
    plan_subqueries_with_llm,
    route_query_with_llm,
    substitute_variables,
    get_fused_final_answer
)
import time

def run_single_query(
    query_id: int,
    query_info: dict,
    local_rag,
    global_rag,
    merged_rag,
    use_routing: bool,
    use_reflection: bool,
    max_reflexion_times: int,
    decompose: bool,
    local_profile: str,
    global_profile: str,
    model_config: dict
) -> dict:
    query = query_info["query"]
    ground_truth = query_info["ground_truth"]

    print(f"[Query {query_id}] {query}")

    start_time = time.time()
    failed_sources = []
    subquery_answers = []
    subqueries = [query] if not decompose else plan_subqueries_with_llm(query, model_config)

    for i, subquery in enumerate(subqueries):
        subquery_id = f"q{i+1}"
        print(f"[Subquery {subquery_id}] {subquery}")

        if i > 0:
            subquery = substitute_variables(subquery, subquery_answers)
            print(f"[Rewritten] {subquery}")

        if use_routing:
            route_info = route_query_with_llm(
                question=subquery,
                local_profile=local_profile,
                global_profile=global_profile,
                model_config=model_config,
                fail_history=failed_sources,
            )
            source = route_info["source"]
            print(f"[Routing] Source: {source}")

            rag = local_rag if source == "local" else global_rag
        else:
            rag = merged_rag

        retrieved = rag.rag_qa(subquery, model_config)

        if not retrieved.get("success", True):
            failed_sources.append(source if use_routing else "merged")
            print(f"[Failed] Subquery {subquery_id}")
            subquery_answers.append("")
        else:
            subquery_answers.append(retrieved["answer"])

    fused_answer, fused_prompt = get_fused_final_answer(query, subqueries, subquery_answers, model_config)
    print(f"[Fused Answer] {fused_answer}")

    elapsed = time.time() - start_time
    return {
        "query": query,
        "ground_truth": ground_truth,
        "subqueries": subqueries,
        "subquery_answers": subquery_answers,
        "fused_answer": fused_answer,
        "fusion_prompt": fused_prompt,
        "elapsed_time": elapsed
    }

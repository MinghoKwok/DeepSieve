"""
pipeline/reasoning_pipeline.py

This module implements the multi-hop reasoning control pipeline for DeepSieve.
It includes:
- Subquestion decomposition
- Knowledge routing (local vs global)
- Answer fusion
- Variable substitution for query dependencies
"""

import json
from typing import List, Dict
from utils.llm_call import call_openai_chat
from utils.metrics import count_tokens


__all__ = [
    "plan_subqueries_with_llm",
    "route_query_with_llm",
    "get_fused_final_answer",
    "substitute_variables"
]



def plan_subqueries_with_llm(decompose: bool, query: str, api_key: str, model: str, base_url: str) -> dict:
    if decompose == False: 
        # Do not decompose, treat as a single question
        return{"subqueries": [{
                    "id": "q1",
                    "query": query,
                    "depends_on": [],
                    "variables": []
                }]}

    # Decompose the query into a sequence of dependent sub-questions
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

import tiktoken


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

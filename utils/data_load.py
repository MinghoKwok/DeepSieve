import json
from typing import List, Dict, Tuple


def load_queries(dataset: str, sample_size: int = None) -> List[Dict[str, str]]:
    with open(f"data/rag/{dataset}.json", "r") as f:
        data = json.load(f)
    return [
        {"query": item["question"], "ground_truth": item["answer"]}
        for item in data[:sample_size]
    ]


# utils/data_utils.py
def load_corpus_and_profiles(dataset: str) -> Tuple[List[str], List[str], str, str]:
    with open(f"data/rag/{dataset}_corpus_local.json", "r") as f:
        local = [f"{x['title']}. {x['text']}" for x in json.load(f)]
    with open(f"data/rag/{dataset}_corpus_global.json", "r") as f:
        global_ = [f"{x['title']}. {x['text']}" for x in json.load(f)]
    with open(f"data/rag/{dataset}_corpus_profiles.json", "r") as f:
        profiles = json.load(f)
    return local, global_, profiles["local_profile"], profiles["global_profile"]

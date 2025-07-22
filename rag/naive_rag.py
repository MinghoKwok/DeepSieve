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




class NaiveRAG:
    def __init__(self, docs: List[str], embed_model: str = "all-MiniLM-L6-v2"):
        self.docs = docs
        self.embedder = SentenceTransformer(embed_model)
        self.doc_vecs = self.embedder.encode(docs, convert_to_numpy=True)
        print(f"初始化知识库，文档数量: {len(docs)}")

    def rag_qa(self, question: str, k: int = 5) -> Dict:
        start_time = time.time()
        
        # 编码问题
        q_vec = self.embedder.encode([question], convert_to_numpy=True)
        
        # 计算相似度
        scores = cosine_similarity(q_vec, self.doc_vecs)[0]
        topk_idx = np.argsort(scores)[-k:][::-1]
        topk_docs = [self.docs[i] for i in topk_idx]
        topk_scores = [float(scores[i]) for i in topk_idx]
        
        # 计算检索指标
        retrieval_time = time.time() - start_time
        avg_similarity = np.mean(topk_scores)
        max_similarity = np.max(topk_scores)
        
        return {
            "docs": topk_docs,
            "doc_scores": topk_scores,
            "metrics": {
                "retrieval_time": retrieval_time,
                "avg_similarity": avg_similarity,
                "max_similarity": max_similarity,
                "total_docs_searched": len(self.docs)
            }
        }

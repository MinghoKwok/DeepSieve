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

class GraphRAG:
    def __init__(self, docs: List[str], embed_model: str = "all-MiniLM-L6-v2"):
        self.docs = docs
        self.embedder = SentenceTransformer(embed_model)
        self.doc_vecs = self.embedder.encode(docs, convert_to_numpy=True)
        self.doc_map = {doc: f"doc_{i}" for i, doc in enumerate(docs)}  # 添加doc_map
        self.graph = self._build_knowledge_graph()
        print(f"初始化图知识库，文档数量: {len(docs)}")

    def _extract_entities(self, text: str) -> List[str]:
        """从文本中提取实体"""
        # 简单的实体提取：假设大写开头的词组是实体
        entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
        return list(set(entities))

    def _build_knowledge_graph(self) -> nx.Graph:
        """构建知识图谱"""
        G = nx.Graph()
        entity_to_docs = defaultdict(set)
        
        # 为每个文档提取实体并建立映射
        for doc_id, doc in enumerate(self.docs):
            entities = self._extract_entities(doc)
            for entity in entities:
                entity_to_docs[entity].add(self.doc_map[doc])  # 使用doc_map获取doc_id
                G.add_node(entity, type='entity')
                G.add_node(self.doc_map[doc], type='document', text=doc)  # 使用doc_map获取doc_id
                G.add_edge(entity, self.doc_map[doc], weight=1.0)

        # 建立实体间的关系（如果它们出现在同一文档中）
        for entity1 in entity_to_docs:
            for entity2 in entity_to_docs:
                if entity1 < entity2:  # 避免重复边
                    common_docs = entity_to_docs[entity1] & entity_to_docs[entity2]
                    if common_docs:
                        G.add_edge(entity1, entity2, weight=len(common_docs))

        return G

    def _get_relevant_subgraph(self, question: str, k: int = 5) -> List[str]:
        """获取与问题相关的子图并返回相关文档"""
        # 从问题中提取实体
        question_entities = self._extract_entities(question)
        
        # 如果没有找到实体，回退到向量检索
        if not question_entities:
            return self._vector_search(question, k)
            
        # 收集与问题实体相关的所有文档节点
        relevant_docs = set()
        for entity in question_entities:
            if entity in self.graph:
                # 使用个性化PageRank找到最相关的节点
                personalization = {node: 1.0 if node == entity else 0.0 
                                for node in self.graph.nodes()}
                ranks = nx.pagerank(self.graph, personalization=personalization)
                
                # 获取文档节点
                doc_ranks = {node: rank for node, rank in ranks.items() 
                           if node.startswith('doc_')}
                
                # 添加top-k文档
                top_docs = sorted(doc_ranks.items(), key=lambda x: x[1], reverse=True)[:k]
                relevant_docs.update(doc_id for doc_id, _ in top_docs)
        
        # 如果通过图检索没有找到足够的文档，补充向量检索的结果
        if len(relevant_docs) < k:
            vector_docs = self._vector_search(question, k - len(relevant_docs))
            # 使用doc_map安全地获取doc_id
            relevant_docs.update(self.doc_map[d] for d in vector_docs if d in self.doc_map)
        
        # 返回文档内容
        return [self.graph.nodes[doc_id]['text'] 
                for doc_id in list(relevant_docs)[:k]]

    def _vector_search(self, question: str, k: int) -> List[str]:
        """向量检索作为备选方案"""
        q_vec = self.embedder.encode([question], convert_to_numpy=True)
        scores = cosine_similarity(q_vec, self.doc_vecs)[0]
        topk_idx = np.argsort(scores)[-k:][::-1]
        return [self.docs[i] for i in topk_idx]

    def rag_qa(self, question: str, k: int = 5) -> Dict:
        """与NaiveRAG接口兼容的检索方法"""
        start_time = time.time()
        
        # 获取相关文档
        retrieved_docs = self._get_relevant_subgraph(question, k)
        
        # 计算文档相似度分数
        q_vec = self.embedder.encode([question], convert_to_numpy=True)
        doc_vecs = self.embedder.encode(retrieved_docs, convert_to_numpy=True)
        scores = cosine_similarity(q_vec, doc_vecs)[0]
        
        # 计算检索指标
        retrieval_time = time.time() - start_time
        
        return {
            "docs": retrieved_docs,
            "doc_scores": [float(score) for score in scores],
            "metrics": {
                "retrieval_time": retrieval_time,
                "avg_similarity": float(np.mean(scores)),
                "max_similarity": float(np.max(scores)),
                "total_docs_searched": len(self.docs)
            }
        }


class GraphRAG_Improved(GraphRAG):
    def __init__(self, docs: List[str], embed_model: str = "all-MiniLM-L6-v2", max_pr_iter: int = 100):
        """
        改进版GraphRAG
        Args:
            docs: 文档列表
            embed_model: 编码模型名称
            max_pr_iter: PageRank最大迭代次数
        """
        # 在调用父类初始化之前先初始化spaCy
        self.max_pr_iter = max_pr_iter
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
            print("✅ 成功加载spaCy模型")
        except Exception as e:
            print(f"⚠️ spaCy加载失败: {e}，将回退到正则表达式")
            self.use_spacy = False
            
        # 现在调用父类的初始化
        super().__init__(docs, embed_model)

    def _extract_entities(self, text: str) -> List[str]:
        """改进的实体提取，使用spaCy"""
        if self.use_spacy:
            doc = self.nlp(text)
            # 提取命名实体和名词短语
            entities = set()
            # 添加命名实体
            entities.update(ent.text for ent in doc.ents)
            # 添加重要的名词短语
            entities.update(
                chunk.text for chunk in doc.noun_chunks 
                if len(chunk.text.split()) > 1  # 只保留多词短语
                and not all(token.is_stop for token in chunk)  # 排除纯停用词
            )
            return list(entities)
        else:
            # 回退到改进的正则表达式
            # 1. 大写开头词组
            upper_entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
            # 2. 重要的小写词组（至少两个词）
            lower_entities = re.findall(r'\b[a-z]+\s+[a-z]+(?:\s+[a-z]+)*\b', text)
            all_entities = set(upper_entities + lower_entities)
            # 过滤掉常见的无意义词组
            stop_patterns = {'is a', 'was a', 'has been', 'will be', 'can be'}
            return [e for e in all_entities if e.lower() not in stop_patterns]

    def _get_relevant_subgraph(self, question: str, k: int = 5) -> List[str]:
        """改进的子图检索方法"""
        start_time = time.time()
        question_entities = self._extract_entities(question)
        print(f"📌 从问题中提取的实体: {question_entities}")
        
        if not question_entities:
            print("⚠️ 未找到实体，回退到向量检索")
            return self._vector_search(question, k)
            
        relevant_docs = set()
        graph_scores = defaultdict(float)  # 存储图检索得分
        
        for entity in question_entities:
            if entity in self.graph:
                # 限制PageRank最大迭代次数，避免无关扩散
                personalization = {node: 1.0 if node == entity else 0.0 
                                for node in self.graph.nodes()}
                try:
                    ranks = nx.pagerank(self.graph, 
                                      personalization=personalization,
                                      max_iter=self.max_pr_iter)
                    
                    # 获取文档节点
                    doc_ranks = {node: rank for node, rank in ranks.items() 
                               if node.startswith('doc_')}
                    
                    # 累加每个实体的PageRank得分
                    for doc_id, rank in doc_ranks.items():
                        graph_scores[doc_id] += rank
                    
                    print(f"✅ 实体 '{entity}' 成功找到相关文档")
                except Exception as e:
                    print(f"⚠️ PageRank计算失败: {e}")
                    continue
        
        # 如果图检索找到了文档
        if graph_scores:
            # 获取初步的top-k*2文档
            candidate_docs = sorted(graph_scores.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:k*2]
            
            # 使用向量相似度重排序
            texts = [self.graph.nodes[doc_id]["text"] for doc_id, _ in candidate_docs]
            if texts:  # 确保有文档再编码
                doc_vecs = self.embedder.encode(texts, convert_to_numpy=True)
                q_vec = self.embedder.encode([question], convert_to_numpy=True)
                sims = cosine_similarity(q_vec, doc_vecs)[0]
                
                # 综合考虑图得分和向量相似度
                final_scores = [(doc_id, 0.5 * graph_score + 0.5 * sims[i])
                              for i, (doc_id, graph_score) in enumerate(candidate_docs)]
                
                # 取top-k
                top_docs = sorted(final_scores, key=lambda x: x[1], reverse=True)[:k]
                relevant_docs.update(doc_id for doc_id, _ in top_docs)
                
                print(f"📊 图检索找到 {len(relevant_docs)} 个相关文档")
        
        # 如果需要补充文档
        if len(relevant_docs) < k:
            needed = k - len(relevant_docs)
            print(f"⚠️ 图检索文档不足，需要补充 {needed} 个文档")
            
            # 获取向量检索结果
            vector_docs = self._vector_search(question, needed * 2)  # 多检索一些候选
            
            # 计算向量相似度
            vector_vecs = self.embedder.encode(vector_docs, convert_to_numpy=True)
            q_vec = self.embedder.encode([question], convert_to_numpy=True)
            sims = cosine_similarity(q_vec, vector_vecs)[0]
            
            # 按相似度排序并过滤已有文档
            scored_docs = [(doc, sim) for doc, sim in zip(vector_docs, sims)
                          if self.doc_map[doc] not in relevant_docs]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # 添加top-needed文档
            fallback_docs = {self.doc_map[doc] for doc, _ in scored_docs[:needed]}
            relevant_docs.update(fallback_docs)
            print(f"📊 向量检索补充了 {len(fallback_docs)} 个文档")
        
        retrieval_time = time.time() - start_time
        print(f"⏱️ 总检索时间: {retrieval_time:.2f}秒")
        
        return [self.graph.nodes[doc_id]['text'] 
                for doc_id in list(relevant_docs)[:k]]

    def _vector_search(self, question: str, k: int) -> List[str]:
        """改进的向量检索，增加调试信息"""
        start_time = time.time()
        q_vec = self.embedder.encode([question], convert_to_numpy=True)
        scores = cosine_similarity(q_vec, self.doc_vecs)[0]
        topk_idx = np.argsort(scores)[-k:][::-1]
        topk_scores = scores[topk_idx]
        
        print(f"📊 向量检索得分范围: {topk_scores.min():.3f} - {topk_scores.max():.3f}")
        retrieval_time = time.time() - start_time
        print(f"⏱️ 向量检索时间: {retrieval_time:.2f}秒")
        
        return [self.docs[i] for i in topk_idx]

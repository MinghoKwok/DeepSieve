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

# LLM 调用

def call_openai_chat(prompt: str, api_key: str, model: str, base_url: str, max_retries: int = 3) -> str:
    """调用 OpenAI Chat API 的函数，包含重试机制

    Args:
        prompt: 提示文本
        api_key: API密钥
        model: 模型名称
        base_url: API基础URL
        max_retries: 最大重试次数

    Returns:
        str: API响应的文本内容
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
        backoff_factor=2,  # 增加退避时间
        status_forcelist=[500, 502, 503, 504, 408, 429],  # 添加超时和限流状态码
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
        print(f"🔴 API请求错误: {str(e)}")
        if isinstance(e, (requests.exceptions.ChunkedEncodingError, requests.exceptions.ReadTimeout)):
            print("检测到连接错误，正在重试...")
            # 对于连接错误，我们特别处理
            for i in range(max_retries):
                try:
                    print(f"重试 #{i+1}...")
                    time.sleep(2 ** i)  # 指数退避
                    response = session.post(url, headers=headers, json=payload, timeout=60)
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
                except requests.exceptions.RequestException as retry_e:
                    print(f"重试 #{i+1} 失败: {str(retry_e)}")
                    if i == max_retries - 1:  # 如果是最后一次重试
                        print("所有重试都失败了")
                        return ""
    except Exception as e:
        print(f"🔴 其他错误: {str(e)}")
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
        # 清理响应中可能存在的markdown代码块标记
        cleaned_response = str(response).strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        result = json.loads(cleaned_response)
        if "subqueries" not in result:
            print("⚠️ 响应中缺少subqueries字段:")
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
    替换查询中的变量为其实际值
    例如：将 "What country is [birthplace] in?" 中的 [birthplace] 替换为实际值
    """
    result = query
    for var_name, value in variable_values.items():
        result = result.replace(f"[{var_name}]", value)
    return result


def route_query_with_llm(query: str, local_profile: str, global_profile: str,
                         api_key: str, model: str, base_url: str, fail_history: str) -> str:
    """路由查询到合适的知识库

    Args:
        query: 查询文本
        local_profile: 本地知识库描述
        global_profile: 全局知识库描述
        api_key: API密钥
        model: 模型名称
        base_url: API基础URL
        fail_history: 失败历史

    Returns:
        str: 路由结果 ("local" 或 "global")
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
            print("⚠️ 路由响应为空，默认使用local路由")
            return "local"
        
        route = response.strip().lower()
        if route not in {"local", "global"}:
            print(f"⚠️ 意外的路由输出: {route}，默认使用local路由")
            return "local"
        return route
    except Exception as e:
        print(f"⚠️ 路由过程出错: {str(e)}，默认使用local路由")
        return "local"



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


class RaptorRAG:
    def __init__(self, 
                 docs: List[str], 
                 embed_model: str = "all-MiniLM-L6-v2", 
                 max_tokens_per_node: int = 1000,
                 use_summary: bool = True,
                 use_summary_for_embedding: bool = True,
                 summary_batch_size: int = 5):
        """初始化 RAPTOR 检索系统
        
        Args:
            docs: 文档列表
            embed_model: 编码模型名称
            max_tokens_per_node: 每个节点的最大token数
            use_summary: 是否使用摘要功能
            use_summary_for_embedding: 是否使用摘要计算embedding
            summary_batch_size: 每批处理的摘要数量
        """
        self.docs = docs
        self.embedder = SentenceTransformer(embed_model)
        self.max_tokens_per_node = max_tokens_per_node
        self.use_summary = use_summary
        self.use_summary_for_embedding = use_summary_for_embedding
        self.summary_batch_size = summary_batch_size
        self.tree = self._build_tree()
        self.summaries_updated = False
        print(f"✅ 初始化 RAPTOR 知识库，文档数量: {len(docs)}")
        print(f"📊 配置：使用摘要={use_summary}, 摘要embedding={use_summary_for_embedding}, 批量大小={summary_batch_size}")
        
    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        try:
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
        
    def _chunk_text(self, text: str) -> List[str]:
        """将文本分块，确保每块不超过最大token数"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        sentences = re.split('([.!?。！？] )', text)
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]  # 添加分隔符
                
            sentence_length = self._count_tokens(sentence)
            
            if current_length + sentence_length > self.max_tokens_per_node:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def _generate_summary(self, texts: List[str], api_key: str = None, model: str = None, base_url: str = None) -> str:
        """生成文本摘要"""
        if not texts:
            return ""
            
        combined_text = " ".join(texts)
        # 简化提示模板
        prompt = f"""Please summarize the following text in 3 sentences:

{combined_text}"""
        
        if api_key and model and base_url:
            try:
                print(f"🔄 正在为长度为 {len(combined_text)} 的文本生成摘要...")
                summary = call_openai_chat(prompt, api_key, model, base_url)
                if not summary:
                    print("⚠️ API 返回空摘要，使用提取式摘要")
                    return self._extract_summary(combined_text)
                return summary.strip()
            except Exception as e:
                print(f"⚠️ 生成摘要时出错: {e}")
                return self._extract_summary(combined_text)
        else:
            return self._extract_summary(combined_text)
            
    def _extract_summary(self, text: str, max_length: int = 200) -> str:
        """提取式摘要方法（作为生成式摘要的降级方案）
        
        Args:
            text: 源文本
            max_length: 最大摘要长度
            
        Returns:
            str: 提取的摘要
        """
        # 1. 分句
        sentences = re.split('[.!?。！？] ', text)
        if not sentences:
            return text[:max_length] + "..."
            
        # 2. 计算每句话的重要性分数（这里用简单的长度和位置加权）
        scores = []
        for i, sent in enumerate(sentences):
            length_score = min(len(sent.split()) / 20, 1.0)  # 长度分数
            position_score = 1.0 - (i / len(sentences))  # 位置分数
            scores.append(length_score * 0.5 + position_score * 0.5)
            
        # 3. 选择最重要的句子
        sorted_sents = [sent for _, sent in sorted(zip(scores, sentences), reverse=True)]
        summary = " ".join(sorted_sents[:3])  # 取top3句子
        
        return summary[:max_length] + ("..." if len(summary) > max_length else "")

    def _build_tree(self) -> Dict:
        """构建文档树结构"""
        tree = {
            "root": {
                "content": "",
                "summary": "",
                "children": [],
                "embedding": None,
                "id": "root"
            }
        }
        
        # 第一层：将文档分组
        for i, doc in enumerate(self.docs):
            chunks = self._chunk_text(doc)
            doc_node = {
                "content": doc,
                "summary": "",
                "children": [],
                "embedding": None,  # 将在后续更新
                "id": f"doc_{i}"
            }
            
            # 第二层：文档块
            for j, chunk in enumerate(chunks):
                chunk_node = {
                    "content": chunk,
                    "summary": "",
                    "children": [],
                    "embedding": self.embedder.encode(chunk, convert_to_numpy=True),
                    "id": f"doc_{i}_chunk_{j}"
                }
                doc_node["children"].append(chunk_node)
            
            # 计算文档级别的embedding
            if self.use_summary and self.use_summary_for_embedding and doc_node["summary"]:
                doc_node["embedding"] = self.embedder.encode(doc_node["summary"], convert_to_numpy=True)
            else:
                doc_node["embedding"] = self.embedder.encode(doc_node["content"], convert_to_numpy=True)
            
            tree["root"]["children"].append(doc_node)
            
        return tree

    def _recursive_search(self, query_vec: np.ndarray, node: Dict, top_k: int, results: List[tuple]) -> None:
        """递归搜索相关内容
        
        Args:
            query_vec: 查询向量
            node: 当前节点
            top_k: 返回结果数量
            results: 结果列表，每个元素为 (node_id, content, score) 三元组
        """
        # 如果节点有 embedding，计算相似度
        if node.get("embedding") is not None:
            score = float(cosine_similarity(query_vec.reshape(1, -1), 
                                         node["embedding"].reshape(1, -1))[0][0])
            results.append((node["id"], node["content"], score))
            
        # 递归搜索子节点
        for child in node.get("children", []):
            self._recursive_search(query_vec, child, top_k, results)

    def rag_qa(self, question: str, k: int = 5) -> Dict:
        """与 NaiveRAG 兼容的检索接口
        
        Args:
            question: 查询问题
            k: 返回结果数量
            
        Returns:
            Dict: 包含检索结果和性能指标的字典
        """
        start_time = time.time()
        
        # 编码问题
        q_vec = self.embedder.encode([question], convert_to_numpy=True)[0]
        
        # 收集检索结果
        results = []
        self._recursive_search(q_vec, self.tree["root"], k, results)
        
        # 排序并获取 top-k
        results.sort(key=lambda x: x[2], reverse=True)
        top_k_results = results[:k]
        
        # 计算检索指标
        retrieval_time = time.time() - start_time
        scores = [score for _, _, score in top_k_results]
        
        return {
            "docs": [{"id": doc_id, "text": content} for doc_id, content, _ in top_k_results],
            "doc_scores": scores,
            "metrics": {
                "retrieval_time": retrieval_time,
                "avg_similarity": float(np.mean(scores)),
                "max_similarity": float(np.max(scores)),
                "total_docs_searched": len(self.docs)
            }
        }

    def update_summaries(self, api_key: str, model: str, base_url: str) -> None:
        """更新树中所有节点的摘要"""
        if not self.use_summary:
            print("⚠️ 摘要功能未启用，跳过更新")
            return
            
        def recursive_update(node):
            if node.get("children"):
                # 分批处理子节点
                children = node.get("children", [])
                for i in range(0, len(children), self.summary_batch_size):
                    batch = children[i:i + self.summary_batch_size]
                    print(f"📝 处理第 {i//self.summary_batch_size + 1} 批，共 {len(batch)} 个节点")
                    
                    # 为每个节点生成摘要
                    for child in batch:
                        if child.get("children"):
                            child_contents = [c["content"] for c in child["children"]]
                            child["summary"] = self._generate_summary(child_contents, api_key, model, base_url)
                            
                            # 如果配置为使用摘要计算embedding，则更新embedding
                            if self.use_summary_for_embedding and child["summary"]:
                                child["embedding"] = self.embedder.encode(child["summary"], convert_to_numpy=True)
                                
                            # 递归处理子节点
                            recursive_update(child)
                    
                    # 每批处理完后暂停一下
                    if i + self.summary_batch_size < len(children):
                        print("⏳ 暂停1秒后处理下一批...")
                        time.sleep(1)
        
        print("📝 开始更新树结构摘要...")
        recursive_update(self.tree["root"])
        self.summaries_updated = True
        print("✅ 摘要更新完成")


# 主程序

def normalize_answer(s: str) -> str:
    """
    规范化答案字符串，用于比较
    1. 转换为小写
    2. 移除标点符号和多余空格
    3. 移除冠词(a, an, the)等停用词
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
    计算Exact Match分数
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    计算F1分数
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
    评估预测答案的质量
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
    计算所有查询的平均性能指标
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
    """将不同数据集的DataFrame转换为统一的查询格式"""
    if dataset_name in ["hotpot_qa", "trivia_qa", "gsm8k", "sotu_qa"]:
        # 这些数据集已经使用 question/answer 格式
        return [{"query": row["question"], "ground_truth": row["answer"]} 
                for _, row in df.iterrows()]
    elif dataset_name in ["physics_question", "sports_understanding", "disfl_qa", "strategy_qa"]:
        # 这些数据集使用 input/target 格式
        return [{"query": row["input"], "ground_truth": row["target"]} 
                for _, row in df.iterrows()]
    elif dataset_name == "fever":
        # FEVER数据集使用 claim/label 格式
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
    主函数
    Args:
        decompose: 是否分解查询
        use_routing: 是否使用路由
        use_reflection: 是否使用反思机制
        max_reflexion_times: 最大反思次数
        dataset: 数据集名称
        sample_size: 样本大小
        openai_model: OpenAI模型名称
        openai_api_key: OpenAI API密钥
        openai_base_url: OpenAI API基础URL
        rag_type: RAG类型，可选值为"naive"、"graph"或"raptor"
    """
    # 定义多个查询和对应的ground truth
    # 加载数据
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

    # 准备知识库文档
    with open(f"data/rag/{dataset}_corpus_local.json", "r") as f:
        data = json.load(f)
    local_docs = [f"{item['title']}. {item['text']}" for item in data]
    print(f"✅ Loaded {len(local_docs)} documents into local_docs.")

    with open(f"data/rag/{dataset}_corpus_global.json", "r") as f:
        data = json.load(f)
    global_docs = [f"{item['title']}. {item['text']}" for item in data]
    print(f"✅ Loaded {len(global_docs)} documents into global_docs.")

    # 读取 profiles.json
    with open(f"data/rag/{dataset}_corpus_profiles.json", "r") as f:
        profiles = json.load(f)
    local_profile = profiles["local_profile"]
    global_profile = profiles["global_profile"]
    print(f"✅ Loaded local_profile and global_profile.")

    merged_docs = local_docs + global_docs

    # 初始化RAG系统
    if use_routing:
        if rag_type == "naive":
            local_rag = NaiveRAG(local_docs)
            global_rag = NaiveRAG(global_docs)
        elif rag_type == "graph":
            local_rag = GraphRAG_Improved(local_docs)
            global_rag = GraphRAG_Improved(global_docs)
        elif rag_type == "raptor":
            # 从环境变量获取配置
            use_summary = os.environ.get("RAPTOR_USE_SUMMARY", "true").lower() == "true"
            use_summary_for_embedding = os.environ.get("RAPTOR_USE_SUMMARY_EMBEDDING", "true").lower() == "true"
            summary_batch_size = int(os.environ.get("RAPTOR_SUMMARY_BATCH_SIZE", "1"))
            
            print(f"🔧 RAPTOR配置：摘要={use_summary}, 摘要向量={use_summary_for_embedding}, 批量={summary_batch_size}")
            
            local_rag = RaptorRAG(
                docs=local_docs,
                use_summary=use_summary,
                use_summary_for_embedding=use_summary_for_embedding,
                summary_batch_size=summary_batch_size
            )
            global_rag = RaptorRAG(
                docs=global_docs,
                use_summary=use_summary,
                use_summary_for_embedding=use_summary_for_embedding,
                summary_batch_size=summary_batch_size
            )
            
            if use_summary:
                print("正在生成本地知识库摘要...")
                local_rag.update_summaries(openai_api_key, openai_model, openai_base_url)
                print("正在生成全局知识库摘要...")
                global_rag.update_summaries(openai_api_key, openai_model, openai_base_url)
        else:
            raise ValueError(f"不支持的 RAG 类型: {rag_type}")
        print(f"🔍 使用路由模式：分别初始化local和global知识库，RAG类型：{rag_type}")
    else:
        # 合并数据集
        if rag_type == "naive":
            merged_rag = NaiveRAG(merged_docs)
        elif rag_type == "graph":
            merged_rag = GraphRAG_Improved(merged_docs)
        elif rag_type == "raptor":
            # 从环境变量获取配置
            use_summary = os.environ.get("RAPTOR_USE_SUMMARY", "false").lower() == "true"
            use_summary_for_embedding = os.environ.get("RAPTOR_USE_SUMMARY_EMBEDDING", "false").lower() == "true"
            summary_batch_size = int(os.environ.get("RAPTOR_SUMMARY_BATCH_SIZE", "1"))
            
            print(f"🔧 RAPTOR配置：摘要={use_summary}, 摘要向量={use_summary_for_embedding}, 批量={summary_batch_size}")
            
            merged_rag = RaptorRAG(
                docs=merged_docs,
                use_summary=use_summary,
                use_summary_for_embedding=use_summary_for_embedding,
                summary_batch_size=summary_batch_size
            )
            
            if use_summary:
                print("正在生成合并知识库摘要...")
                merged_rag.update_summaries(openai_api_key, openai_model, openai_base_url)
        else:
            raise ValueError(f"不支持的 RAG 类型: {rag_type}")
        print(f"🔍 使用无路由模式：合并local和global知识库，RAG类型：{rag_type}")

    all_metrics = []  # 存储所有查询的性能指标

    # 处理每个查询
    for idx, query_info in enumerate(queries_and_truth, 1):
        multi_hop_query = query_info["query"]
        ground_truth = query_info["ground_truth"]
        
        print(f"\n📝 处理查询 {idx}/{len(queries_and_truth)}:")
        print(f"Query: {multi_hop_query}")
        print(f"Ground Truth: {ground_truth}")
        
        # 初始化variable_values字典
        variable_values = {}
        
        if decompose:
            # 获取带依赖关系的子查询计划
            query_plan = plan_subqueries_with_llm(multi_hop_query, openai_api_key, openai_model, openai_base_url)
            if not query_plan or not query_plan["subqueries"]:
                print("❌ 子问题规划失败，跳过当前查询。")
                continue
        else:
            # 不分解查询，直接作为单个问题处理
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

        # 按顺序处理子查询，处理依赖关系
        for subquery_info in query_plan["subqueries"]:
            subquery_start_time = time.time()
            subquery_id = subquery_info["id"]
            original_query = subquery_info["query"]
            
            # 检查并等待所有依赖项完成
            if subquery_info["depends_on"]:
                print(f"\n⏳ 处理查询 {subquery_id} 的依赖项: {subquery_info['depends_on']}")
                
            # 替换查询中的变量
            current_variables = {}
            for var in subquery_info.get("variables", []):
                var_name = var["name"]
                source_query = var["source_query"]
                if source_query not in variable_values:
                    print(f"❌ 错误：查询 {subquery_id} 依赖于未完成的查询 {source_query}")
                    continue
                current_variables[var_name] = variable_values[source_query]
            
            # 替换变量后的实际查询
            actual_query = substitute_variables(original_query, current_variables)
            print(f"\n🔍 处理查询 {subquery_id}: {actual_query}")
            print(f"原始查询: {original_query}")
            if current_variables:
                print(f"替换变量: {current_variables}")

            # Loop for reflexion
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
                    
                    # 收集性能指标
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
                        
                        # 存储答案供后续查询使用
                        if success == 1:
                            variable_values[subquery_id] = answer
                            print(f"提取的答案: {answer}")
                            print(f"推理过程: {reason}")
                            print(f"是否成功: {success}")
                        else:
                            variable_values[subquery_id] = ""
                            fail_history += f"Fail History: Last routing failed because {reason}. Last routing result is {route}. So please try another routing choice, don't choose {route} again."
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"⚠️ 解析答案失败: {str(e)}")
                        print(f"原始响应: {response}")
                        answer, reason = f"Error: {str(e)}", ""
                        success = 0
                    
                    # 更新token统计
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
                    print(f"⚠️ 处理查询时发生错误: {str(e)}")
                    answer, retrieved = f"Error: {str(e)}", {"docs": [], "doc_scores": []}
                    success = 0  # 设置success为0

                if success == 1 or use_routing == False or use_reflection == False or left_reflexion_times <= 0:
                    break   # 如果成功或者不使用路由或者不使用反射或者反射次数用完，则跳出循环

        # 计算汇总指标
        performance_metrics["avg_retrieval_time"] = performance_metrics["total_retrieval_time"] / len(query_plan["subqueries"])
        performance_metrics["avg_similarity"] = np.mean(performance_metrics["avg_similarity_scores"])
        performance_metrics["max_similarity"] = np.max(performance_metrics["max_similarity_scores"])
        
        # 计算token统计指标
        performance_metrics["avg_prompt_tokens"] = performance_metrics["total_prompt_tokens"] / len(query_plan["subqueries"])
        performance_metrics["max_prompt_tokens"] = max(performance_metrics["prompt_token_counts"], default=0)
        performance_metrics["min_prompt_tokens"] = min(performance_metrics["prompt_token_counts"], default=0)

        # 获取 fallback（最后一跳）的答案（用于对比分析）
        fallback_answer = results[-1]["answer"] if results else ""
        performance_metrics["evaluation_metrics"]["fallback_answer"] = fallback_answer


        # 获取最终答案（融合所有 reasoning 步骤）
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

        
        # 计算评估指标
        eval_results = evaluate_answer(final_answer, ground_truth)
        eval_results_fallback = evaluate_answer(fallback_answer, ground_truth)
        performance_metrics["evaluation_metrics"].update(eval_results)
        performance_metrics["evaluation_metrics_fallback"].update(eval_results_fallback)
        
        # 保存当前查询的结果
        query_results_path = os.path.join(save_dir, f"query_{idx}_results.jsonl")
        with open(query_results_path, "w") as f:
            # 第1条：Query metadata
            f.write(json.dumps({
                "type": "query_info",
                "query": multi_hop_query,
                "ground_truth": ground_truth
            }) + "\n")

            # 第2条：Final answer summary
            f.write(json.dumps({
                "type": "final_answer",
                "final_answer": final_answer,
                "final_reason": final_reason,
                "fusion_prompt_tokens": fusion_token_count,
                "fallback_answer": fallback_answer,
                "fusion_equals_fallback": fallback_answer.strip().lower() == final_answer.strip().lower()
            }) + "\n")

            # 第3条：Evaluation metrics
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

            # 第4条：Performance metrics
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

            # 第5条：Subquery performance details
            for metrics in performance_metrics["subquery_metrics"]:
                f.write(json.dumps({
                    "type": "subquery_metric",
                    "subquery_id": metrics["subquery_id"],
                    "retrieval_time": metrics["retrieval_time"],
                    "docs_searched": metrics["docs_searched"],
                    "avg_similarity": metrics["avg_similarity"],
                    "max_similarity": metrics["max_similarity"]
                }) + "\n")

            # 第6条：Execution results (per subquery)
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

            # 第7条：Answer chain
            for step in fused_answer_texts:
                f.write(json.dumps({
                    "type": "fused_answer_step",
                    "text": step
                }) + "\n")        
        # 保存融合提示
        fusion_prompt_path = os.path.join(save_dir, f"query_{idx}_fusion_prompt.txt")
        with open(fusion_prompt_path, "w") as f_prompt:
            f_prompt.write(fusion_prompt)


        all_metrics.append(performance_metrics)

    # 计算并保存整体性能指标
    overall_metrics = calculate_overall_metrics(all_metrics)
    
    # 保存整体结果
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

    # 更新控制台输出
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
    # 默认使用分解和路由模式
    main(decompose=True, 
         use_routing=True, 
         use_reflection=True, 
         max_reflexion_times=2, 
         dataset="hotpot_qa", 
         sample_size=5, 
         openai_model=os.environ.get("OPENAI_MODEL"),  
         openai_api_key=os.environ.get("OPENAI_API_KEY"), 
         openai_base_url=os.environ.get("OPENAI_API_BASE"),
         rag_type=os.environ.get("RAG_TYPE", "naive"))  # naive or others
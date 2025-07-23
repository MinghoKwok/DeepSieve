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
        self.doc_map = {doc: f"doc_{i}" for i, doc in enumerate(docs)}  # Add doc_map
        self.graph = self._build_knowledge_graph()
        print(f"Initialized graph knowledge base, number of documents: {len(docs)}")

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text"""
        # Simple entity extraction: assume capitalized phrases are entities
        entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
        return list(set(entities))

    def _build_knowledge_graph(self) -> nx.Graph:
        """Build knowledge graph"""
        G = nx.Graph()
        entity_to_docs = defaultdict(set)
        
        # Extract entities for each document and build mapping
        for doc_id, doc in enumerate(self.docs):
            entities = self._extract_entities(doc)
            for entity in entities:
                entity_to_docs[entity].add(self.doc_map[doc])  # Use doc_map to get doc_id
                G.add_node(entity, type='entity')
                G.add_node(self.doc_map[doc], type='document', text=doc)  # Use doc_map to get doc_id
                G.add_edge(entity, self.doc_map[doc], weight=1.0)

        # Build relationships between entities (if they appear in the same document)
        for entity1 in entity_to_docs:
            for entity2 in entity_to_docs:
                if entity1 < entity2:  # Avoid duplicate edges
                    common_docs = entity_to_docs[entity1] & entity_to_docs[entity2]
                    if common_docs:
                        G.add_edge(entity1, entity2, weight=len(common_docs))

        return G

    def _get_relevant_subgraph(self, question: str, k: int = 5) -> List[str]:
        """Get relevant subgraph and return relevant documents"""
        # Extract entities from the question
        question_entities = self._extract_entities(question)
        
        # If no entities found, fallback to vector search
        if not question_entities:
            return self._vector_search(question, k)
        
        # Collect all document nodes related to question entities
        relevant_docs = set()
        for entity in question_entities:
            if entity in self.graph:
                # Use personalized PageRank to find most relevant nodes
                personalization = {node: 1.0 if node == entity else 0.0 
                                for node in self.graph.nodes()}
                ranks = nx.pagerank(self.graph, personalization=personalization)
                
                # Get document nodes
                doc_ranks = {node: rank for node, rank in ranks.items() 
                           if node.startswith('doc_')}
                
                # Add top-k documents
                top_docs = sorted(doc_ranks.items(), key=lambda x: x[1], reverse=True)[:k]
                relevant_docs.update(doc_id for doc_id, _ in top_docs)
        
        # If not enough documents found by graph retrieval, supplement with vector search results
        if len(relevant_docs) < k:
            vector_docs = self._vector_search(question, k - len(relevant_docs))
            # Use doc_map to safely get doc_id
            relevant_docs.update(self.doc_map[d] for d in vector_docs if d in self.doc_map)
        
        # Return document contents
        return [self.graph.nodes[doc_id]['text'] 
                for doc_id in list(relevant_docs)[:k]]

    def _vector_search(self, question: str, k: int) -> List[str]:
        """Vector search as a fallback"""
        q_vec = self.embedder.encode([question], convert_to_numpy=True)
        scores = cosine_similarity(q_vec, self.doc_vecs)[0]
        topk_idx = np.argsort(scores)[-k:][::-1]
        return [self.docs[i] for i in topk_idx]

    def rag_qa(self, question: str, k: int = 5) -> Dict:
        """Retrieval method compatible with NaiveRAG interface"""
        start_time = time.time()
        
        # Get relevant documents
        retrieved_docs = self._get_relevant_subgraph(question, k)
        
        # Compute document similarity scores
        q_vec = self.embedder.encode([question], convert_to_numpy=True)
        doc_vecs = self.embedder.encode(retrieved_docs, convert_to_numpy=True)
        scores = cosine_similarity(q_vec, doc_vecs)[0]
        
        # Compute retrieval metrics
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
        Improved GraphRAG
        Args:
            docs: List of documents
            embed_model: Encoder model name
            max_pr_iter: Max PageRank iterations
        """
        # Initialize spaCy before calling parent init
        self.max_pr_iter = max_pr_iter
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
            print("✅ spaCy model loaded successfully")
        except Exception as e:
            print(f"⚠️ spaCy loading failed: {e}, falling back to regex")
            self.use_spacy = False
        
        # Now call parent init
        super().__init__(docs, embed_model)

    def _extract_entities(self, text: str) -> List[str]:
        """Improved entity extraction using spaCy"""
        if self.use_spacy:
            doc = self.nlp(text)
            # Extract named entities and noun phrases
            entities = set()
            # Add named entities
            entities.update(ent.text for ent in doc.ents)
            # Add important noun phrases
            entities.update(
                chunk.text for chunk in doc.noun_chunks 
                if len(chunk.text.split()) > 1  # Only keep multi-word phrases
                and not all(token.is_stop for token in chunk)  # Exclude stopwords only
            )
            return list(entities)
        else:
            # Fallback to improved regex
            # 1. Capitalized phrases
            upper_entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
            # 2. Important lowercase phrases (at least two words)
            lower_entities = re.findall(r'\b[a-z]+\s+[a-z]+(?:\s+[a-z]+)*\b', text)
            all_entities = set(upper_entities + lower_entities)
            # Filter out common meaningless phrases
            stop_patterns = {'is a', 'was a', 'has been', 'will be', 'can be'}
            return [e for e in all_entities if e.lower() not in stop_patterns]

    def _get_relevant_subgraph(self, question: str, k: int = 5) -> List[str]:
        """Improved subgraph retrieval method"""
        start_time = time.time()
        question_entities = self._extract_entities(question)
        print(f"📌 Entities extracted from question: {question_entities}")
        
        if not question_entities:
            print("⚠️ No entities found, falling back to vector search")
            return self._vector_search(question, k)
        
        relevant_docs = set()
        graph_scores = defaultdict(float)  # Store graph retrieval scores
        
        for entity in question_entities:
            if entity in self.graph:
                # Limit PageRank max iterations to avoid irrelevant diffusion
                personalization = {node: 1.0 if node == entity else 0.0 
                                for node in self.graph.nodes()}
                try:
                    ranks = nx.pagerank(self.graph, 
                                      personalization=personalization,
                                      max_iter=self.max_pr_iter)
                    
                    # Get document nodes
                    doc_ranks = {node: rank for node, rank in ranks.items() 
                               if node.startswith('doc_')}
                    
                    # Accumulate PageRank scores for each entity
                    for doc_id, rank in doc_ranks.items():
                        graph_scores[doc_id] += rank
                    
                    print(f"✅ Entity '{entity}' found relevant documents")
                except Exception as e:
                    print(f"⚠️ PageRank calculation failed: {e}")
                    continue
        
        # If graph retrieval found documents
        if graph_scores:
            # Get initial top-k*2 documents
            candidate_docs = sorted(graph_scores.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:k*2]
            
            # Re-rank using vector similarity
            texts = [self.graph.nodes[doc_id]["text"] for doc_id, _ in candidate_docs]
            if texts:  # Ensure there are documents to encode
                doc_vecs = self.embedder.encode(texts, convert_to_numpy=True)
                q_vec = self.embedder.encode([question], convert_to_numpy=True)
                sims = cosine_similarity(q_vec, doc_vecs)[0]
                
                # Combine graph score and vector similarity
                final_scores = [(doc_id, 0.5 * graph_score + 0.5 * sims[i])
                              for i, (doc_id, graph_score) in enumerate(candidate_docs)]
                
                # Take top-k
                top_docs = sorted(final_scores, key=lambda x: x[1], reverse=True)[:k]
                relevant_docs.update(doc_id for doc_id, _ in top_docs)
                
                print(f"📊 Graph retrieval found {len(relevant_docs)} relevant documents")
        
        # Supplement documents if needed
        if len(relevant_docs) < k:
            needed = k - len(relevant_docs)
            print(f"⚠️ Not enough documents from graph retrieval, supplementing {needed} documents")
            
            # Get vector search results
            vector_docs = self._vector_search(question, needed * 2)  # Retrieve more candidates
            
            # Compute vector similarity
            vector_vecs = self.embedder.encode(vector_docs, convert_to_numpy=True)
            q_vec = self.embedder.encode([question], convert_to_numpy=True)
            sims = cosine_similarity(q_vec, vector_vecs)[0]
            
            # Sort by similarity and filter out existing documents
            scored_docs = [(doc, sim) for doc, sim in zip(vector_docs, sims)
                          if self.doc_map[doc] not in relevant_docs]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Add top-needed documents
            fallback_docs = {self.doc_map[doc] for doc, _ in scored_docs[:needed]}
            relevant_docs.update(fallback_docs)
            print(f"📊 Vector search supplemented {len(fallback_docs)} documents")
        
        retrieval_time = time.time() - start_time
        print(f"⏱️ Total retrieval time: {retrieval_time:.2f}s")
        
        return [self.graph.nodes[doc_id]['text'] 
                for doc_id in list(relevant_docs)[:k]]

    def _vector_search(self, question: str, k: int) -> List[str]:
        """Improved vector search with debug info"""
        start_time = time.time()
        q_vec = self.embedder.encode([question], convert_to_numpy=True)
        scores = cosine_similarity(q_vec, self.doc_vecs)[0]
        topk_idx = np.argsort(scores)[-k:][::-1]
        topk_scores = scores[topk_idx]
        
        print(f"📊 Vector search score range: {topk_scores.min():.3f} - {topk_scores.max():.3f}")
        retrieval_time = time.time() - start_time
        print(f"⏱️ Vector search time: {retrieval_time:.2f}s")
        
        return [self.docs[i] for i in topk_idx]

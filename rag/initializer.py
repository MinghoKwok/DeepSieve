"""
rag/initializer.py

This module initializes the RAG system based on the specified RAG type and routing mode.
"""

from rag.naive_rag import NaiveRAG
from rag.graph_rag import GraphRAG_Improved

def initialize_rag_system(rag_type: str, use_routing: bool, local_docs: list, global_docs: list):
    merged_docs = local_docs + global_docs

    if use_routing:
        if rag_type == "naive":
            local_rag = NaiveRAG(local_docs)
            global_rag = NaiveRAG(global_docs)
        elif rag_type == "graph":
            local_rag = GraphRAG_Improved(local_docs)
            global_rag = GraphRAG_Improved(global_docs)
        else:
            raise ValueError(f"Unsupported RAG type: {rag_type}")
        
        print(f"🔍 Using routing mode: initialized local and global knowledge bases, RAG type: {rag_type}")
        return local_rag, global_rag, None

    else:
        if rag_type == "naive":
            merged_rag = NaiveRAG(merged_docs)
        elif rag_type == "graph":
            merged_rag = GraphRAG_Improved(merged_docs)
        else:
            raise ValueError(f"Unsupported RAG type: {rag_type}")
        
        print(f"🔍 Using no-routing mode: merged local and global knowledge bases, RAG type: {rag_type}")
        return None, None, merged_rag

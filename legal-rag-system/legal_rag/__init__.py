"""
Legal RAG System - Advanced Retrieval-Augmented Generation for Legal Documents

A modular, production-ready RAG system optimized for legal documents with
hierarchical chunking, hybrid search (vector + BM25), and legal-specific prompting.

Usage:
    from legal_rag import LegalRAGPipeline
    pipeline = LegalRAGPipeline()
    pipeline.ingest_document(text="...", title="Contract")
    result = pipeline.query("What are the termination conditions?")
"""

__version__ = "1.0.0"
__author__ = "Legal RAG Team"
__license__ = "MIT"

from .config import LegalRAGConfig
from .pipeline import LegalRAGPipeline
from .document_processor import LegalDocumentProcessor
from .retrievers import ParentChildRetriever, LegalBM25Retriever
from .llm_provider import LLMProvider
from .embeddings import EmbeddingManager
from .vector_store import VectorStoreManager
from .prompts import LegalPromptTemplates

__all__ = [
    "LegalRAGConfig",
    "LegalRAGPipeline",
    "LegalDocumentProcessor",
    "ParentChildRetriever",
    "LegalBM25Retriever",
    "LLMProvider",
    "EmbeddingManager",
    "VectorStoreManager",
    "LegalPromptTemplates",
]

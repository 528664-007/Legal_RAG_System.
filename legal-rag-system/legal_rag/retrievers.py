"""
Advanced retrievers for legal document search.
Implements BM25 and Hybrid retrieval for legal documents.
"""

from typing import List, Optional, Tuple, Dict, Any
import nltk
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from rich.console import Console

console = Console()

# NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class LegalBM25Retriever(BaseRetriever):
    """
    BM25 retriever optimized for legal documents.
    
    Performs keyword-based matching essential for legal research:
    - Exact legal terms and definitions (e.g., "Confidential Information")
    - Statutory citations (e.g., "Section 3426.1", "Article 2.3")
    - Case references and party names
    """
    
    # Pydantic fields
    documents: List[Document] = []
    k: int = 5
    
    # Internal fields (not validated by Pydantic)
    _tokenized_corpus: List = []
    _bm25: Any = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        documents: List[Document],
        k: int = 5,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        Initialize BM25 retriever with legal-optimized parameters.
        
        Args:
            documents: List of documents to index and search
            k: Number of top documents to retrieve
            k1: Term frequency saturation parameter (1.2-2.0 typical)
            b: Length normalization parameter (0.5-0.8 typical)
            epsilon: IDF smoothing parameter
        """
        # Initialize parent with Pydantic-compatible fields
        super().__init__(documents=documents, k=k)
        
        console.print(f"🔤 Building BM25 index for {len(documents)} documents...")
        
        # Use object.__setattr__ to bypass Pydantic validation for internal fields
        object.__setattr__(self, '_tokenized_corpus', [
            word_tokenize(doc.page_content.lower()) for doc in documents
        ])
        object.__setattr__(self, '_bm25', BM25Okapi(
            self._tokenized_corpus,
            k1=k1,
            b=b,
            epsilon=epsilon
        ))
        
        console.print(f"✅ BM25 index ready (k1={k1}, b={b})")
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents using BM25 scoring.
        
        Args:
            query: Search query string
            run_manager: Optional callback manager
            
        Returns:
            List of documents sorted by BM25 relevance
        """
        # Tokenize query
        tokenized_query = word_tokenize(query.lower())
        
        # Get BM25 scores
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:self.k]
        
        # Return documents with BM25 scores in metadata
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "bm25_score": float(scores[idx])}
            )
            results.append(doc_copy)
        
        return results
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self._get_relevant_documents(query, run_manager=run_manager)


class ParentChildRetriever:
    """
    Hierarchical retriever implementing parent-child document strategy.
    
    Workflow:
    1. Search child chunks using hybrid retrieval (vector + BM25)
    2. Map matched children to their parent documents
    3. Aggregate and deduplicate parent documents
    4. Return top-k parent contexts for LLM processing
    
    This ensures:
    - Precise matching via small child chunks
    - Complete context via large parent chunks
    - No loss of legal context or hierarchical structure
    """
    
    def __init__(
        self,
        vector_store,
        parent_documents: List[Document],
        child_documents: List[Document],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        top_k: int = 5,
        parent_top_k: int = 3
    ):
        """
        Initialize parent-child retriever.
        
        Args:
            vector_store: ChromaDB vector store containing child documents
            parent_documents: List of parent (large context) documents
            child_documents: List of child (small chunk) documents
            vector_weight: Weight for vector similarity search (0-1)
            bm25_weight: Weight for BM25 keyword search (0-1)
            top_k: Number of child chunks to retrieve
            parent_top_k: Maximum number of parent documents to return
        """
        self.vector_store = vector_store
        
        # Build parent lookup dictionary
        self.parent_docs = {}
        for doc in parent_documents:
            doc_id = doc.metadata.get("doc_id")
            if doc_id:
                self.parent_docs[doc_id] = doc
        
        self.child_docs = child_documents
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        self.parent_top_k = parent_top_k
        
        # Validate child-parent links
        self._validate_child_parent_links()
        
        # Initialize retrievers
        self._initialize_retrievers()
    
    def _validate_child_parent_links(self):
        """Validate that child documents have valid parent references."""
        missing_parents = 0
        for child in self.child_docs:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id not in self.parent_docs:
                missing_parents += 1
        
        if missing_parents > 0:
            console.print(
                f"[yellow]⚠️  {missing_parents} child documents have missing parent references[/yellow]"
            )
    
    def _initialize_retrievers(self):
        """Initialize vector and BM25 retrievers."""
        # Vector retriever from ChromaDB (without score_threshold - not supported in current version)
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        
        # BM25 retriever on child documents
        self.bm25_retriever = LegalBM25Retriever(
            documents=self.child_docs,
            k=self.top_k
        )
        
        # Ensemble retriever combining both approaches
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[self.vector_weight, self.bm25_weight]
        )
        
        console.print(
            f"🔄 Hybrid retriever ready "
            f"(Vector: {self.vector_weight:.0%}, BM25: {self.bm25_weight:.0%})"
        )
    
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve parent documents with aggregated relevance scores.
        
        Process:
        1. Search child chunks with ensemble retriever
        2. Aggregate scores by parent document
        3. Sort parents by combined score
        4. Return top-k parents with scores
        
        Args:
            query: Search query
            
        Returns:
            List of (document, score) tuples sorted by relevance
        """
        # Step 1: Get matched child documents using invoke (not deprecated get_relevant_documents)
        try:
            matched_children = self.ensemble_retriever.invoke(query)
        except Exception:
            # Fallback for older versions
            matched_children = self.ensemble_retriever.get_relevant_documents(query)
        
        # Step 2: Aggregate scores by parent
        parent_scores: Dict[str, float] = {}
        parent_children_count: Dict[str, int] = {}
        
        for child in matched_children:
            parent_id = child.metadata.get("parent_id")
            if not parent_id or parent_id not in self.parent_docs:
                continue
            
            # Get individual retriever scores
            vector_score = child.metadata.get("score", 0)
            bm25_score = child.metadata.get("bm25_score", 0)
            
            # Calculate weighted combined score
            combined_score = (
                vector_score * self.vector_weight +
                bm25_score * self.bm25_weight
            )
            
            # Update parent scores (take maximum child score)
            if parent_id in parent_scores:
                parent_scores[parent_id] = max(parent_scores[parent_id], combined_score)
                parent_children_count[parent_id] += 1
            else:
                parent_scores[parent_id] = combined_score
                parent_children_count[parent_id] = 1
        
        # Step 3: Sort parents by score
        sorted_parents = sorted(
            parent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.parent_top_k]
        
        # Step 4: Return parent documents with metadata
        results = []
        for parent_id, score in sorted_parents:
            if parent_id in self.parent_docs:
                doc = self.parent_docs[parent_id]
                # Add relevance metadata
                doc.metadata["relevance_score"] = round(score, 4)
                doc.metadata["matching_children"] = parent_children_count.get(parent_id, 0)
                results.append((doc, score))
        
        return results
    
    def retrieve_parent_contexts(self, query: str) -> List[Document]:
        """
        Retrieve parent documents for a query (without scores).
        
        Args:
            query: Search query
            
        Returns:
            List of parent documents
        """
        scored_results = self.retrieve_with_scores(query)
        return [doc for doc, _ in scored_results]
    
    def get_retrieval_stats(self, query: str) -> Dict[str, Any]:
        """
        Get detailed retrieval statistics for analysis and debugging.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with retrieval statistics
        """
        # Get matched children
        try:
            matched_children = self.ensemble_retriever.invoke(query)
        except Exception:
            matched_children = self.ensemble_retriever.get_relevant_documents(query)
        
        # Get parent results
        scored_parents = self.retrieve_with_scores(query)
        
        # Collect statistics
        child_sections = []
        for c in matched_children:
            section = c.metadata.get("section")
            if section:
                child_sections.append(section)
        
        parent_sections = []
        for doc, score in scored_parents:
            section = doc.metadata.get("section")
            if section:
                parent_sections.append(section)
        
        return {
            "query": query,
            "num_children_matched": len(matched_children),
            "num_parents_returned": len(scored_parents),
            "child_sections": list(set(child_sections)),
            "parent_sections": list(set(parent_sections)),
            "top_parent_scores": [
                {
                    "doc_id": doc.metadata.get("doc_id"),
                    "section": doc.metadata.get("section"),
                    "clause_type": doc.metadata.get("clause_type"),
                    "score": round(score, 4)
                }
                for doc, score in scored_parents
            ]
        }
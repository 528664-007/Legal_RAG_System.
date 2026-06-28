"""
Embedding model management for the Legal RAG system.

Handles loading and configuring sentence-transformer models
for generating dense vector embeddings of legal text.
"""

from typing import List, Optional
from rich.console import Console

# Handle HuggingFaceEmbeddings import across different LangChain versions
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "Could not import HuggingFaceEmbeddings. "
                "Install with: pip install langchain-huggingface"
            )

console = Console()


class EmbeddingManager:
    """
    Manages embedding model initialization and operations.
    
    Uses sentence-transformers models via HuggingFace for generating
    dense vector embeddings of legal text.
    
    Default model: all-MiniLM-L6-v2 (384-dimensional embeddings)
    """
    
    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2": {
            "dimension": 384,
            "description": "Fast, lightweight (80MB), good general purpose",
        },
        "all-mpnet-base-v2": {
            "dimension": 768,
            "description": "Higher quality (420MB), better for legal text",
        },
        "multi-qa-mpnet-base-dot-v1": {
            "dimension": 768,
            "description": "Optimized for question-answering retrieval",
        }
    }
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device for inference ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.device = device
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self._initialized = False
        
        # Get model info if available
        self.model_info = self.AVAILABLE_MODELS.get(
            model_name,
            {"dimension": "unknown", "description": "Custom model"}
        )
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Load and initialize the embedding model.
        Downloads the model on first use if not cached locally.
        """
        console.print(f"\n[bold]🔄 Loading Embedding Model[/bold]")
        console.print(f"   Model: {self.model_name}")
        console.print(f"   Device: {self.device}")
        console.print(f"   Info: {self.model_info.get('description', 'N/A')}")
        
        try:
            # FIXED: Removed show_progress_bar from encode_kwargs to avoid conflict
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={
                    "normalize_embeddings": True,  # Normalize for cosine similarity
                    # Removed show_progress_bar - causes conflict in newer versions
                }
            )
            
            # Test the model with a sample embedding
            test_text = "Confidential Information shall not be disclosed to third parties."
            test_embedding = self.embeddings.embed_query(test_text)
            
            self._initialized = True
            
            console.print(
                f"[green]✅ Embedding model loaded successfully[/green]"
            )
            console.print(f"   Embedding dimension: {len(test_embedding)}")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load embedding model: {e}[/red]")
            console.print("\n[yellow]Troubleshooting:[/yellow]")
            console.print("1. Check internet connection (first download)")
            console.print("2. Try: pip install --upgrade sentence-transformers")
            console.print("3. Try a different model name")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.embeddings or not self._initialized:
            raise RuntimeError("Embedding model not initialized")
        
        if not texts:
            return []
        
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            console.print(f"[red]❌ Embedding failed: {e}[/red]")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query string.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector
        """
        if not self.embeddings or not self._initialized:
            raise RuntimeError("Embedding model not initialized")
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            console.print(f"[red]❌ Query embedding failed: {e}[/red]")
            raise
    
    def get_embeddings(self):
        """
        Get the underlying LangChain embeddings object.
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        if not self.embeddings or not self._initialized:
            raise RuntimeError("Embedding model not initialized")
        return self.embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension (e.g., 384 for MiniLM)
        """
        if not self.embeddings or not self._initialized:
            raise RuntimeError("Embedding model not initialized")
        
        test_embedding = self.embeddings.embed_query("test")
        return len(test_embedding)
    
    def is_ready(self) -> bool:
        """Check if the embedding model is initialized and ready."""
        return self._initialized and self.embeddings is not None
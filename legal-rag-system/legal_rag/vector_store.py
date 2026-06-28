"""
ChromaDB vector store management for legal documents.
Handles creation, loading, and querying of vector embeddings.
"""

import os
from typing import List, Optional, Dict, Any
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from rich.console import Console
from .embeddings import EmbeddingManager

console = Console()


class VectorStoreManager:
    """
    Manages ChromaDB vector store operations for legal documents.
    
    Features:
    - Persistent storage with automatic directory creation
    - Collection management (create, load, delete)
    - Document addition and retrieval
    - Statistics and health checks
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        persist_directory: str = "./chroma_db",
        collection_name: str = "legal_documents"
    ):
        """
        Initialize vector store manager.
        
        Args:
            embedding_manager: EmbeddingManager instance for generating embeddings
            persist_directory: Directory path for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
        """
        self.embedding_manager = embedding_manager
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore: Optional[Chroma] = None
        
        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
        console.print(f"📁 Vector store directory: {persist_directory}")
    
    def create_from_documents(
        self,
        documents: List[Document],
        reset: bool = False
    ) -> Chroma:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of LangChain Document objects to index
            reset: If True, delete existing collection before creating new one
            
        Returns:
            Chroma vector store instance
        """
        if reset:
            console.print("[yellow]🔄 Resetting existing vector store...[/yellow]")
            self.delete_collection()
        
        console.print(f"\n[bold]📊 Indexing {len(documents)} documents...[/bold]")
        console.print(f"   Collection: {self.collection_name}")
        console.print(f"   Embedding dimension: {self.embedding_manager.get_embedding_dimension()}")
        
        try:
            # Create vector store from documents
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_manager.get_embeddings(),
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                collection_metadata={
                    "hnsw:space": "cosine",
                    "description": "Legal document embeddings for RAG system"
                }
            )
            
            # Persist to disk
            #self.vectorstore.persist()
            
            # Get collection stats
            count = self.vectorstore._collection.count()
            console.print(f"[green]✅ Vector store created with {count} documents[/green]")
            
            return self.vectorstore
            
        except Exception as e:
            console.print(f"[red]❌ Failed to create vector store: {e}[/red]")
            raise
    
    def load_existing(self) -> Optional[Chroma]:
        """
        Load an existing vector store from disk.
        
        Returns:
            Chroma vector store instance or None if not found
        """
        persist_path = os.path.join(self.persist_directory, "chroma.sqlite3")
        
        if not os.path.exists(persist_path):
            console.print("[yellow]⚠️  No existing vector store found[/yellow]")
            return None
        
        console.print(f"[bold]📂 Loading existing vector store...[/bold]")
        
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_manager.get_embeddings(),
                collection_name=self.collection_name
            )
            
            count = self.vectorstore._collection.count()
            console.print(f"[green]✅ Loaded vector store with {count} documents[/green]")
            
            return self.vectorstore
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not load vector store: {e}[/yellow]")
            return None
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to existing vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Number of documents added
        """
        if not self.vectorstore:
            raise ValueError(
                "Vector store not initialized. "
                "Call create_from_documents() or load_existing() first."
            )
        
        console.print(f"📝 Adding {len(documents)} documents to vector store...")
        
        try:
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            
            new_count = self.vectorstore._collection.count()
            console.print(f"[green]✅ Total documents: {new_count}[/green]")
            
            return len(documents)
            
        except Exception as e:
            console.print(f"[red]❌ Failed to add documents: {e}[/red]")
            raise
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get a retriever from the vector store.
        
        Args:
            search_kwargs: Search parameters dict (e.g., {"k": 5})
            
        Returns:
            Retriever instance configured for vector similarity search
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        # Note: score_threshold is not supported in all ChromaDB versions
        # Only add it if explicitly requested
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        try:
            if filter_metadata:
                return self.vectorstore.similarity_search(
                    query, k=k, filter=filter_metadata
                )
            else:
                return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            console.print(f"[red]❌ Similarity search failed: {e}[/red]")
            raise
    
    def delete_collection(self):
        """Delete the current collection and its persisted data."""
        if self.vectorstore:
            try:
                self.vectorstore.delete_collection()
                console.print("[yellow]🗑️  Collection deleted from memory[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not delete collection: {e}[/yellow]")
            
            self.vectorstore = None
        
        # Also remove persisted files
        persist_path = os.path.join(self.persist_directory, "chroma.sqlite3")
        if os.path.exists(persist_path):
            try:
                os.remove(persist_path)
                console.print(f"[yellow]🗑️  Persisted data removed: {persist_path}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not remove persisted data: {e}[/yellow]")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.vectorstore:
            return {"status": "not_initialized", "document_count": 0}
        
        try:
            collection = self.vectorstore._collection
            return {
                "status": "active",
                "name": collection.name,
                "document_count": collection.count(),
                "metadata": collection.metadata,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "document_count": 0
            }
    
    def health_check(self) -> bool:
        """
        Perform a health check on the vector store.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.vectorstore:
            return False
        
        try:
            # Try a simple query
            results = self.vectorstore.similarity_search("test", k=1)
            return len(results) > 0
        except Exception:
            return False
    
    def clear_all(self):
        """Clear all data - both in memory and on disk."""
        self.delete_collection()
        
        # Remove all files in persist directory
        if os.path.exists(self.persist_directory):
            for filename in os.listdir(self.persist_directory):
                file_path = os.path.join(self.persist_directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception:
                    pass
            
            console.print(f"[yellow]🗑️  All vector store data cleared[/yellow]")
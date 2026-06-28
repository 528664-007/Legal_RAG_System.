"""
Configuration management for Legal RAG System.
Loads settings from environment variables with sensible defaults.
Pre-configured to use Groq as the LLM provider.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LegalRAGConfig:
    """Central configuration for the Legal RAG system."""

    # ── LLM Provider ─────────────────────────────────────────────────────────
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "groq")
    )

    # Groq Settings (default provider)
    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # ← UPDATED
    )

    # Ollama Settings (optional fallback)
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3:8b-instruct")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )

    # OpenAI Settings (optional)
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    )

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    embedding_device: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu")
    )

    # ── Vector Store ─────────────────────────────────────────────────────────
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "legal_documents")
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    child_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHILD_CHUNK_SIZE", "250"))
    )
    child_chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))
    )
    parent_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("PARENT_CHUNK_SIZE", "1200"))
    )
    parent_chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("PARENT_CHUNK_OVERLAP", "100"))
    )

    # ── Retrieval ────────────────────────────────────────────────────────────
    vector_weight: float = field(
        default_factory=lambda: float(os.getenv("VECTOR_WEIGHT", "0.7"))
    )
    bm25_weight: float = field(
        default_factory=lambda: float(os.getenv("BM25_WEIGHT", "0.3"))
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "5"))
    )
    parent_top_k: int = field(
        default_factory=lambda: int(os.getenv("PARENT_TOP_K", "3"))
    )

    # ── LLM Parameters ───────────────────────────────────────────────────────
    temperature: float = 0.1
    max_tokens: int = 1024

    # ── Available Groq Models (for reference) ────────────────────────────────
    AVAILABLE_GROQ_MODELS = [
        "llama-3.1-8b-instant",      # Fast, efficient (recommended)
        "llama-3.3-70b-versatile",   # Powerful, large context
        "mixtral-8x7b-32768",        # Large context window
        "gemma2-9b-it",              # Google's model
        "deepseek-r1-distill-llama-70b",  # DeepSeek reasoning model
    ]

    # ─────────────────────────────────────────────────────────────────────────

    def validate(self) -> bool:
        """Validate the configuration."""
        valid_providers = ["ollama", "groq", "openai"]
        if self.llm_provider not in valid_providers:
            raise ValueError(
                f"Unsupported LLM provider: {self.llm_provider}. "
                f"Must be one of: {valid_providers}"
            )
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is required. Set it in your .env file.\n"
                "Get a free key at: https://console.groq.com"
            )
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider.")
        if not (0 <= self.vector_weight <= 1 and 0 <= self.bm25_weight <= 1):
            raise ValueError("Retrieval weights must be between 0 and 1")
        total = self.vector_weight + self.bm25_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0 (currently: {total})")
        if self.child_chunk_size >= self.parent_chunk_size:
            raise ValueError(
                f"Child chunk size ({self.child_chunk_size}) must be less than "
                f"parent chunk size ({self.parent_chunk_size})"
            )
        if self.child_chunk_size < 50:
            raise ValueError("Child chunk size must be at least 50 characters")
        return True

    def display(self):
        """Display current configuration in a formatted table."""
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(
            title="[bold cyan]Legal RAG System Configuration[/bold cyan]",
            border_style="cyan"
        )
        table.add_column("Category", style="yellow", no_wrap=True)
        table.add_column("Setting", style="green")
        table.add_column("Value", style="white")
        table.add_row("LLM", "Provider", self.llm_provider)
        if self.llm_provider == "groq":
            table.add_row("", "Model", self.groq_model)
            table.add_row("", "API Key", "****" + (self.groq_api_key or "")[-4:])
        elif self.llm_provider == "ollama":
            table.add_row("", "Model", self.ollama_model)
        elif self.llm_provider == "openai":
            table.add_row("", "Model", self.openai_model)
        table.add_row("", "Temperature", str(self.temperature))
        table.add_row("Embeddings", "Model", self.embedding_model_name)
        table.add_row("", "Device", self.embedding_device)
        table.add_row("Chunking", "Child Size", str(self.child_chunk_size))
        table.add_row("", "Parent Size", str(self.parent_chunk_size))
        table.add_row("Retrieval", "Vector Weight", str(self.vector_weight))
        table.add_row("", "BM25 Weight", str(self.bm25_weight))
        table.add_row("", "Top-K", str(self.top_k))
        table.add_row("Storage", "Persist Dir", self.chroma_persist_dir)
        console.print(table)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "llm_provider": self.llm_provider,
            "embedding_model": self.embedding_model_name,
            "child_chunk_size": self.child_chunk_size,
            "parent_chunk_size": self.parent_chunk_size,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "top_k": self.top_k,
            "parent_top_k": self.parent_top_k,
        }
"""
Main Legal RAG Pipeline orchestrating all components.
"""

from typing import List, Dict, Any, Optional

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from rich.console import Console
from rich.panel import Panel

from .config import LegalRAGConfig
from .embeddings import EmbeddingManager
from .document_processor import LegalDocumentProcessor
from .vector_store import VectorStoreManager
from .retrievers import ParentChildRetriever
from .llm_provider import LLMProvider
from .prompts import LegalPromptTemplates
from .utils import display_search_results, save_query_log, get_timestamp, create_error_response, count_tokens

console = Console()


class LegalRAGPipeline:
    """Complete Legal RAG pipeline with hierarchical retrieval."""

    def __init__(self, config: Optional[LegalRAGConfig] = None):
        self.config = config or LegalRAGConfig()
        self.config.validate()
        self._display_banner()
        self.config.display()
        self._initialize_components()
        self.parent_documents: List = []
        self.child_documents: List = []
        self.retriever: Optional[ParentChildRetriever] = None
        self.is_ingested = False
        self.query_count = 0

    def _display_banner(self):
        console.print(
            "\n[bold cyan]╔══════════════════════════════════════╗\n"
            "║  ⚖️  LEGAL RAG SYSTEM v1.0  ⚖️       ║\n"
            "║  Groq-Powered Legal Document Analysis ║\n"
            "║  Parent-Child Chunking + Hybrid Search║\n"
            "╚══════════════════════════════════════╝[/bold cyan]"
        )

    def _initialize_components(self):
        console.print("[bold]Initializing components...[/bold]\n")
        self.embedding_manager = EmbeddingManager(
            model_name=self.config.embedding_model_name,
            device=self.config.embedding_device,
        )
        self.document_processor = LegalDocumentProcessor(
            parent_chunk_size=self.config.parent_chunk_size,
            parent_chunk_overlap=self.config.parent_chunk_overlap,
            child_chunk_size=self.config.child_chunk_size,
            child_chunk_overlap=self.config.child_chunk_overlap,
        )
        self.vector_store_manager = VectorStoreManager(
            embedding_manager=self.embedding_manager,
            persist_directory=self.config.chroma_persist_dir,
            collection_name=self.config.collection_name,
        )
        self.llm_provider = LLMProvider(
            provider=self.config.llm_provider,
            model_name=self._get_model_name(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self._get_provider_kwargs(),
        )
        self.llm_provider.test_connection()
        self.research_prompt = LegalPromptTemplates.get_research_prompt()
        console.print("[green]✅ All components initialized[/green]\n")

    def _get_model_name(self) -> str:
        return {
            "groq": self.config.groq_model,
            "ollama": self.config.ollama_model,
            "openai": self.config.openai_model,
        }.get(self.config.llm_provider, "llama3-8b-8192")

    def _get_provider_kwargs(self) -> dict:
        if self.config.llm_provider == "groq":
            return {"api_key": self.config.groq_api_key}
        elif self.config.llm_provider == "ollama":
            return {"base_url": self.config.ollama_base_url}
        elif self.config.llm_provider == "openai":
            return {"api_key": self.config.openai_api_key}
        return {}

    def ingest_document(self, text: str, title: Optional[str] = None, reset: bool = False) -> Dict:
        console.print(f"\n[bold]📚 Ingesting document[/bold]: {title or 'Auto-detect'}")
        if reset:
            self.parent_documents = []
            self.child_documents = []
            self.vector_store_manager.delete_collection()
            self.is_ingested = False

        parents, children, stats = self.document_processor.create_parent_child_documents(
            text=text, document_title=title
        )
        self.parent_documents.extend(parents)
        self.child_documents.extend(children)

        if self.vector_store_manager.vectorstore is None:
            self.vector_store_manager.create_from_documents(documents=self.child_documents)
        else:
            self.vector_store_manager.add_documents(children)

        self._initialize_retriever()
        self.is_ingested = True
        console.print("[green]✅ Document ingested[/green]")
        return stats

    def ingest_multiple_documents(self, documents: List[Dict], reset: bool = True) -> List[Dict]:
        if reset:
            self.parent_documents = []
            self.child_documents = []
            self.vector_store_manager.delete_collection()
            self.is_ingested = False

        parents, children, stats = self.document_processor.process_multiple_documents(documents)
        self.parent_documents.extend(parents)
        self.child_documents.extend(children)
        self.vector_store_manager.create_from_documents(documents=self.child_documents)
        self._initialize_retriever()
        self.is_ingested = True
        return stats

    def _initialize_retriever(self):
        self.retriever = ParentChildRetriever(
            vector_store=self.vector_store_manager.vectorstore,
            parent_documents=self.parent_documents,
            child_documents=self.child_documents,
            vector_weight=self.config.vector_weight,
            bm25_weight=self.config.bm25_weight,
            top_k=self.config.top_k,
            parent_top_k=self.config.parent_top_k,
        )

    def query(self, user_query: str, log: bool = True) -> Dict[str, Any]:
        if not self.is_ingested:
            raise ValueError("No documents ingested. Call ingest_document() first.")

        self.query_count += 1
        console.print(f"\n[bold]🔍 Query #{self.query_count}:[/bold] {user_query[:80]}...")

        try:
            parent_contexts = self.retriever.retrieve_parent_contexts(user_query)
            stats = self.retriever.get_retrieval_stats(user_query)
            context_text = self._format_context(parent_contexts)

            chain = (
                {"context": lambda x: context_text, "query": RunnablePassthrough()}
                | self.research_prompt
                | self.llm_provider.get_llm()
                | StrOutputParser()
            )
            response = chain.invoke(user_query)

            display_search_results(
                query=user_query,
                documents=parent_contexts,
                response=response,
                retrieval_stats=stats,
            )

            if log:
                save_query_log(query=user_query, response=response, documents=parent_contexts)

            return {
                "query": user_query,
                "response": response,
                "retrieved_documents": parent_contexts,
                "retrieval_stats": stats,
                "query_number": self.query_count,
                "timestamp": get_timestamp(),
            }

        except Exception as e:
            console.print(f"[red]❌ Query failed: {e}[/red]")
            return {
                "query": user_query,
                "response": create_error_response(str(e)),
                "retrieved_documents": [],
                "retrieval_stats": {},
                "error": str(e),
            }

    def _format_context(self, documents: List) -> str:
        if not documents:
            return "No relevant legal context found."
        sections = []
        for i, doc in enumerate(documents, 1):
            header = (
                f"{'─'*60}\n"
                f"DOCUMENT {i} | {doc.metadata.get('document_title', 'N/A')}\n"
                f"Section: {doc.metadata.get('section', 'N/A')} | "
                f"Clause Type: {doc.metadata.get('clause_type', 'general')}\n"
                f"{'─'*60}\n"
            )
            sections.append(header + doc.page_content)
        return "\n\n".join(sections)

    def interactive_mode(self):
        console.print(Panel.fit(
            "[bold green]⚖️  Interactive Legal Research Mode[/bold green]\n\n"
            "Commands: [cyan]help[/cyan] | [cyan]stats[/cyan] | [cyan]exit[/cyan]\n"
            "Example: What are the remedies for breach of confidentiality?",
            border_style="green",
        ))
        while True:
            try:
                query = console.input("\n[bold cyan]Legal Query > [/bold cyan]").strip()
                if not query:
                    continue
                if query.lower() == "exit":
                    console.print(f"[yellow]👋 Goodbye! Processed {self.query_count} queries.[/yellow]")
                    break
                elif query.lower() == "stats":
                    self._display_stats()
                elif query.lower() == "help":
                    console.print("[italic]Enter any legal question to search the documents.[/italic]")
                else:
                    self.query(query)
            except KeyboardInterrupt:
                console.print(f"\n[yellow]👋 Exiting... {self.query_count} queries processed.[/yellow]")
                break

    def _display_stats(self):
        from rich.table import Table
        table = Table(title="System Statistics", border_style="blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Queries Processed", str(self.query_count))
        table.add_row("Parent Chunks", str(len(self.parent_documents)))
        table.add_row("Child Chunks", str(len(self.child_documents)))
        vs = self.vector_store_manager.get_collection_stats()
        table.add_row("Vector Store", vs.get("status", "N/A"))
        table.add_row("Vector Docs", str(vs.get("document_count", "N/A")))
        llm = self.llm_provider.get_provider_info()
        table.add_row("LLM", f"{llm['provider']}/{llm['model']}")
        console.print(table)

    def get_system_status(self) -> Dict[str, Any]:
        return {
            "initialized": True,
            "documents_ingested": self.is_ingested,
            "num_parents": len(self.parent_documents),
            "num_children": len(self.child_documents),
            "queries_processed": self.query_count,
            "vector_store": self.vector_store_manager.get_collection_stats(),
            "llm": self.llm_provider.get_provider_info(),
            "config": self.config.to_dict(),
        }

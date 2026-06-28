"""
Utility functions for the Legal RAG system.
"""

import re
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain.docstore.document import Document
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()


def display_search_results(
    query: str,
    documents: List[Document],
    response: str,
    retrieval_stats: Optional[Dict[str, Any]] = None,
):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit(
        f"[bold cyan]Legal Research Query[/bold cyan]\n{query}",
        border_style="cyan",
    ))

    if retrieval_stats:
        console.print(
            f"[dim]Retrieved {retrieval_stats['num_parents_returned']} parent contexts "
            f"from {retrieval_stats['num_children_matched']} child matches[/dim]\n"
        )

    if documents:
        table = Table(title="[bold]Retrieved Legal Contexts[/bold]", border_style="blue")
        table.add_column("#", style="dim", width=3)
        table.add_column("Document", style="green", width=30)
        table.add_column("Section", style="yellow", width=10)
        table.add_column("Type", style="blue", width=15)
        table.add_column("Preview", style="white", width=50)

        for i, doc in enumerate(documents, 1):
            preview = doc.page_content[:80].replace("\n", " ")
            table.add_row(
                str(i),
                doc.metadata.get("document_title", "N/A")[:28],
                doc.metadata.get("section", "N/A")[:8],
                doc.metadata.get("clause_type", "general")[:13],
                preview + "...",
            )
        console.print(table)

    console.print("\n[bold green]🤖 Legal Analysis:[/bold green]")
    console.print(Panel(Markdown(response), border_style="green"))
    console.print("=" * 80 + "\n")


def save_query_log(query: str, response: str, documents: List[Document], log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "legal_rag_queries.json")
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "num_documents": len(documents),
        "documents": [
            {
                "title": d.metadata.get("document_title", "N/A"),
                "section": d.metadata.get("section", "N/A"),
                "clause_type": d.metadata.get("clause_type", "N/A"),
            }
            for d in documents
        ],
    }
    try:
        with open(log_path, "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []
    logs.append(entry)
    with open(log_path, "w") as f:
        json.dump(logs[-1000:], f, indent=2)


def count_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_error_response(error_message: str) -> str:
    return (
        f"An error occurred while processing your query: {error_message}\n\n"
        "DISCLAIMER: This response is provided for informational purposes only "
        "and does not constitute legal advice."
    )

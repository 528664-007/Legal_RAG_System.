#!/usr/bin/env python3
"""
Demonstration script for Legal RAG System (Groq-powered).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legal_rag import LegalRAGConfig, LegalRAGPipeline
from data.sample_contracts import ALL_SAMPLES
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

TEST_QUERIES = [
    "What remedies are available if the receiving party breaches confidentiality?",
    "What constitutes a trade secret under California law?",
    "How long does the confidentiality obligation survive after termination?",
    "Can the disclosing party get an injunction without proving actual damages?",
    "What are the exceptions to confidentiality obligations?",
    "What happens to intellectual property created during employment?",
    "What are the grounds for termination for cause under the employment agreement?",
]


def run_demo():
    console.print(Panel.fit(
        "[bold cyan]⚖️  Legal RAG System - Groq Demo[/bold cyan]\n"
        "Ingesting 3 sample legal documents and running 7 test queries.",
        border_style="cyan",
    ))

    config = LegalRAGConfig(llm_provider="groq")
    pipeline = LegalRAGPipeline(config)
    pipeline.ingest_multiple_documents(ALL_SAMPLES, reset=True)

    for i, q in enumerate(TEST_QUERIES, 1):
        console.print(f"\n[yellow]─── Query {i}/{len(TEST_QUERIES)} ───[/yellow]")
        try:
            pipeline.query(q, log=True)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    status = pipeline.get_system_status()
    table = Table(title="Demo Summary", border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_row("Documents Ingested", str(len(ALL_SAMPLES)))
    table.add_row("Parent Chunks", str(status["num_parents"]))
    table.add_row("Child Chunks", str(status["num_children"]))
    table.add_row("Queries Processed", str(status["queries_processed"]))
    table.add_row("LLM", f"{status['llm']['provider']}/{status['llm']['model']}")
    console.print(table)
    console.print("\n[bold green]✅ Demo complete! Check logs/legal_rag_queries.json[/bold green]")


if __name__ == "__main__":
    run_demo()

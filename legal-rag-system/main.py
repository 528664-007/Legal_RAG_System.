#!/usr/bin/env python3
"""
Legal RAG System - Main Entry Point (Groq-powered)

Usage:
    python main.py                         # Interactive mode with sample docs
    python main.py --demo                  # Full demo with 7 test queries
    python main.py --query "question"      # Single query
    python main.py --ingest file.txt       # Ingest custom document
"""

import sys
import os
import argparse
from dotenv import load_dotenv
import chromadb
chromadb.config.Settings.anonymized_telemetry = False

load_dotenv()

from legal_rag import LegalRAGConfig, LegalRAGPipeline
from data.sample_contracts import ALL_SAMPLES
from rich.console import Console

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="⚖️  Legal RAG System (Groq-powered)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", help="Run full demonstration")
    group.add_argument("--interactive", action="store_true", help="Start interactive mode")
    p.add_argument("--query", "-q", type=str, help="Single query to process")
    p.add_argument("--ingest", "-i", type=str, help="Path to a text file to ingest")
    p.add_argument("--title", "-t", type=str, help="Title for the ingested document")
    p.add_argument(
        "--provider", "-p",
        choices=["groq", "ollama", "openai"],
        default=None,
        help="LLM provider (default: groq)",
    )
    p.add_argument("--no-log", action="store_true", help="Disable query logging")
    p.add_argument("--reset", action="store_true", help="Reset vector store")
    return p.parse_args()


def main():
    args = parse_args()

    config = LegalRAGConfig()
    if args.provider:
        config.llm_provider = args.provider
    config.validate()

    pipeline = LegalRAGPipeline(config)

    if args.demo:
        from scripts.run_demo import run_demo
        run_demo()

    elif args.ingest:
        if not os.path.exists(args.ingest):
            console.print(f"[red]File not found: {args.ingest}[/red]")
            sys.exit(1)
        with open(args.ingest, "r", encoding="utf-8") as f:
            text = f.read()
        title = args.title or os.path.basename(args.ingest)
        pipeline.ingest_document(text=text, title=title, reset=args.reset)
        if args.query:
            pipeline.query(args.query, log=not args.no_log)
        else:
            pipeline.interactive_mode()

    elif args.query:
        if not pipeline.is_ingested:
            console.print("[yellow]Loading sample documents...[/yellow]")
            pipeline.ingest_multiple_documents(ALL_SAMPLES)
        pipeline.query(args.query, log=not args.no_log)

    else:
        # Default: interactive mode with sample documents
        if not pipeline.is_ingested:
            console.print("[yellow]Loading sample documents for interactive mode...[/yellow]")
            pipeline.ingest_multiple_documents(ALL_SAMPLES)
        pipeline.interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        sys.exit(1)

"""
Advanced legal document processing with hierarchical chunking.
"""

import re
import uuid
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class LegalMetadata:
    section: str = ""
    article: str = ""
    clause_type: str = "general"
    document_title: str = ""
    jurisdiction: str = ""
    party_names: List[str] = field(default_factory=list)


class LegalDocumentProcessor:
    """Processes legal documents with hierarchical chunking strategy."""

    def __init__(
        self,
        parent_chunk_size: int = 1200,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 250,
        child_chunk_overlap: int = 50,
    ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _extract_legal_metadata(self, text: str) -> LegalMetadata:
        metadata = LegalMetadata()
        for pattern in [
            r"(?:Section|SECTION)\s*(\d+[\.\.\d]*)",
            r"§\s*(\d+[\.\.\d]*)",
            r"^(\d+\.\d+)\s",
        ]:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                metadata.section = m.group(1)
                break

        am = re.search(r"Article\s*(\d+[\.\.\w]*)", text, re.IGNORECASE)
        if am:
            metadata.article = am.group(1)

        clause_patterns = {
            "confidentiality": [r"\bconfidential\b", r"\bnon-disclosure\b", r"\btrade\s+secret\b"],
            "indemnification": [r"\bindemnif", r"\bhold\s+harmless\b", r"\bliability\b"],
            "termination": [r"\bterminat", r"\bcancellation\b", r"\bexpir"],
            "dispute_resolution": [r"\barbitration\b", r"\bdispute\s+resolution\b"],
            "governing_law": [r"\bgoverning\s+law\b", r"\bjurisdiction\b"],
            "definitions": [r"\bshall\s+mean\b", r"\bas\s+used\s+herein\b"],
            "intellectual_property": [r"\bintellectual\s+property\b", r"\bcopyright\b"],
            "compensation": [r"\bcompensation\b", r"\bsalary\b", r"\bpayment\b"],
        }
        for ctype, pats in clause_patterns.items():
            if any(re.search(p, text, re.IGNORECASE) for p in pats):
                metadata.clause_type = ctype
                break

        jm = re.search(
            r"(?:State of|laws of the State of|laws of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            text,
            re.IGNORECASE,
        )
        if jm:
            metadata.jurisdiction = jm.group(1)

        return metadata

    def _detect_document_type(self, text: str) -> str:
        tl = text.lower()
        if any(t in tl for t in ["non-disclosure agreement", "confidentiality agreement", "nda"]):
            return "Non-Disclosure Agreement"
        elif "employment agreement" in tl or "employment contract" in tl:
            return "Employment Agreement"
        elif "service agreement" in tl:
            return "Service Agreement"
        elif "civil code" in tl or "statute" in tl:
            return "Statutory Text"
        elif "contract" in tl or "agreement" in tl:
            return "Contract"
        return "Legal Document"

    def create_parent_child_documents(
        self, text: str, document_title: Optional[str] = None
    ) -> Tuple[List[Document], List[Document], Dict]:
        if not document_title:
            document_title = self._detect_document_type(text)

        console.print(f"\n[bold]📄 Processing:[/bold] {document_title}")
        console.print(f"   Original length: {len(text):,} characters")

        parent_texts = self.parent_splitter.split_text(text)
        parent_docs, child_docs = [], []
        stats = {
            "document_title": document_title,
            "original_length": len(text),
            "num_parents": len(parent_texts),
            "num_children": 0,
            "clause_types_found": set(),
        }

        for idx, pt in enumerate(parent_texts):
            pid = f"parent_{uuid.uuid4().hex[:8]}"
            meta = self._extract_legal_metadata(pt)
            stats["clause_types_found"].add(meta.clause_type)

            parent_docs.append(Document(
                page_content=pt,
                metadata={
                    "doc_id": pid,
                    "document_title": document_title,
                    "chunk_type": "parent",
                    "chunk_index": idx,
                    "section": meta.section,
                    "article": meta.article,
                    "clause_type": meta.clause_type,
                    "jurisdiction": meta.jurisdiction,
                    "word_count": len(pt.split()),
                    "char_count": len(pt),
                },
            ))

            for cidx, ct in enumerate(self.child_splitter.split_text(pt)):
                child_docs.append(Document(
                    page_content=ct,
                    metadata={
                        "parent_id": pid,
                        "document_title": document_title,
                        "chunk_type": "child",
                        "parent_index": idx,
                        "child_index": cidx,
                        "section": meta.section,
                        "clause_type": meta.clause_type,
                        "word_count": len(ct.split()),
                        "char_count": len(ct),
                    },
                ))

        stats["num_children"] = len(child_docs)
        stats["clause_types_found"] = list(stats["clause_types_found"])
        self._display_summary(stats, parent_docs, child_docs)
        return parent_docs, child_docs, stats

    def _display_summary(self, stats, parent_docs, child_docs):
        table = Table(title="Document Processing Summary", border_style="green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")
        table.add_row("Document Title", stats["document_title"])
        table.add_row("Original Length", f"{stats['original_length']:,} chars")
        table.add_row("Parent Chunks", str(stats["num_parents"]))
        table.add_row("Child Chunks", str(stats["num_children"]))
        if parent_docs:
            avg = sum(d.metadata["word_count"] for d in parent_docs) / len(parent_docs)
            table.add_row("Avg Parent Size", f"{avg:.0f} words")
        if child_docs:
            avg = sum(d.metadata["word_count"] for d in child_docs) / len(child_docs)
            table.add_row("Avg Child Size", f"{avg:.0f} words")
        table.add_row("Clause Types", ", ".join(stats["clause_types_found"]))
        console.print(table)

    def process_multiple_documents(self, documents: List[Dict]) -> Tuple[List, List, List]:
        all_parents, all_children, all_stats = [], [], []
        console.print(f"\n[bold]📚 Processing {len(documents)} documents...[/bold]")
        for i, doc in enumerate(documents, 1):
            console.print(f"\n[yellow]Document {i}/{len(documents)}[/yellow]")
            p, c, s = self.create_parent_child_documents(
                text=doc["text"], document_title=doc.get("title")
            )
            all_parents.extend(p)
            all_children.extend(c)
            all_stats.append(s)
        console.print(f"\n[green]✅ Total: {len(all_parents)} parents, {len(all_children)} children[/green]")
        return all_parents, all_children, all_stats

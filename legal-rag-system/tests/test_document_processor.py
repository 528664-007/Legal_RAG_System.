"""Tests for LegalDocumentProcessor."""

import pytest
from legal_rag.document_processor import LegalDocumentProcessor


@pytest.fixture
def processor():
    return LegalDocumentProcessor(
        parent_chunk_size=500, parent_chunk_overlap=50,
        child_chunk_size=150, child_chunk_overlap=25,
    )


def test_create_parent_child_documents(processor):
    text = """
    Article 1 - Definitions
    Section 1.1 "Confidential Information" means any non-public information.
    Section 1.2 "Receiving Party" means the party receiving information.
    Article 2 - Obligations
    Section 2.1 The Receiving Party shall maintain confidentiality.
    Section 2.2 The Receiving Party shall not disclose to third parties.
    """ * 5
    parents, children, stats = processor.create_parent_child_documents(text, "Test Agreement")
    assert len(parents) > 0
    assert len(children) >= len(parents)
    for p in parents:
        assert p.metadata["chunk_type"] == "parent"
        assert "doc_id" in p.metadata
    for c in children:
        assert c.metadata["chunk_type"] == "child"
        assert "parent_id" in c.metadata
        pid = c.metadata["parent_id"]
        assert any(p.metadata["doc_id"] == pid for p in parents)


def test_detect_document_type(processor):
    assert processor._detect_document_type("This Non-Disclosure Agreement...") == "Non-Disclosure Agreement"
    assert processor._detect_document_type("This Employment Agreement...") == "Employment Agreement"
    assert processor._detect_document_type("California Civil Code...") == "Statutory Text"
    assert processor._detect_document_type("Random text here...") == "Legal Document"


def test_extract_metadata_section(processor):
    text = "Section 3426.1 - Definitions\nAs used herein, trade secret means..."
    meta = processor._extract_legal_metadata(text)
    assert meta.section == "3426.1"

"""Tests for retrieval components."""

import pytest
from langchain.docstore.document import Document
from legal_rag.retrievers import LegalBM25Retriever


@pytest.fixture
def documents():
    return [
        Document(page_content="Confidential Information shall not be disclosed to third parties.",
                 metadata={"section": "2.1", "clause_type": "confidentiality"}),
        Document(page_content="The Receiving Party shall maintain strict confidentiality.",
                 metadata={"section": "2.2", "clause_type": "confidentiality"}),
        Document(page_content="This Agreement shall be governed by California law.",
                 metadata={"section": "6.1", "clause_type": "governing_law"}),
        Document(page_content="Upon termination, all materials shall be returned.",
                 metadata={"section": "5.2", "clause_type": "termination"}),
    ]


def test_basic_retrieval(documents):
    retriever = LegalBM25Retriever(documents=documents, k=2)
    results = retriever.get_relevant_documents("confidentiality obligations")
    assert len(results) == 2
    assert any("confidentiality" in d.metadata.get("clause_type", "") for d in results)


def test_bm25_scores_present(documents):
    retriever = LegalBM25Retriever(documents=documents, k=2)
    results = retriever.get_relevant_documents("confidentiality")
    for doc in results:
        assert "bm25_score" in doc.metadata
        assert isinstance(doc.metadata["bm25_score"], float)


def test_result_count_respected(documents):
    retriever = LegalBM25Retriever(documents=documents, k=1)
    results = retriever.get_relevant_documents("California law")
    assert len(results) == 1


def test_original_docs_unchanged(documents):
    original = documents[0].page_content
    retriever = LegalBM25Retriever(documents=documents, k=2)
    retriever.get_relevant_documents("test query")
    assert documents[0].page_content == original

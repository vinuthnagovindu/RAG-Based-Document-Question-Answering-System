"""
Tests for main.py (RAG Document Assistant).

These tests mock out HuggingFace embeddings, Chroma, and the Gemini LLM so
they run fast, offline, and without needing a GOOGLE_API_KEY. They focus on
verifying route behavior and contracts (status codes, response shape,
error handling) rather than actual retrieval quality.

Run with:  pytest
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure the app can be imported without a real .env / API key present.
os.environ.setdefault("GOOGLE_API_KEY", "test-key-not-real")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main  # noqa: E402


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset module-level chain/retriever/embeddings state between tests."""
    main._chain = None
    main._retriever = None
    main._embeddings = None
    yield
    main._chain = None
    main._retriever = None
    main._embeddings = None


@pytest.fixture
def client():
    return TestClient(main.app)


# ── / (serve UI) ──────────────────────────────────────────────────────────

def test_serve_ui_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200


# ── /ask ──────────────────────────────────────────────────────────────────

def test_ask_without_any_documents_returns_400(client, monkeypatch):
    """If no chain is built and no chroma_db exists, /ask should 400, not crash."""
    monkeypatch.setattr(main.os.path, "exists", lambda path: False)
    response = client.post("/ask", json={"question": "What is this document about?"})
    assert response.status_code == 400
    assert "No documents ingested" in response.json()["detail"]


def test_ask_with_existing_chain_returns_answer_and_sources(client):
    """When a chain/retriever already exist, /ask should use them directly."""
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "This document is about testing."

    fake_doc = MagicMock()
    fake_doc.metadata = {"page": 1}
    fake_doc.page_content = "Some relevant chunk of text " * 5

    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = [fake_doc]

    main._chain = fake_chain
    main._retriever = fake_retriever

    response = client.post("/ask", json={"question": "What is this about?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "This document is about testing."
    assert len(body["sources"]) == 1
    assert body["sources"][0]["page"] == 1
    assert body["sources"][0]["chunk"] == 1
    fake_chain.invoke.assert_called_once_with("What is this about?")


def test_ask_missing_question_field_returns_422(client):
    """Pydantic validation should reject a body without 'question'."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_builds_chain_when_chroma_db_exists(client, monkeypatch):
    """If chroma_db exists on disk but chain isn't in memory, build_chain() runs."""
    monkeypatch.setattr(main.os.path, "exists", lambda path: True)

    def fake_build_chain():
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = "Rebuilt answer."
        fake_retriever = MagicMock()
        fake_retriever.invoke.return_value = []
        main._chain = fake_chain
        main._retriever = fake_retriever

    with patch.object(main, "build_chain", side_effect=fake_build_chain) as mock_build:
        response = client.post("/ask", json={"question": "Anything?"})

    mock_build.assert_called_once()
    assert response.status_code == 200
    assert response.json()["answer"] == "Rebuilt answer."


# ── /ingest ───────────────────────────────────────────────────────────────

def test_ingest_txt_file_returns_chunk_count(client, tmp_path):
    fake_docs = [MagicMock(page_content="Hello world, this is a test document.")]
    fake_chunks = [MagicMock(), MagicMock()]  # pretend the splitter made 2 chunks

    with patch.object(main, "TextLoader") as mock_loader_cls, \
         patch.object(main, "RecursiveCharacterTextSplitter") as mock_splitter_cls, \
         patch.object(main, "get_embeddings", return_value=MagicMock()), \
         patch.object(main.Chroma, "from_documents") as mock_from_docs, \
         patch.object(main, "build_chain") as mock_build_chain:

        mock_loader_cls.return_value.load.return_value = fake_docs
        mock_splitter_cls.return_value.split_documents.return_value = fake_chunks

        file_content = b"Hello world, this is a test document."
        response = client.post(
            "/ingest",
            files={"files": ("notes.txt", file_content, "text/plain")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks"] == 2
    mock_from_docs.assert_called_once()
    mock_build_chain.assert_called_once()


def test_ingest_requires_at_least_one_file(client):
    response = client.post("/ingest", files={})
    assert response.status_code == 422


# ── /clear ────────────────────────────────────────────────────────────────

def test_clear_resets_globals(client):
    main._chain = MagicMock()
    main._retriever = MagicMock()
    main._embeddings = MagicMock()

    response = client.post("/clear")

    assert response.status_code == 200
    assert response.json() == {"status": "cleared"}
    assert main._chain is None
    assert main._retriever is None
    assert main._embeddings is None

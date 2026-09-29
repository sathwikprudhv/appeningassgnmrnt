"""Offline checks: python -m unittest -v test_project.py

Synthetic evidence and mocked service calls only; no actual eBook evaluation.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from rag import build_graph, Chatbot, Selection
from config import REFUSAL
from ingest import read_chunks
from api import app


def chunk(eligible=True):
    return {"id": "chunk-1", "page": 2, "text": "Memory stores previous observations.",
            "score": 0.8 if eligible else 0.1, "eligible": eligible}


class GraphTests(unittest.TestCase):
    def run_graph(self, chunks, selection):
        selector = Mock(return_value=selection)
        graph = build_graph(lambda question: chunks, selector)
        return graph.invoke({"question": "What does memory store?"}), selector

    def test_valid_quote_has_real_citation(self):
        result, selector = self.run_graph([chunk()], {"answerable": True, "evidence": [
            {"chunk_id": "chunk-1", "quote": "Memory stores previous observations."}]})
        self.assertEqual(result["status"], "answered")
        self.assertIn("PDF page 2", result["answer"])
        self.assertEqual(result["relevance_score"], 0.8)
        selector.assert_called_once()

    def test_empty_retrieval_skips_model(self):
        result, selector = self.run_graph([], {})
        self.assertEqual(result["answer"], REFUSAL)
        self.assertIsNone(result["relevance_score"])
        selector.assert_not_called()

    def test_low_relevance_skips_model_but_retains_chunks(self):
        result, selector = self.run_graph([chunk(False)], {})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(len(result["contexts"]), 1)
        selector.assert_not_called()

    def test_high_relevance_does_not_override_unanswerable(self):
        result, _ = self.run_graph([chunk()], {"answerable": False, "evidence": []})
        self.assertEqual(result["answer"], REFUSAL)

    def test_fabricated_text_and_source_fail_closed(self):
        for evidence in [
            {"chunk_id": "chunk-1", "quote": "Argentina won the World Cup."},
            {"chunk_id": "unknown", "quote": "Memory stores previous observations."},
        ]:
            with self.subTest(evidence=evidence):
                result, _ = self.run_graph([chunk()], {"answerable": True, "evidence": [evidence]})
                self.assertEqual(result["answer"], REFUSAL)

    def test_api_failure_is_not_a_grounded_answer(self):
        graph = build_graph(Mock(side_effect=ConnectionError("offline")), Mock())
        with self.assertRaises(ConnectionError):
            graph.invoke({"question": "Memory?"})

    def test_public_result_contract(self):
        bot = object.__new__(Chatbot)
        bot.graph = build_graph(lambda question: [], Mock())
        output = bot.ask("A valid question")
        self.assertTrue({"answer", "retrieved_chunks", "confidence_score"} <= output.keys())
        for question in ("  ", "x" * 2001):
            with self.assertRaises(ValueError):
                bot.ask(question)


class IngestionTests(unittest.TestCase):
    @patch("ingest.PyPDFLoader")
    def test_chunking_preserves_pdf_page_numbers(self, loader):
        loader.return_value.load.return_value = [
            Document(page_content=" "), Document(page_content="Memory and planning. " * 150)]
        chunks, blank = read_chunks("synthetic.pdf")
        self.assertEqual(blank, [1])
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c["page"] == 2 and len(c["text"]) <= 1000 for c in chunks))

    def test_real_blank_pdf_requires_ocr(self):
        from pypdf import PdfWriter
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            writer.write(path)
            with self.assertRaisesRegex(ValueError, "No extractable PDF text"):
                read_chunks(path)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("api.get_bot")
    def test_chat_accepts_query_and_returns_required_keys(self, get_bot):
        get_bot.return_value.ask.return_value = {
            "answer": REFUSAL, "retrieved_chunks": [], "confidence_score": None}
        response = self.client.post("/chat", json={"query": "What is Agentic AI?"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("retrieved_chunks", response.json())
        get_bot.return_value.ask.assert_called_once_with("What is Agentic AI?")

    def test_invalid_input(self):
        for payload in ({}, {"query": ""}, {"query": "  "}, {"query": "x" * 2001}):
            self.assertEqual(self.client.post("/chat", json=payload).status_code, 422)

    @patch("api.get_bot", side_effect=RuntimeError("private provider detail"))
    def test_service_failure_is_503_and_does_not_leak_details(self, get_bot):
        response = self.client.post("/chat", json={"query": "What is memory?"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private provider detail", response.text)


if __name__ == "__main__":
    unittest.main()

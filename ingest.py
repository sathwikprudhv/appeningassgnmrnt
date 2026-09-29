"""Run once: python ingest.py data/Ebook-Agentic-AI.pdf"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from config import settings, MANIFEST, EMBEDDING_MODEL, DIMENSIONS, PIPELINE_VERSION


def read_chunks(path):
    pages = PyPDFLoader(str(path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks, empty_pages = [], []
    for page_number, page in enumerate(pages, start=1):
        text = " ".join(page.page_content.split())
        if not text:
            empty_pages.append(page_number)
        for part in splitter.split_text(text):
            chunks.append({"page": page_number, "text": part})
    if not chunks:
        raise ValueError("No extractable PDF text. Use a text PDF or run OCR first.")
    return chunks, empty_pages


def ingest(path):
    cfg = settings()
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise ValueError("Provide the path to the downloaded Agentic AI eBook PDF.")
    chunks, empty_pages = read_chunks(path)
    doc_id = hashlib.sha256(path.read_bytes()).hexdigest()
    namespace = hashlib.sha256((doc_id + PIPELINE_VERSION).encode()).hexdigest()
    if empty_pages:
        print(f"Warning: no text on PDF pages {empty_pages}; inspect them for images needing OCR.")
    print(f"Extracted {len(chunks)} chunks. Creating/checking Pinecone index...")
    pc = Pinecone()
    if cfg["index"] not in pc.list_indexes().names():
        pc.create_index(name=cfg["index"], dimension=DIMENSIONS, metric="cosine",
                        spec=ServerlessSpec(cloud=cfg["cloud"], region=cfg["region"]))
    deadline = time.monotonic() + 120
    while True:
        description = pc.describe_index(cfg["index"])
        if description.dimension != DIMENSIONS or description.metric != "cosine":
            raise ValueError("Index must have dimension 1536 and cosine metric. Choose a new index name.")
        if description.status["ready"]:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Index is still starting. Wait briefly, then rerun ingestion.")
        time.sleep(2)
    index = pc.Index(host=description.host)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=DIMENSIONS,
                                  request_timeout=45, max_retries=2)
    store = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)
    for start in range(0, len(chunks), 32):
        batch = chunks[start:start + 32]
        documents = [Document(page_content=chunk["text"], metadata={
            "page": chunk["page"], "chunk_id": f"chunk-{start + offset}",
            "doc_id": doc_id, "source": "Agentic AI eBook"})
            for offset, chunk in enumerate(batch)]
        store.add_documents(documents=documents,
                            ids=[doc.metadata["chunk_id"] for doc in documents],
                            async_req=False)
        print(f"Stored {min(start + 32, len(chunks))}/{len(chunks)} chunks")
    # Pinecone writes are eventually consistent; don't publish a partial manifest.
    deadline = time.monotonic() + 90
    while True:
        stats = index.describe_index_stats().to_dict()
        count = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
        if count >= len(chunks):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Vectors are not visible yet. Rerun ingestion; IDs prevent duplicates.")
        time.sleep(2)
    manifest = {"doc_id": doc_id, "namespace": namespace, "index": cfg["index"],
                "embedding_model": EMBEDDING_MODEL, "dimensions": DIMENSIONS,
                "pipeline_version": PIPELINE_VERSION, "chunks": len(chunks),
                "empty_pages": empty_pages}
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print("Ingestion complete. Run: python -m streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    args = parser.parse_args()
    try:
        ingest(args.pdf)
    except Exception as exc:
        raise SystemExit(f"Ingestion failed ({type(exc).__name__}): {exc}")

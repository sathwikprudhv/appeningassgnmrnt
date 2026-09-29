"""Read local configuration. Never put real keys in Python files."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
MANIFEST = ROOT / "ingestion.json"
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
PIPELINE_VERSION = "page-char1000-overlap200-v1"
REFUSAL = "I couldn't find enough information in the Agentic AI eBook to answer that question."


def settings():
    for key in ("OPENAI_API_KEY", "PINECONE_API_KEY"):
        value = os.getenv(key, "").strip()
        if not value or value.startswith("replace_"):
            raise ValueError(f"Set {key} in your local .env file, then restart.")
    threshold = float(os.getenv("MIN_RELEVANCE", "0.35"))
    if not -1 <= threshold <= 1:
        raise ValueError("MIN_RELEVANCE must be between -1 and 1.")
    return {
        "index": os.getenv("PINECONE_INDEX", "agentic-ai-ebook"),
        "cloud": os.getenv("PINECONE_CLOUD", "aws"),
        "region": os.getenv("PINECONE_REGION", "us-east-1"),
        "model": os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        "threshold": threshold,
    }

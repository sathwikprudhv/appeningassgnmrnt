"""Optional local API: python -m uvicorn api:app --reload"""
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag import Chatbot

app = FastAPI(title="Agentic AI eBook RAG API")


class Question(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@lru_cache(maxsize=1)
def get_bot():
    return Chatbot()


@app.get("/health")
def health():
    return {"status": "running", "note": "Process check only; does not test external services."}


@app.post("/chat")
def ask(request: Question):
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Question must not be blank.")
    try:
        return get_bot().ask(request.query)
    except Exception:
        raise HTTPException(status_code=503, detail="Chatbot unavailable. Check ingestion, configuration and service access.") from None

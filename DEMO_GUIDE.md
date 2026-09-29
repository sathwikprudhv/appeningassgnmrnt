# Demonstrate the project — beginner guide

Use this as a speaking outline after you have run the real eBook successfully. A recording is not required by the supplied submission checklist; the requested submission is the GitHub URL. This outline helps with a live review or an optional demo. Do not claim a result until you have observed it.

## Before presenting

1. Complete ingestion and all six live queries in README section 9.
2. Review the quoted answers against the PDF and record actual results.
3. Start Streamlit and FastAPI in separate terminals using the README commands.
4. Open the UI, the PDF and the GitHub README in separate tabs.
5. Close `.env` and any account/key pages. Never show keys on screen.

## A 4–5 minute walkthrough

| Time | Show on screen | What to say |
|---|---|---|
| 0:00–0:35 | Chatbot title and README objective | “This Python chatbot answers questions using the provided Agentic AI eBook. It retrieves relevant passages and returns supporting text, page references and a retrieval score. It is designed to refuse questions without enough document evidence.” |
| 0:35–1:15 | `ingest.py`, `read_chunks` and the storage loop | “I load the PDF page by page with PyPDFLoader. I split each page into chunks of up to 1,000 characters with a target overlap of 200. OpenAI converts chunks into 1,536-dimensional embeddings. LangChain's Pinecone wrapper stores the vectors with text, page numbers and chunk IDs.” |
| 1:15–2:00 | `rag.py`, `AgentState`, then `build_graph` | “LangGraph passes state between retrieval, generation and validation. Retrieval gets five chunks from the active PDF. The model selects relevant quotations. Python checks that each quote exists in its named source. Unsupported output is refused.” |
| 2:00–2:50 | UI: ask “What is Agentic AI according to the eBook?” | Read the actual response briefly. Expand a retrieved chunk. Say: “This passage is the evidence for the answer. I can check the PDF page directly. The number shown is retrieval similarity, not an accuracy percentage.” |
| 2:50–3:25 | UI: ask the World Cup question | If it refuses, say: “This question lacks support in the retrieved eBook passages, so the system refuses.” If it fails, say it failed and explain the observed issue; do not present it as a pass. |
| 3:25–4:00 | FastAPI `/docs`, execute `POST /chat` | “The same workflow can be called through an API. The request contains query. The response returns answer, retrieved_chunks and confidence_score, plus citations and status.” |
| 4:00–4:40 | Actual `benchmark_results.json` and README limitations | “I reviewed these six live responses against the eBook. Here are the observed results. Limitations include PDF extraction quality, retrieval misses and the model's judgment about relevance. The current answer format uses direct quotations.” |

Adjust the wording to match work you performed. Explain any assistance honestly if asked.

## A simple explanation you should understand

Imagine an open-book exam. Pinecone finds relevant pages. The LLM chooses useful lines. Python checks that those lines really occur in the retrieved text. LangGraph organizes that sequence. FastAPI and Streamlit let someone use it.

## Questions a reviewer may ask

**Did you train an AI model?**

No. I used pretrained embedding and language models. Ingestion prepares a searchable index; it is not model training.

**Why split the PDF?**

Small passages are easier to retrieve and fit into a model request. The overlap helps preserve nearby context when a section is split. The example settings are initial choices, not proven optimal values.

**Why Pinecone dimension 1536?**

The code explicitly requests embeddings with 1536 values. The index must use the same dimension. Both document chunks and questions use the same embedding model.

**Why use LangGraph instead of only functions?**

The assignment requires it, and the graph makes state, order and refusal branches explicit. Individual nodes can also be tested with controlled inputs.

**What does confidence_score mean?**

It is the best retrieved cosine similarity. The API uses the assignment's field name, but labels its meaning with score_type. It is not a calibrated probability of correctness.

**How do you prevent hallucinations?**

Only retrieved PDF text reaches the evidence-selection prompt. Python then requires exact quotations from valid chunk IDs and builds the displayed answer from those quotations. This prevents invented answer text from passing the quote check. The model could still select irrelevant or incomplete quotes, so I also review relevance and completeness.

**Why not return a fluent summary?**

The assignment emphasizes strict grounding. Extractive answers make provenance straightforward to check. A future version could support paraphrases with a separate claim-verification process, but that is not implemented here.

**Why can the app refuse a question that the book actually answers?**

Retrieval may miss the right passage, the threshold may be too high, or the selected chunks may be incomplete. I would inspect the retrieved text before tuning chunking, top-k or the threshold.

**What happens if I ingest the same PDF twice?**

The hash-based namespace and deterministic chunk IDs are reused, so vectors are updated rather than duplicated. Embeddings are still recomputed. Changed PDFs get separate namespaces.

**Does it remember previous questions?**

The UI displays history, but questions are answered independently. LangGraph state exists during each execution. Persistent conversation memory is not implemented.

**How was it tested?**

Twelve offline tests check graph routes, evidence validation, API handling and extraction behavior. Separately, the benchmark runner calls the real API for six queries. Report the live outcomes only after running and reviewing them yourself.

**What would you improve?**

Better extraction for tables/images, OCR, a larger labeled evaluation set, retrieval tuning, optional reranking, calibrated relevance decisions, and authentication/rate limiting before public hosting. These are future improvements, not current features.

## Final submission

Open your actual GitHub repository in a private/signed-out browser window. Confirm it is readable and contains the README and reviewed benchmark results. Submit only that repository URL as instructed; do not send your keys or a local browser address.

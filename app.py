"""Launch with: python -m streamlit run app.py"""
import streamlit as st
from rag import Chatbot

st.set_page_config(page_title="Agentic AI eBook Chatbot", page_icon="📖")
st.title("📖 Agentic AI eBook Chatbot")
st.caption("Ask a complete question. Answers contain verified quotations from the eBook.")
st.info("Relevance is cosine similarity, not the probability that an answer is correct. "
        "Questions are independent; displayed chat history is not sent to the model.")


@st.cache_resource
def load_bot():
    return Chatbot()


def show_result(result):
    st.text(result["answer"])  # Render source content as text, not active Markdown links.
    score = result["confidence_score"]
    st.caption(f"Status: {result['status']} | Top retrieval relevance: " +
               (f"{score:.3f}" if score is not None else "no matches"))
    for chunk in result["retrieved_chunks"]:
        label = f"PDF page {chunk['page']} · {chunk['id']} · relevance {chunk['score']:.3f}"
        with st.expander(label):
            st.caption("Eligible for answering" if chunk["eligible"] else "Below relevance threshold")
            st.text(chunk["text"])


if "history" not in st.session_state:
    st.session_state.history = []
if st.sidebar.button("Clear conversation"):
    st.session_state.history = []
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.text(turn["question"])
    with st.chat_message("assistant"):
        show_result(turn["result"])

question = st.chat_input("Ask about the Agentic AI eBook", max_chars=2000)
if question:
    with st.chat_message("user"):
        st.text(question)
    try:
        with st.spinner("Searching the eBook and checking evidence..."):
            result = load_bot().ask(question)
        st.session_state.history.append({"question": question, "result": result})
        with st.chat_message("assistant"):
            show_result(result)
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        st.error("The request failed. Check API keys, quota, model access and Pinecone status. "
                 "An API failure is not an eBook answer. See README troubleshooting.")

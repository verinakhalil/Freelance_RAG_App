"""
streamlit_app.py
------------------
Streamlit UI for the RAG assistant.

Locally: set OPENROUTER_API_KEY as an environment variable.
On Streamlit Cloud: set OPENROUTER_API_KEY / OPENROUTER_MODEL in
Manage app -> Secrets (TOML format). See project instructions.
"""

import streamlit as st
import rag

st.set_page_config(page_title="RAG Assistant", page_icon="📚")

# ---------------------------------------------------------------------
# Load the API key from Streamlit secrets if it wasn't already set via
# environment variable. This mirrors the required pattern exactly:
# never write the real key into a Python file.
# ---------------------------------------------------------------------
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass

st.title("📚 RAG Assistant")
st.caption("Ask a question about the documents in this project.")

if not rag.OPENROUTER_API_KEY:
    st.warning(
        "No OPENROUTER_API_KEY found. Add it in Streamlit Cloud "
        "(Manage app -> Secrets) or as a local environment variable."
    )

# ---------------------------------------------------------------------
# Cache the Chroma collection so it isn't reopened on every rerun
# ---------------------------------------------------------------------
@st.cache_resource
def load_collection():
    return rag.get_chroma_collection()


collection = load_collection()

# ---------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            contexts = rag.retrieve_context(query, k=rag.TOP_K, collection=collection)

            if not contexts:
                answer = (
                    "I couldn't find any relevant context. Make sure documents "
                    "have been added and the Chroma store has been built "
                    "(run 01 through 05 first)."
                )
            else:
                answer = rag.generate_answer(query, contexts)

            st.markdown(answer)

            if contexts:
                with st.expander("Sources used"):
                    for c in contexts:
                        st.markdown(f"- **{c['source']}** (distance: {c['distance']:.4f})")

    st.session_state.messages.append({"role": "assistant", "content": answer})

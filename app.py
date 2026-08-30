"""
Simple Streamlit front-end for the Nokia 1830 PSS RAG pipeline.

Reuses the same index built by 03_embed_index.py and the same grounded
system prompt as 04_retrieve_query.py, just wrapped in a UI instead of a
CLI. Run from the scripts/ folder:

    streamlit run app.py
"""
import json

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import os
   
INDEX_DIR = "../index"
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are a Nokia 1830 PSS site-engineering assistant. You answer ONLY using \
the CONTEXT chunks provided below, which are excerpts from the official 1830 PSS Product \
Information and Planning Guide.

Rules (follow all of them):
1. Base your answer strictly on the CONTEXT. Do not use outside knowledge of Nokia \
products, and do not guess or estimate a plausible-sounding number.
2. Every factual claim in your answer must be followed by a citation in the form \
[Section: <heading>, p.<page>], using the heading/page metadata attached to the chunk \
you drew it from.
3. If the CONTEXT does not contain the answer, reply EXACTLY: \
"Not found in the provided document." — do not soften this into a guess, and do not \
apologize at length. It is always better to say this than to invent a number.
4. Keep answers concise: 1-4 sentences plus citations, unless the question needs a list.
"""

USER_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

Answer using only the CONTEXT above, with citations, following the rules in the system prompt."""


@st.cache_resource(show_spinner="Loading embedding model...")
def load_model(model_name: str):
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner="Loading index...")
def load_index(index_dir: str):
    embeddings = np.load(f"{index_dir}/embeddings.npy")
    metadata = [json.loads(l) for l in open(f"{index_dir}/metadata.jsonl", encoding="utf-8")]
    manifest = json.load(open(f"{index_dir}/manifest.json"))
    return embeddings, metadata, manifest


def top_k(question_vec: np.ndarray, embeddings: np.ndarray, k: int):
    sims = embeddings @ question_vec
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]


def format_context(hits, metadata):
    blocks = []
    for i, score in hits:
        m = metadata[i]
        blocks.append(f"[Section: {m['heading']}, p.{m['page']}]\n{m['text']}")
    return "\n\n---\n\n".join(blocks)


def call_groq(api_key: str, system_prompt: str, user_prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content


st.set_page_config(page_title="1830 PSS RAG Assistant", page_icon="📡", layout="centered")
st.title("📡 1830 PSS Site-Engineering Assistant")
st.caption("Answers are grounded in Chapters 1-2 of the 1830 PSS Product Information & Planning Guide.")

with st.sidebar:
    st.subheader("Settings")
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=4)
    show_chunks = st.checkbox("Show retrieved chunks", value=True)

try:
    embeddings, metadata, manifest = load_index(INDEX_DIR)
    model = load_model(manifest["model"])
    st.sidebar.success(f"Index loaded: {manifest['n_chunks']} chunks")
except FileNotFoundError:
    st.error(
        f"No index found at `{INDEX_DIR}`. Run `python 03_embed_index.py` first "
        "to build embeddings.npy, metadata.jsonl, and manifest.json."
    )
    st.stop()

question = st.text_input(
    "Ask a question about the 1830 PSS shelves, cards, or hardware",
    placeholder="e.g. How many slots does the 1830 PSS-8 shelf provide?",
)
ask = st.button("Ask", type="primary", disabled=not question)

if ask:
    if not GROQ_API_KEY:
        st.warning("Set GROQ_API_KEY at the top of app.py to your real Groq API key before asking questions.")
        st.stop()

    with st.spinner("Retrieving relevant sections..."):
        q_vec = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
        hits = top_k(q_vec, embeddings, k)
        context = format_context(hits, metadata)

    if show_chunks:
        st.subheader("Retrieved chunks")
        for i, score in hits:
            m = metadata[i]
            with st.expander(f"{m['heading']} — p.{m['page']} (score {score:.3f})"):
                st.write(m["text"])

    with st.spinner("Generating grounded answer..."):
        try:
            user_prompt = USER_TEMPLATE.format(context=context, question=question)
            generated = call_groq(GROQ_API_KEY, SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

    st.subheader("Answer")
    st.markdown(generated)
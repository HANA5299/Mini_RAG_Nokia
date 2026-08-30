"""
Part C - Query-time half of the pipeline.

  1. Embed the incoming question with the SAME model used for indexing.
  2. Manual top-k cosine similarity search (plain numpy — no FAISS/Chroma
     needed for a few hundred chunks; that's an optional comparison, not
     a replacement, per the assignment).
  3. Build a grounded, guardrailed prompt and call an LLM for generation.
     Retrieval + grounding logic is all local code; only the final text
     completion is delegated to the LLM. Uses Groq's free tier (matches
     prior tooling) — swap GROQ_MODEL / the client call for any other
     provider without touching the retrieval code.

Usage:
    export GROQ_API_KEY=...
    python 04_retrieve_query.py --q "How many slots does the 1830 PSS-8 shelf provide?"
"""
import argparse
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

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


def load_index(index_dir: str):
    embeddings = np.load(f"{index_dir}/embeddings.npy")
    metadata = [json.loads(l) for l in open(f"{index_dir}/metadata.jsonl", encoding="utf-8")]
    manifest = json.load(open(f"{index_dir}/manifest.json"))
    return embeddings, metadata, manifest


def top_k(question_vec: np.ndarray, embeddings: np.ndarray, k: int):
    # embeddings and question_vec are already L2-normalized -> dot product == cosine similarity
    sims = embeddings @ question_vec
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]


def format_context(hits, metadata):
    blocks = []
    for i, score in hits:
        m = metadata[i]
        blocks.append(f"[Section: {m['heading']}, p.{m['page']}]\n{m['text']}")
    return "\n\n---\n\n".join(blocks)


def call_groq(system_prompt: str, user_prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content


def answer(question: str, index_dir: str = "../index", k: int = 4, model=None):
    embeddings, metadata, manifest = load_index(index_dir)
    if model is None:
        model = SentenceTransformer(manifest["model"])
    q_vec = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
    hits = top_k(q_vec, embeddings, k)
    context = format_context(hits, metadata)
    user_prompt = USER_TEMPLATE.format(context=context, question=question)
    generated = call_groq(SYSTEM_PROMPT, user_prompt)
    return {
        "question": question,
        "retrieved": [{"chunk_id": metadata[i]["chunk_id"], "heading": metadata[i]["heading"],
                        "page": metadata[i]["page"], "score": round(s, 3)} for i, s in hits],
        "answer": generated,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--index_dir", default="../index")
    args = ap.parse_args()
    result = answer(args.q, args.index_dir, args.k)
    print(json.dumps(result, indent=2, ensure_ascii=False))

"""
Part B - Embed every chunk with sentence-transformers and persist the index
so it does not need to be rebuilt on every run ("Cache & version your
embeddings" from the Best Practices slide).

We store:
  - index/embeddings.npy     float32 matrix, one row per chunk (L2-normalized)
  - index/metadata.jsonl     chunk_id, heading, page, text (parallel to rows)
  - index/manifest.json      model name + chunk-file hash, so 04_retrieve.py
                              can detect a stale index and refuse to use it
                              silently instead of returning wrong results.

Similarity search itself is implemented from scratch with plain numpy dot
products in 04_retrieve_query.py — this script only produces the vectors.
"""
import argparse
import hashlib
import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def main(chunks_path: str, index_dir: str):
    chunks = [json.loads(l) for l in open(chunks_path, encoding="utf-8")]
    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(MODEL_NAME)
    # normalize_embeddings=True -> dot product == cosine similarity at query time
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)

    np.save(f"{index_dir}/embeddings.npy", embeddings)
    with open(f"{index_dir}/metadata.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    manifest = {
        "model": MODEL_NAME,
        "n_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "chunks_file_hash": file_hash(chunks_path),
    }
    with open(f"{index_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Indexed {len(chunks)} chunks, dim={embeddings.shape[1]} -> {index_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="../data/chunks.jsonl")
    ap.add_argument("--index_dir", default="../index")
    args = ap.parse_args()
    main(args.chunks, args.index_dir)

"""ColBERT + NanoBEIR retrieval benchmark: Native MaxSim vs FDE.

This script reproduces the evaluation from sionic-ai/muvera-py using the
simplified Muvera API. It compares:

  1. ColBERT native late-interaction (MaxSim) retrieval
  2. ColBERT + Muvera FDE single-vector retrieval

Embeddings (both documents and queries) are cached to disk on first run
so that subsequent runs skip the expensive model inference entirely and
only re-run the FDE / retrieval logic.

Prerequisites
-------------
Install additional dependencies before running:

    pip install pylate datasets torch

Usage
-----
    python examples/colbert_nanobeir.py
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from pylate.models import ColBERT as PylateColBERT

from muvera import Muvera

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_REPO_ID = "zeta-alpha-ai/NanoFiQA2018"
COLBERT_MODEL_NAME = "raphaelsty/neural-cherche-colbert"
TOP_K = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path("examples/.cache")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def _cache_key(model_name: str, dataset_id: str, split: str) -> str:
    raw = f"{model_name}::{dataset_id}::{split}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]  # noqa: S324
    safe_model = model_name.replace("/", "_")
    return f"{safe_model}_{split}_{h}.npz"


def _save_embeddings(path: Path, ids: list[str], embeddings: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"ids": np.array(ids, dtype=object), "n": np.array([len(embeddings)])}
    for i, emb in enumerate(embeddings):
        data[f"emb_{i}"] = emb
    np.savez(path, **data)
    logging.info(f"Cached {len(embeddings)} embeddings to {path}")


def _load_embeddings(path: Path) -> tuple[list[str], list[np.ndarray]] | None:
    if not path.exists():
        return None
    logging.info(f"Loading cached embeddings from {path}")
    data = np.load(path, allow_pickle=True)
    ids = data["ids"].tolist()
    n = int(data["n"][0])
    return ids, [data[f"emb_{i}"] for i in range(n)]


def _encode_cached(
    model: PylateColBERT,
    ids: list[str],
    texts: list[str],
    *,
    is_query: bool,
    cache_path: Path,
    as_numpy: bool,
) -> tuple[list[str], list]:
    cached = _load_embeddings(cache_path)
    if cached is not None:
        cached_ids, embs = cached
        if not as_numpy:
            embs = [torch.from_numpy(e) for e in embs]
        return cached_ids, embs

    label = "queries" if is_query else "documents"
    logging.info(f"Encoding {len(texts)} {label} (first run, will be cached)...")
    embeddings = model.encode(
        sentences=texts,
        is_query=is_query,
        convert_to_numpy=as_numpy,
        convert_to_tensor=not as_numpy,
        normalize_embeddings=True,
    )

    np_embs = embeddings if as_numpy else [e.cpu().numpy() for e in embeddings]
    _save_embeddings(cache_path, ids, np_embs)
    return ids, embeddings


def encode_corpus_cached(
    model: PylateColBERT, corpus: dict, *, as_numpy: bool = True
) -> tuple[list[str], list]:
    doc_ids = list(corpus.keys())
    texts = [
        f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip() for did in doc_ids
    ]
    cache_path = CACHE_DIR / _cache_key(COLBERT_MODEL_NAME, DATASET_REPO_ID, "docs")
    return _encode_cached(
        model, doc_ids, texts, is_query=False, cache_path=cache_path, as_numpy=as_numpy
    )


def encode_queries_cached(
    model: PylateColBERT, queries: dict[str, str], *, as_numpy: bool = True
) -> tuple[list[str], list]:
    qids = list(queries.keys())
    texts = [queries[qid] for qid in qids]
    cache_path = CACHE_DIR / _cache_key(COLBERT_MODEL_NAME, DATASET_REPO_ID, "queries")
    return _encode_cached(
        model, qids, texts, is_query=True, cache_path=cache_path, as_numpy=as_numpy
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_nanobeir_dataset(repo_id: str) -> tuple[dict, dict, dict]:
    logging.info(f"Loading dataset: '{repo_id}'...")
    corpus_ds = load_dataset(repo_id, "corpus", split="train")
    queries_ds = load_dataset(repo_id, "queries", split="train")
    qrels_ds = load_dataset(repo_id, "qrels", split="train")

    corpus = {
        row["_id"]: {"title": row.get("title", ""), "text": row.get("text", "")}
        for row in corpus_ds
    }
    queries = {row["_id"]: row["text"] for row in queries_ds}

    # Properly accumulate qrels (a query can have multiple relevant documents)
    qrels = defaultdict(dict)
    for row in qrels_ds:
        qrels[str(row["query-id"])][str(row["corpus-id"])] = 1
    qrels = dict(qrels)

    total_relevant = sum(len(docs) for docs in qrels.values())
    logging.info(
        f"Loaded {len(corpus)} documents, {len(queries)} queries, {total_relevant} relevance judgments."
    )
    return corpus, queries, qrels


def evaluate_recall(results: dict, qrels: dict, k: int) -> float:
    hits, total = 0, 0
    for query_id, ranked_docs in results.items():
        relevant = set(qrels.get(str(query_id), {}).keys())
        if not relevant:
            continue
        total += 1
        top_k = set(list(ranked_docs.keys())[:k])
        if not relevant.isdisjoint(top_k):
            hits += 1
    return hits / total if total > 0 else 0.0


def to_numpy(tensor_or_array: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().detach().numpy().astype(np.float32)
    return np.asarray(tensor_or_array, dtype=np.float32)


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------


class ColBERTNativeRetriever:
    """ColBERT late-interaction (MaxSim) retrieval."""

    def __init__(self, model_name: str = COLBERT_MODEL_NAME) -> None:
        self.model = PylateColBERT(model_name_or_path=model_name, device=DEVICE)
        self.doc_embeddings: dict[str, torch.Tensor] = {}
        self.query_embeddings: dict[str, torch.Tensor] = {}

    def index(self, corpus: dict) -> None:
        doc_ids, embeddings = encode_corpus_cached(self.model, corpus, as_numpy=False)
        self.doc_embeddings = dict(zip(doc_ids, embeddings))

    def cache_queries(self, queries: dict[str, str]) -> None:
        qids, embeddings = encode_queries_cached(self.model, queries, as_numpy=False)
        self.query_embeddings = dict(zip(qids, embeddings))

    def search(self, qid: str) -> dict[str, float]:
        q_emb = self.query_embeddings[qid]
        scores = {}
        with torch.no_grad():
            for doc_id, d_emb in self.doc_embeddings.items():
                sim = torch.einsum("sh,th->st", q_emb.to(DEVICE), d_emb.to(DEVICE))
                scores[doc_id] = sim.max(dim=1).values.sum().item()
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


class ColBERTFdeRetriever:
    """ColBERT + Muvera FDE single-vector retrieval."""

    def __init__(self, model_name: str = COLBERT_MODEL_NAME) -> None:
        self.model = PylateColBERT(model_name_or_path=model_name, device=DEVICE)
        self.encoder: Muvera | None = None
        self.fde_index: np.ndarray | None = None
        self.doc_ids: list[str] = []
        self.query_embeddings: dict[str, np.ndarray] = {}

    def index(self, corpus: dict) -> None:
        self.doc_ids, embeddings = encode_corpus_cached(self.model, corpus, as_numpy=True)

        dim = embeddings[0].shape[1]
        self.encoder = Muvera(
            num_repetitions=20,
            num_simhash_projections=7,
            dimension=dim,
            fill_empty_partitions=True,
            seed=42,
        )
        logging.info(
            f"[FDE] Generating document FDEs "
            f"(dim={dim}, output_dim={self.encoder.output_dimension})..."
        )

        # Pass list of variable-length arrays directly (vectorised batch).
        self.fde_index = self.encoder.encode_documents(embeddings)

    def cache_queries(self, queries: dict[str, str]) -> None:
        qids, embeddings = encode_queries_cached(self.model, queries, as_numpy=True)
        self.query_embeddings = {qid: to_numpy(emb) for qid, emb in zip(qids, embeddings)}

    def search(self, qid: str) -> dict[str, float]:
        q_emb = self.query_embeddings[qid]
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)

        query_fde = self.encoder.encode_queries(q_emb)
        scores = self.fde_index @ query_fde
        return dict(sorted(zip(self.doc_ids, scores.tolist()), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the ColBERT + NanoBEIR benchmark."""
    corpus, queries, qrels = load_nanobeir_dataset(DATASET_REPO_ID)

    retrievers = {
        "ColBERT (Native MaxSim)": ColBERTNativeRetriever(),
        "ColBERT + Muvera FDE": ColBERTFdeRetriever(),
    }

    timings: dict[str, dict] = {}
    all_results: dict[str, dict] = {}

    # --- Indexing ---
    logging.info("=== PHASE 1: INDEXING ===")
    for name, retriever in retrievers.items():
        t0 = time.perf_counter()
        retriever.index(corpus)
        elapsed = time.perf_counter() - t0
        timings[name] = {"indexing": elapsed}
        logging.info(f"'{name}' indexing: {elapsed:.2f}s")

    # --- Cache queries ---
    logging.info("=== PHASE 1.5: CACHE QUERIES ===")
    for retriever in retrievers.values():
        retriever.cache_queries(queries)

    # --- Search ---
    logging.info("=== PHASE 2: SEARCH ===")
    for name, retriever in retrievers.items():
        logging.info(f"Searching with '{name}' ({len(queries)} queries)...")
        query_times = []
        results = {}
        for qid in queries:
            t0 = time.perf_counter()
            results[str(qid)] = retriever.search(str(qid))
            query_times.append(time.perf_counter() - t0)

        timings[name]["avg_query_ms"] = np.mean(query_times) * 1000
        all_results[name] = results
        logging.info(f"'{name}' avg query: {timings[name]['avg_query_ms']:.2f}ms")

    # --- Report ---
    print()
    print("=" * 85)
    print(f"{'RESULTS':^85}")
    print(f"{'(' + DATASET_REPO_ID + ')':^85}")
    print("=" * 85)
    recall_header = f"Recall@{TOP_K}"
    print(f"{'Retriever':<30} | {'Index (s)':<12} | {'Query (ms)':<12} | {recall_header:<10}")
    print("-" * 85)

    for name in retrievers:
        recall = evaluate_recall(all_results[name], qrels, k=TOP_K)
        print(
            f"{name:<30} | "
            f"{timings[name]['indexing']:<12.2f} | "
            f"{timings[name]['avg_query_ms']:<12.2f} | "
            f"{recall:<10.4f}"
        )

    print("=" * 85)


if __name__ == "__main__":
    main()

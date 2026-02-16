"""Generate small ColBERT embedding fixtures for testing.

This script creates a small sample of real ColBERT embeddings from NanoBEIR
for use in tests. The embeddings are cached to avoid slow model inference
during test runs.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from pylate.models import ColBERT as PylateColBERT

# Configuration
NUM_QUERIES = 5
NUM_DOCUMENTS_PER_QUERY = 10  # Take top N relevant docs per query
FIXTURE_DIR = Path("tests/fixtures/colbert_nanobeir")
DATASET_REPO_ID = "zeta-alpha-ai/NanoFiQA2018"
COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"


def main():
    """Generate test fixtures."""
    print(f"Loading dataset from {DATASET_REPO_ID}...")

    # Load full datasets
    corpus_ds = load_dataset(DATASET_REPO_ID, "corpus", split="train")
    queries_ds = load_dataset(DATASET_REPO_ID, "queries", split="train")
    qrels_ds = load_dataset(DATASET_REPO_ID, "qrels", split="train")

    # Build qrels map
    print("Building qrels mapping...")
    qrels_map = defaultdict(dict)
    for row in qrels_ds:
        qid = str(row["query-id"])
        did = str(row["corpus-id"])
        qrels_map[qid][did] = 1

    # Create corpus and queries dicts
    corpus = {row["_id"]: row for row in corpus_ds}
    queries = {row["_id"]: row for row in queries_ds}

    # Select queries with enough relevant documents
    selected_query_ids = []
    for qid in sorted(qrels_map.keys()):
        if len(qrels_map[qid]) >= 3:  # At least 3 relevant docs
            selected_query_ids.append(qid)
            if len(selected_query_ids) >= NUM_QUERIES:
                break

    print(f"Selected {len(selected_query_ids)} queries with sufficient relevance judgments")

    # Collect all relevant document IDs
    selected_doc_ids = set()
    final_qrels = {}
    for qid in selected_query_ids:
        relevant_docs = list(qrels_map[qid].keys())[:NUM_DOCUMENTS_PER_QUERY]
        selected_doc_ids.update(relevant_docs)
        final_qrels[qid] = {did: 1 for did in relevant_docs}

    selected_doc_ids = sorted(selected_doc_ids)

    print(f"Selected {len(selected_doc_ids)} documents")
    print(f"Total relevance judgments: {sum(len(v) for v in final_qrels.values())}")

    # Extract texts
    doc_texts = [
        f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip()
        for did in selected_doc_ids
    ]
    query_texts = [queries[qid]["text"] for qid in selected_query_ids]

    # Encode with ColBERT
    print(f"Loading ColBERT model: {COLBERT_MODEL_NAME}...")
    model = PylateColBERT(model_name_or_path=COLBERT_MODEL_NAME, device="cpu")

    print(f"Encoding {len(doc_texts)} documents...")
    doc_embeddings = model.encode(
        sentences=doc_texts,
        is_query=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"Encoding {len(query_texts)} queries...")
    query_embeddings = model.encode(
        sentences=query_texts,
        is_query=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Save fixtures
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving fixtures to {FIXTURE_DIR}...")

    # Save document embeddings
    doc_data = {"ids": np.array(selected_doc_ids, dtype=object), "n": len(doc_embeddings)}
    for i, emb in enumerate(doc_embeddings):
        doc_data[f"emb_{i}"] = emb
    np.savez_compressed(FIXTURE_DIR / "documents.npz", **doc_data)

    # Save query embeddings
    query_data = {"ids": np.array(selected_query_ids, dtype=object), "n": len(query_embeddings)}
    for i, emb in enumerate(query_embeddings):
        query_data[f"emb_{i}"] = emb
    np.savez_compressed(FIXTURE_DIR / "queries.npz", **query_data)

    # Save qrels
    import json

    with open(FIXTURE_DIR / "qrels.json", "w") as f:
        json.dump(final_qrels, f, indent=2)

    # Print statistics
    print("\n" + "=" * 60)
    print("Fixture Generation Complete")
    print("=" * 60)
    print(f"Documents: {len(doc_embeddings)}")
    print(f"Queries: {len(query_embeddings)}")
    print(f"Relevance judgments: {sum(len(v) for v in final_qrels.values())}")
    print("\nSample relevance judgments:")
    for qid in list(final_qrels.keys())[:2]:
        print(f"  Query {qid}: {len(final_qrels[qid])} relevant docs")
    print("\nDocument embedding shapes:")
    for i in range(min(3, len(doc_embeddings))):
        print(f"  Doc {i}: {doc_embeddings[i].shape}")
    print("\nQuery embedding shapes:")
    for i in range(min(3, len(query_embeddings))):
        print(f"  Query {i}: {query_embeddings[i].shape}")
    print(f"\nFixtures saved to: {FIXTURE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Basic usage example: sanity check with random data.

This script demonstrates the core Muvera API and evaluates how well
FDE-based retrieval recovers the true Chamfer (MaxSim) nearest neighbors.

MuVERA approximates the Chamfer similarity:
    Chamfer(Q, P) = sum_{q in Q} max_{p in P} <q, p>

This is achieved via an asymmetric encoding: query FDEs use SUM
aggregation per partition, while document FDEs use AVERAGE (centroid)
aggregation. See https://arxiv.org/abs/2405.19504 for details.

Note: Even on random data, FDE Recall@N significantly outperforms
random baselines, though real ColBERT embeddings yield much better
results due to their inherent structure.
"""

import numpy as np

from muvera import Muvera


def chamfer(query: np.ndarray, document: np.ndarray) -> float:
    """Compute the Chamfer similarity (MaxSim) between a query and a document.

    Chamfer(Q, P) = sum_{q in Q} max_{p in P} <q, p>

    Parameters
    ----------
    query : np.ndarray
        Shape ``(num_query_vectors, dimension)``.
    document : np.ndarray
        Shape ``(num_doc_vectors, dimension)``.

    Returns
    -------
    float
        Chamfer similarity score.
    """
    sim_matrix = query @ document.T
    return float(sim_matrix.max(axis=1).sum())


def recall_at_n(
    fde_scores: np.ndarray,
    true_scores: np.ndarray,
    n: int,
) -> float:
    """Compute Recall@N: fraction of queries whose true top-1 is in FDE top-N.

    Parameters
    ----------
    fde_scores : np.ndarray
        Shape ``(num_queries, num_documents)``.
    true_scores : np.ndarray
        Shape ``(num_queries, num_documents)``.
    n : int
        Number of top candidates to consider.

    Returns
    -------
    float
        Recall@N.
    """
    num_queries = fde_scores.shape[0]
    hits = 0
    for q in range(num_queries):
        true_top1 = np.argmax(true_scores[q])
        fde_top_n = set(np.argsort(fde_scores[q])[-n:])
        if true_top1 in fde_top_n:
            hits += 1
    return hits / num_queries


def main() -> None:
    """Run the basic usage example."""
    rng = np.random.default_rng(0)
    dimension = 128
    num_documents = 1000
    num_queries = 100
    num_doc_vectors = 80
    num_query_vectors = 32

    # --- Generate random normalized data (ColBERT-style unit vectors) ---
    print("Generating random normalized embeddings...")
    documents_raw = rng.standard_normal((num_documents, num_doc_vectors, dimension)).astype(
        np.float32
    )
    queries_raw = rng.standard_normal((num_queries, num_query_vectors, dimension)).astype(
        np.float32
    )

    documents = documents_raw / np.linalg.norm(documents_raw, axis=-1, keepdims=True)
    queries = queries_raw / np.linalg.norm(queries_raw, axis=-1, keepdims=True)

    # --- Initialize encoder (Pareto-optimal params from the paper) ---
    encoder = Muvera(
        num_repetitions=20,
        num_simhash_projections=5,
        dimension=dimension,
        fill_empty_partitions=True,
        seed=42,
    )
    print(f"Encoder: {encoder}")
    print(f"Output dimension: {encoder.output_dimension:,}")
    print()

    # --- Encode ---
    print("Encoding documents...")
    doc_fdes = encoder.encode_documents(documents)
    print("Encoding queries...")
    query_fdes = encoder.encode_queries(queries)
    print(f"Document FDEs shape: {doc_fdes.shape}")
    print(f"Query FDEs shape:    {query_fdes.shape}")
    print()

    # --- Compute FDE scores ---
    fde_scores = query_fdes @ doc_fdes.T

    # --- Compute true Chamfer (MaxSim) scores ---
    print("Computing true Chamfer (MaxSim) scores for all query-document pairs...")
    true_scores = np.zeros((num_queries, num_documents), dtype=np.float32)
    for q in range(num_queries):
        for d in range(num_documents):
            true_scores[q, d] = chamfer(queries[q], documents[d])

    # --- Recall@N evaluation ---
    ks = [1, 10, 50, 100, 200]
    random_baseline = {n: min(n / num_documents, 1.0) for n in ks}

    print("=" * 60)
    print(f"{'Recall@N':^60}")
    print(f"{'(N documents = ' + str(num_documents) + ')':^60}")
    print("=" * 60)
    print(f"{'N':<10} {'FDE Recall':>15} {'Random Baseline':>18}")
    print("-" * 60)

    for n in ks:
        r = recall_at_n(fde_scores, true_scores, n)
        print(f"{n:<10} {r:>15.1%} {random_baseline[n]:>18.1%}")

    print("=" * 60)
    print()
    print(
        "Note: MuVERA approximates Chamfer similarity (MaxSim).\n"
        "Even on random data, FDE recall significantly exceeds the\n"
        "random baseline. On real ColBERT embeddings with inherent\n"
        "structure, recall is much higher (see colbert_nanobeir.py)."
    )


if __name__ == "__main__":
    main()

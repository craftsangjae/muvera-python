# MuVERA

A Python implementation of **Mu**lti-**Ve**ctor **R**etrieval via Fixed Dimensional Encoding **A**lgorithm.

Converts multi-vector embeddings (point clouds) into fixed-dimensional single vectors, enabling the use of existing single-vector search infrastructure (MIPS, ANN, etc.) as-is.

## Why this library?

The original MuVERA algorithm is described in a [research paper](https://arxiv.org/abs/2405.19504) and implemented in C++ within Google's [graph-mining](https://github.com/google/graph-mining) repository. While a Python reference exists, it exposes low-level config objects and separate functions for queries vs. documents, making it cumbersome to integrate into real workflows.

This library wraps the full algorithm behind a **single `Muvera` class** with a minimal, intuitive interface — initialize once, then call `encode_documents()` and `encode_queries()`. No config dataclasses, no encoding-type enums, no manual seed juggling. Just NumPy arrays in, NumPy arrays out.

## Installation

```bash
pip install muvera-python
```

Development install:

```bash
git clone https://github.com/craftsangjae/muvera-python.git
cd muvera-python
pip install -e .
```

## Quick Start

```python
import numpy as np
from muvera import Muvera

# Initialize encoder
encoder = Muvera(
    num_repetitions=10,
    num_simhash_projections=4,
    dimension=128,
    seed=42,
)

# Encode documents (batch)
# shape: (num_documents, num_vectors_per_doc, embedding_dim)
documents = np.random.randn(100, 80, 128).astype(np.float32)
doc_fdes = encoder.encode_documents(documents)  # (100, output_dimension)

# Encode queries (batch)
queries = np.random.randn(10, 32, 128).astype(np.float32)
query_fdes = encoder.encode_queries(queries)  # (10, output_dimension)

# Compute similarity (dot product)
scores = query_fdes @ doc_fdes.T  # (10, 100)
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_repetitions` | 20 | Number of FDE repetitions. Higher values improve accuracy but increase output dimension |
| `num_simhash_projections` | 5 | Number of SimHash projections. Number of partitions = 2^n |
| `dimension` | 16 | Input embedding dimension |
| `projection_type` | `"identity"` | `"identity"` or `"ams_sketch"` |
| `projection_dimension` | None | Projected dimension when using AMS Sketch |
| `fill_empty_partitions` | True | Whether to fill empty partitions with the nearest vector |
| `final_projection_dimension` | None | Final dimension reduction via Count Sketch |
| `seed` | 42 | Random seed for reproducibility |

## Benchmark

End-to-end retrieval on [NanoFiQA2018](https://huggingface.co/datasets/zeta-alpha-ai/NanoFiQA2018) (4598 documents, 50 queries) using `raphaelsty/neural-cherche-colbert` (dim=128):

```
=====================================================================================
                                       RESULTS
                            (zeta-alpha-ai/NanoFiQA2018)
=====================================================================================
Retriever                      | Index (s)    | Query (ms)   | Recall@25
-------------------------------------------------------------------------------------
ColBERT (Native MaxSim)        | 240.04       | 836.94       | 0.8400
ColBERT + Muvera FDE           | 77.29        | 69.48        | 0.7600
=====================================================================================
```

FDE achieves **90% of native MaxSim recall** while being **12x faster** at query time. See `examples/colbert_nanobeir.py` to reproduce.

## Acknowledgments

This library was inspired by [sionic-ai/muvera-py](https://github.com/sionic-ai/muvera-py), the first Python implementation of the MuVERA algorithm. Their faithful port of the C++ reference made it possible to validate correctness and understand the algorithm deeply. This project builds on that foundation with a simplified API designed for easier integration.

## References

- [MuVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings](https://arxiv.org/abs/2405.19504)
- [Google graph-mining C++ implementation](https://github.com/google/graph-mining/blob/main/sketching/point_cloud/fixed_dimensional_encoding.cc)
- [sionic-ai/muvera-py](https://github.com/sionic-ai/muvera-py)
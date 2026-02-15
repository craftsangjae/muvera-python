# MuVERA

A Python implementation of **Mu**lti-**Ve**ctor **R**etrieval via Fixed Dimensional Encoding **A**lgorithm.

Converts multi-vector embeddings (point clouds) into fixed-dimensional single vectors, enabling the use of existing single-vector search infrastructure (MIPS, ANN, etc.) as-is.

## Why this library?

The original MuVERA algorithm is described in a [research paper](https://arxiv.org/abs/2405.19504) and implemented in C++ within Google's [graph-mining](https://github.com/google/graph-mining) repository. While a Python reference exists, it exposes low-level config objects and separate functions for queries vs. documents, making it cumbersome to integrate into real workflows.

This library wraps the full algorithm behind a **single `Muvera` class** with a minimal, intuitive interface — initialize once, then call `encode_documents()` and `encode_queries()`. No config dataclasses, no encoding-type enums, no manual seed juggling. Just NumPy arrays in, NumPy arrays out.

## Installation

```bash
pip install muvera
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

## References

- [MuVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings](https://arxiv.org/abs/2405.19504)
- [Google graph-mining C++ implementation](https://github.com/google/graph-mining/blob/main/sketching/point_cloud/fixed_dimensional_encoding.cc)

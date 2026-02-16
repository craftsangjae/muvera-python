# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MuVERA (Multi-Vector Retrieval via Fixed Dimensional Encoding Algorithm) is a Python library that converts multi-vector embeddings (point clouds) into fixed-dimensional single vectors. This enables using existing single-vector search infrastructure (MIPS, ANN) without modification.

### Project Goals

**Simplicity First**: This project pursues a simple, intuitive interface to make MuVERA easy to use. Unlike the reference implementation which exposes low-level config objects and separate functions for queries vs. documents, this library wraps everything behind a single `Muvera` class:
- `encode_documents()` - encodes document embeddings using AVERAGE method
- `encode_queries()` - encodes query embeddings using SUM method

No config dataclasses, no encoding-type enums, no manual seed juggling. Just NumPy arrays in, NumPy arrays out.

**Distribution Plan**: The library will be published to PyPI for easy installation via `pip install muvera-python`.

**Key use case**: Efficiently encode ColBERT-style multi-vector embeddings for retrieval without specialized infrastructure.

## Git & PR Conventions
- **Do NOT** add `Co-Authored-By` lines to commit messages.
- **Do NOT** add "Generated with Claude Code" or similar attribution to PR descriptions.

## Development Commands

### Environment Activation
**IMPORTANT**: Always activate the virtual environment before running any commands:
```bash
source .venv/bin/activate
```

### Post-Code-Writing Checklist
**IMPORTANT**: After writing or modifying any code, always run:
```bash
ruff check .                  # Lint check (must pass)
pytest                        # Tests (must pass)
```

### Setup
```bash
pip install maturin            # Required for building Rust extension
maturin develop --release      # Build Rust extension (needs Rust toolchain)
pip install -e ".[dev]"        # Install with dev dependencies
```

### Testing
```bash
pytest                        # Run all tests
pytest tests/test_muvera.py   # Run specific test file
pytest -v --tb=short          # Verbose output with short traceback
pytest -k test_name           # Run specific test by name
```

### Code Quality
```bash
ruff check .                  # Lint all files
ruff check --fix .            # Lint and auto-fix issues
ruff format .                 # Format code
mypy muvera                   # Type checking
pre-commit run --all-files    # Run all pre-commit hooks
```

### Running Examples
```bash
python examples/basic_usage.py
python examples/colbert_nanobeir.py
```

### Deployment

**Version bump → merge → auto-release:**

```bash
# 1. Update version in pyproject.toml
# version = "0.2.0"

# 2. Create PR and merge to main
# GitHub Actions will automatically:
# - Detect the new version (tag doesn't exist yet)
# - Run full test suite
# - Create git tag v0.2.0
# - Build wheel and sdist
# - Publish to PyPI via OIDC
# - Create GitHub Release
#
# If version is unchanged, all release steps are skipped.
```

**Local build (for testing):**
```bash
maturin build --release        # Build wheel with Rust extension
```

## Architecture

### Core Components

**`muvera/muvera.py`** - Main `Muvera` class implementing Fixed Dimensional Encoding (FDE)
- Two encoding paths: single document, variable-length batch
- Document encoding uses AVERAGE aggregation within partitions
- Query encoding uses SUM aggregation within partitions
- Optional final dimensionality reduction via Count Sketch
- Hot-path methods (`_aggregate_single`, `_scatter_add`, `_fill_empty_batch`) delegate to Rust kernels when available

**`muvera/helper.py`** - Low-level utilities (not public API)
- Gray code manipulation for partition indexing
- Random projection matrices (SimHash, AMS Sketch, Count Sketch)
- Vectorized batch partition indexing
- `partition_index_gray` and `partition_indices_gray_batch` delegate to Rust when available

**`src/`** - Rust extension module (`muvera._rust_kernels`) via PyO3/maturin
- `gray_code.rs` — Gray code append and binary conversion
- `partition.rs` — Single and batch Gray-code partition indexing
- `scatter.rs` — Scatter-add kernel for batch aggregation
- `fill_empty.rs` — Single-point-cloud aggregation + batch empty partition filling
- `lib.rs` — PyO3 module definition exposing 5 functions

**`muvera/_rust_kernels.pyi`** - Type stubs for the Rust extension module

### Algorithm Flow

1. **SimHash Projection**: Maps each vector to a partition using random Gaussian projections
2. **Partition Assignment**: Uses Gray code to assign vectors to one of `2^num_simhash_projections` partitions
3. **Inner Projection**: Optionally reduces dimension via AMS Sketch (or uses identity)
4. **Aggregation**:
   - Documents: compute centroid (average) of vectors in each partition
   - Queries: compute sum of vectors in each partition
5. **Empty Partition Filling** (documents only): Fill empty partitions with nearest vector by Hamming distance
6. **Repetitions**: Repeat steps 1-5 with different random seeds, concatenating results
7. **Final Projection** (optional): Apply Count Sketch to reduce final dimension

### Rust Acceleration

Performance-critical inner loops are implemented in Rust via PyO3, with automatic fallback to pure Python:

```python
# muvera/__init__.py
try:
    import muvera._rust_kernels
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
```

**Accelerated functions:**
| Rust function | Python fallback | Speedup |
|---|---|---|
| `aggregate_single` | `Muvera._aggregate_single_python` | 8-17x (single doc) |
| `scatter_add_partitions` | `Muvera._scatter_add` (np.add.at loop) | 1-2.5x (batch) |
| `fill_empty_partitions_batch` | `Muvera._fill_empty_batch` (Python loop) | 1-2.5x (batch) |
| `partition_index_gray` | `helper._partition_index_gray_python` | part of aggregate |
| `partition_indices_gray_batch` | `helper._partition_indices_gray_batch_python` | part of batch |

**What is NOT in Rust** (intentionally kept in NumPy for seed compatibility):
- `simhash_matrix_from_seed`, `ams_projection_matrix_from_seed` — depend on `np.random.default_rng`
- `count_sketch_vector_from_seed` — same reason
- `Muvera.__init__`, public API signatures — 100% unchanged

### Batch Processing

The library supports two input formats:
- **Single**: `(num_vectors, dimension)` - processes one point cloud
- **Variable-length batch**: `list[np.ndarray]` - each point cloud has different length (recommended for real-world data)

Variable-length batch processing flattens all point clouds, processes them together, then aggregates per-document using Rust `scatter_add_partitions` (or `np.add.at()` fallback).

## Code Conventions

### Python
- NumPy-style docstrings (configured in pyproject.toml)
- Type hints required (Python 3.9+ syntax with `|` for unions)
- Line length: 100 characters
- Use `np.float32` for all embeddings (memory efficiency)
- Use `np.uint32` for partition indices
- Random number generation via `np.random.default_rng(seed)` for reproducibility

### Rust
- Edition 2021
- Dependencies: `pyo3` 0.23, `numpy` 0.23 (Rust crate, not Python package), `ndarray` 0.16
- All three crates are version-locked together (upgrade all at once)
- Use `f32` for all floating-point data, `u32` for partition indices, `i32` for counts, `i64` for boundaries
- PyO3 functions accept `PyReadonlyArray*` for input arrays and `&Bound<PyArray*>` for in-place mutation

## Testing

### Test Organization

- **`test_helper.py`**: Low-level helper function tests (Gray code, projections, etc.)
- **`test_muvera.py`**: Core Muvera class tests (shapes, validation, reproducibility)
- **`test_reference.py`**: Validation against reference implementation (sionic-ai/muvera-py)
- **`test_real_colbert.py`**: Real-world ColBERT embedding tests using NanoBEIR fixtures
- **`test_rust_equivalence.py`**: Numerical equivalence tests between Rust kernels and Python fallbacks (skipped if Rust extension is unavailable)

### Real Data Testing

`test_real_colbert.py` uses cached ColBERT embeddings from NanoBEIR to validate performance on real data:
- **Fixtures**: 35 documents, 5 queries, 35 relevance judgments (~2.2MB cached in git)
- **Tests**: FDE encoding, correlation with native MaxSim, Recall@K metrics
- **Generation**: `python scripts/generate_test_fixtures.py` (requires pylate, datasets, torch)

Fixtures are cached in `tests/fixtures/colbert_nanobeir/` to avoid slow model inference during CI/testing.

## Key Parameters

- `num_repetitions`: Controls accuracy/dimension trade-off (default: 20)
- `num_simhash_projections`: Determines partition count as `2^n` (default: 5, range: [0, 31))
- `fill_empty_partitions`: Whether to fill empty partitions with nearest vector (default: True, documents only)
- `projection_type`: "identity" or "ams_sketch" for dimensionality reduction
- `final_projection_dimension`: Optional Count Sketch final projection

Output dimension: `num_repetitions * 2^num_simhash_projections * projection_dimension` (or `final_projection_dimension` if set)

## CI/CD and Deployment

### GitHub Actions Workflows

**`.github/workflows/test.yml`** - Continuous Integration
- Triggers: Push to main, all pull requests
- Tests across Python 3.9-3.13
- Installs Rust toolchain via `dtolnay/rust-toolchain@stable`
- Builds Rust extension via `pip install ".[dev]"` (maturin build backend)
- Runs ruff (lint + format check), mypy (type checking), pytest

**`.github/workflows/publish.yml`** - PyPI Publishing
- Triggers: Push to main
- Checks if `v{version}` tag already exists; skips release if it does
- Runs full test suite
- Builds cross-platform wheels via `PyO3/maturin-action@v1` (Linux x86_64/aarch64, macOS x86_64/aarch64, Windows x86_64)
- Builds sdist separately
- Creates git tag, publishes to PyPI via OIDC, creates GitHub Release

### Deployment Policy

**Auto-release on version bump:**
- Merging a PR that changes the version in `pyproject.toml` triggers a release
- If the version is unchanged, all release steps are skipped
- Uses OpenID Connect (OIDC) for secure, token-free authentication to PyPI

**Pre-deployment checklist:**
1. Update `version` in `pyproject.toml`
2. Run tests locally: `pytest`
3. Check code quality: `ruff check . && mypy muvera`
4. Create PR and merge to main

**OIDC Setup (one-time):**
1. Go to [PyPI](https://pypi.org) → Account settings → Publishing
2. Add a new pending publisher:
   - PyPI Project Name: `muvera-python`
   - Owner: `craftsangjae`
   - Repository: `muvera-python`
   - Workflow: `publish.yml`
   - Environment: (leave empty)

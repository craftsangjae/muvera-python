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
pip install -e .              # Install package in editable mode
pip install -e ".[dev]"       # Install with dev dependencies
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

**Tag-based release with OIDC trusted publishing:**

```bash
# 1. Update version in pyproject.toml
# version = "0.2.0"

# 2. Commit the version bump
git add pyproject.toml
git commit -m "Bump version to 0.2.0"

# 3. Create and push tag
git tag v0.2.0
git push origin v0.2.0

# GitHub Actions will automatically:
# - Run all tests
# - Verify tag matches pyproject.toml version
# - Build wheel and sdist
# - Publish to PyPI via OIDC (no API token needed)
```

**Local build (for testing):**
```bash
pip install build
python -m build
twine check dist/*
```

## Architecture

### Core Components

**`muvera/muvera.py`** - Main `Muvera` class implementing Fixed Dimensional Encoding (FDE)
- Three encoding paths: single document, uniform batch, variable-length batch
- Document encoding uses AVERAGE aggregation within partitions
- Query encoding uses SUM aggregation within partitions
- Optional final dimensionality reduction via Count Sketch

**`muvera/helper.py`** - Low-level utilities (not public API)
- Gray code manipulation for partition indexing
- Random projection matrices (SimHash, AMS Sketch, Count Sketch)
- Vectorized batch partition indexing

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

### Batch Processing

The library supports three input formats:
- **Single**: `(num_vectors, dimension)` - processes one point cloud
- **Uniform batch**: `(batch_size, num_vectors, dimension)` - all point clouds have same length
- **Variable-length batch**: `list[np.ndarray]` - each point cloud has different length (recommended for real-world data)

Variable-length batch processing flattens all point clouds, processes them together, then aggregates per-document using `np.add.at()` for efficient scatter-add operations.

## Code Conventions

- NumPy-style docstrings (configured in pyproject.toml)
- Type hints required (Python 3.9+ syntax with `|` for unions)
- Line length: 100 characters
- Use `np.float32` for all embeddings (memory efficiency)
- Use `np.uint32` for partition indices
- Random number generation via `np.random.default_rng(seed)` for reproducibility

## Testing

### Test Organization

- **`test_helper.py`**: Low-level helper function tests (Gray code, projections, etc.)
- **`test_muvera.py`**: Core Muvera class tests (shapes, validation, reproducibility)
- **`test_reference.py`**: Validation against reference implementation (sionic-ai/muvera-py)
- **`test_real_colbert.py`**: Real-world ColBERT embedding tests using NanoBEIR fixtures

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
- Runs ruff (lint + format check), mypy (type checking), pytest
- Tests example scripts

**`.github/workflows/publish.yml`** - PyPI Publishing
- Triggers: Version tags (e.g., `v0.2.0`)
- Verifies tag matches `pyproject.toml` version
- Runs full test suite
- Builds wheel and source distribution
- Publishes to PyPI using OIDC trusted publishing (no API token needed)

### Deployment Policy

**Tag-based releases with OIDC:**
- All releases are triggered by pushing version tags
- Uses OpenID Connect (OIDC) for secure, token-free authentication to PyPI
- Automatic version verification prevents mismatched releases

**Pre-deployment checklist:**
1. Update `version` in `pyproject.toml`
2. Run tests locally: `pytest`
3. Check code quality: `ruff check . && mypy muvera`
4. Commit: `git commit -m "Bump version to X.Y.Z"`
5. Tag: `git tag vX.Y.Z`
6. Push tag: `git push origin vX.Y.Z`

**OIDC Setup (one-time):**
1. Go to [PyPI](https://pypi.org) → Account settings → Publishing
2. Add a new pending publisher:
   - PyPI Project Name: `muvera-python`
   - Owner: `craftsangjae`
   - Repository: `muvera-python`
   - Workflow: `publish.yml`
   - Environment: (leave empty)

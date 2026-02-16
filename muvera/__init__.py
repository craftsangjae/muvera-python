"""MuVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings."""

from muvera.muvera import Muvera

__all__ = ["Muvera"]
__version__ = "0.1.0"

try:
    import muvera._rust_kernels  # noqa: F401

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

"""MuVERA speed benchmark for baseline measurement before Rust migration.

Measures initialization and encoding times using Pareto optimal configurations
from the MuVERA paper: (Rreps, ksim, dproj) in
{(20,3,8), (20,4,8), (20,5,8), (20,5,16)}.

Usage:
    python benchmarks/bench_speed.py          # Full benchmark
    python benchmarks/bench_speed.py --quick  # Quick smoke test
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

from muvera import Muvera

RESULTS_DIR = Path(__file__).parent / "results"

# Pareto optimal configurations from the MuVERA paper (Table 1).
# (num_repetitions, num_simhash_projections, projection_dimension)
PARETO_CONFIGS: list[tuple[int, int, int]] = [
    (20, 3, 8),
    (20, 4, 8),
    (20, 5, 8),
    (20, 5, 16),
]


class _Config(NamedTuple):
    rep: int
    simhash: int
    proj_dim: int


def _system_info() -> dict:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _bench(fn, *, warmup: int = 1, repeats: int = 5) -> dict:
    """Run fn with warmup, return timing stats in seconds."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return {"mean_s": mean, "std_s": std, "repeats": repeats}


def _fmt_time(mean_s: float, std_s: float) -> str:
    """Format time with appropriate unit."""
    if mean_s < 1e-3:
        return f"{mean_s * 1e6:.1f} \u00b1 {std_s * 1e6:.1f} \u00b5s"
    if mean_s < 1.0:
        return f"{mean_s * 1e3:.2f} \u00b1 {std_s * 1e3:.2f} ms"
    return f"{mean_s:.3f} \u00b1 {std_s:.3f} s"


def _fmt_single(seconds: float) -> str:
    """Format a single time value with unit (no std)."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} \u00b5s"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _make_encoder(cfg: _Config, dim: int) -> Muvera:
    return Muvera(
        num_repetitions=cfg.rep,
        num_simhash_projections=cfg.simhash,
        dimension=dim,
        projection_type="ams_sketch",
        projection_dimension=cfg.proj_dim,
    )


def _cfg_label(cfg: _Config) -> str:
    return f"R={cfg.rep}, k={cfg.simhash}, d_proj={cfg.proj_dim}"


def _bench_init(
    configs: list[_Config],
    dimensions: list[int],
    *,
    repeats: int,
) -> list[dict]:
    print("\n--- Initialization ---")
    results = []
    for cfg in configs:
        for dim in dimensions:

            def fn(c=cfg, d=dim):
                _make_encoder(c, d)

            stats = _bench(fn, repeats=repeats)
            label = f"{_cfg_label(cfg)}, dim={dim}"
            print(f"  {label}: {_fmt_time(stats['mean_s'], stats['std_s'])}")
            results.append({"params": {**cfg._asdict(), "dim": dim}, **stats})
    return results


def _bench_single_doc(
    configs: list[_Config],
    dimensions: list[int],
    num_vectors_list: list[int],
    *,
    repeats: int,
) -> list[dict]:
    print("\n--- Single Document Encoding ---")
    results = []
    for cfg in configs:
        for dim in dimensions:
            encoder = _make_encoder(cfg, dim)
            for nv in num_vectors_list:
                doc = np.random.default_rng(0).standard_normal((nv, dim)).astype(np.float32)

                def fn(e=encoder, d=doc):
                    e.encode_documents(d)

                stats = _bench(fn, repeats=repeats)
                label = f"{_cfg_label(cfg)}, dim={dim}, vecs={nv}"
                print(f"  {label}: {_fmt_time(stats['mean_s'], stats['std_s'])}")
                results.append(
                    {
                        "params": {**cfg._asdict(), "dim": dim, "num_vectors": nv},
                        **stats,
                    }
                )
    return results


def _bench_batch_doc(
    configs: list[_Config],
    dimensions: list[int],
    batch_sizes: list[int],
    num_vectors_list: list[int],
    *,
    repeats: int,
) -> list[dict]:
    print("\n--- Batch Document Encoding ---")
    results = []
    for cfg in configs:
        for dim in dimensions:
            encoder = _make_encoder(cfg, dim)
            for bs in batch_sizes:
                for nv in num_vectors_list:
                    rng = np.random.default_rng(0)
                    docs = [
                        rng.standard_normal(
                            (rng.integers(int(nv * 0.8), int(nv * 1.2) + 1), dim)
                        ).astype(np.float32)
                        for _ in range(bs)
                    ]

                    def fn(e=encoder, d=docs):
                        e.encode_documents(d)

                    stats = _bench(fn, repeats=repeats)
                    per_doc = stats["mean_s"] / bs
                    label = f"{_cfg_label(cfg)}, dim={dim}, batch={bs}, vecs={nv}"
                    time_str = _fmt_time(stats["mean_s"], stats["std_s"])
                    print(f"  {label}: {time_str}  ({_fmt_single(per_doc)}/doc)")
                    results.append(
                        {
                            "params": {
                                **cfg._asdict(),
                                "dim": dim,
                                "batch_size": bs,
                                "num_vectors": nv,
                            },
                            "per_doc_s": per_doc,
                            **stats,
                        }
                    )
    return results


def _bench_batch_query(
    configs: list[_Config],
    dimensions: list[int],
    batch_sizes: list[int],
    num_vectors_list: list[int],
    *,
    repeats: int,
) -> list[dict]:
    print("\n--- Batch Query Encoding ---")
    results = []
    for cfg in configs:
        for dim in dimensions:
            encoder = _make_encoder(cfg, dim)
            for bs in batch_sizes:
                for nv in num_vectors_list:
                    rng = np.random.default_rng(0)
                    queries = [
                        rng.standard_normal(
                            (rng.integers(int(nv * 0.8), int(nv * 1.2) + 1), dim)
                        ).astype(np.float32)
                        for _ in range(bs)
                    ]

                    def fn(e=encoder, q=queries):
                        e.encode_queries(q)

                    stats = _bench(fn, repeats=repeats)
                    per_query = stats["mean_s"] / bs
                    label = f"{_cfg_label(cfg)}, dim={dim}, batch={bs}, vecs={nv}"
                    time_str = _fmt_time(stats["mean_s"], stats["std_s"])
                    print(f"  {label}: {time_str}  ({_fmt_single(per_query)}/query)")
                    results.append(
                        {
                            "params": {
                                **cfg._asdict(),
                                "dim": dim,
                                "batch_size": bs,
                                "num_vectors": nv,
                            },
                            "per_query_s": per_query,
                            **stats,
                        }
                    )
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description="MuVERA speed benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer parameters and fewer repeats",
    )
    args = parser.parse_args()

    if args.quick:
        configs = [_Config(20, 5, 8)]
        dimensions = [128]
        batch_sizes = [100]
        num_vectors_list = [128]
        repeats = 3
    else:
        configs = [_Config(*t) for t in PARETO_CONFIGS]
        dimensions = [128, 256, 512]
        batch_sizes = [100]
        num_vectors_list = [64, 128]
        repeats = 3

    info = _system_info()
    print("=== MuVERA Speed Benchmark ===")
    print(f"Python version: {info['python_version']} | NumPy version: {info['numpy_version']}")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"Pareto configs: {[(c.rep, c.simhash, c.proj_dim) for c in configs]}")

    all_results: dict = {"system_info": info, "mode": "quick" if args.quick else "full"}

    all_results["init"] = _bench_init(configs, dimensions, repeats=repeats)
    all_results["single_doc"] = _bench_single_doc(
        configs, dimensions, num_vectors_list, repeats=repeats
    )
    all_results["batch_doc"] = _bench_batch_doc(
        configs, dimensions, batch_sizes, num_vectors_list, repeats=repeats
    )
    all_results["batch_query"] = _bench_batch_query(
        configs, dimensions, batch_sizes, num_vectors_list, repeats=repeats
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mode = "quick" if args.quick else "full"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"bench_{mode}_{ts}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    _main()

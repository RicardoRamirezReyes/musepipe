"""Thread-chunked execution for embarrassingly parallel per-channel loops.

The hot loops of the extraction chain (psffit, optimal, local-surface) iterate
independently over spectral channels and write to disjoint per-channel slots of
preallocated arrays. Distributing contiguous channel ranges over threads keeps
the per-channel arithmetic byte-identical to the serial loop (same operations,
same inputs, disjoint outputs); numpy/BLAS release the GIL during the heavy
calls, so threads scale without pickling cubes across processes.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor


def resolve_n_jobs(n_jobs=None, *, max_default=8):
    """Resolve a worker count: explicit value > MUSEPIPE_N_JOBS env > capped cpu_count."""
    if n_jobs is None:
        env = os.environ.get("MUSEPIPE_N_JOBS")
        if env:
            n_jobs = int(env)
        else:
            n_jobs = min(int(max_default), os.cpu_count() or 1)
    return max(1, int(n_jobs))


def channel_chunks(nz, n_chunks):
    """Split range(nz) into up to n_chunks contiguous (z0, z1) ranges."""
    nz = int(nz)
    n_chunks = max(1, min(int(n_chunks), nz))
    step, extra = divmod(nz, n_chunks)
    bounds = []
    z0 = 0
    for i in range(n_chunks):
        z1 = z0 + step + (1 if i < extra else 0)
        bounds.append((z0, z1))
        z0 = z1
    return bounds


def run_channel_chunks(worker, nz, *, n_jobs=1, chunks_per_job=4, min_chunk=8):
    """Run ``worker(z0, z1)`` over disjoint channel ranges, optionally threaded.

    The worker MUST only write to per-channel slots (index z) of preallocated
    outputs; shared accumulators must be handled per-chunk by the caller. With
    ``n_jobs=1`` this is exactly the serial loop (single ``worker(0, nz)``).
    """
    nz = int(nz)
    n_jobs = max(1, int(n_jobs))
    if n_jobs == 1 or nz <= int(min_chunk):
        worker(0, nz)
        return
    bounds = channel_chunks(nz, n_jobs * int(chunks_per_job))
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [pool.submit(worker, z0, z1) for z0, z1 in bounds]
        for future in futures:
            future.result()


__all__ = ["channel_chunks", "resolve_n_jobs", "run_channel_chunks"]

"""gemm_w4a4 — main SVDQuant W4A4 linear, Blackwell NVFP4 path.

See `docs/kernels/gemm_w4a4.md` for the full design (β interleave,
shared-tmem accumulator, warp-spec split).
"""
from .kernel import launch
from .module import SVDQuantW4A4Linear

__all__ = ["launch", "SVDQuantW4A4Linear"]

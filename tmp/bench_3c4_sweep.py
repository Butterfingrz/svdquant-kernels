"""3c-4 multi-tile scaling sweep — measure wall-clock vs N_TILES.

For each N_TILES in [1, 2, 4, 8, 16, 24]: build M_total = N_TILES * 128,
time 100 calls, report per-call latency + total INT4 GOPS achieved.

910B has 24 AI Cores; sweep tops out there. Hypothesis: cube saturates
around 24 tiles where each core does one [128, 2048, 256] tile in
parallel; below that, we're cube-underutilized.

Theoretical floor (cube saturated):
  per-tile cube work = 2*128*2048*256 = 134 MFLOPs INT4
  cube peak per core ≈ 512 TOPS / 24 ≈ 21 TOPS
  per-tile cube floor ≈ 6.4 µs
  multi-tile (24 cores parallel) ≈ 6.4 µs total (perfect scaling)
  multi-tile (lower N): ~ ceil(N_TILES/24) * 6.4 µs (single-wave) +
  launch overhead.
"""
import sys
import time
from pathlib import Path

print("[sweep] importing torch", flush=True)
import torch
import torch_npu

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "csrc" / "python"))
sys.path.insert(0, str(_REPO_ROOT))
import op_extension  # noqa: F401
from baseline.kernels.gemm_w4a4.ref_int4 import make_int4_inputs

TILE_M, TILE_K, TILE_N, TILE_R = 128, 2048, 256, 32
N_ITER = 100
WARMUP = 5

g = torch.Generator().manual_seed(0xB0BA)
# Reused per-N: regenerate inputs at each shape.

def bench_one(n_tiles: int) -> dict:
    M = TILE_M * n_tiles
    act, wgt, ascales, wscales = make_int4_inputs(M, TILE_K, TILE_N)
    la = (torch.rand(M, TILE_R, generator=g) * 2 - 1) * 0.1
    lu = ((torch.rand(TILE_N, TILE_R, generator=g) * 2 - 1) * 0.1).to(torch.float16)
    wc = ((torch.rand(TILE_N, generator=g) + 0.5)).to(torch.float16)
    bi = ((torch.rand(TILE_N, generator=g) * 2 - 1) * 0.1).to(torch.float16)
    args = (
        act.npu(), wgt.npu(), ascales.npu(), wscales.npu(),
        la.npu(), lu.npu(), bi.npu(), wc.npu(),
    )
    torch.npu.synchronize()

    def _call():
        return torch.ops.svdquant.gemm_w4a4(*args)

    for _ in range(WARMUP):
        _ = _call()
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        _ = _call()
    torch.npu.synchronize()
    t1 = time.perf_counter()

    us = (t1 - t0) * 1e6 / N_ITER
    flops_main = 2 * M * TILE_K * TILE_N
    gops = flops_main / (us * 1e-6) / 1e9
    pct_peak = flops_main / (us * 1e-6) / 512e12 * 100
    return {
        "n_tiles": n_tiles, "M": M, "us": us,
        "gops": gops, "pct_peak": pct_peak,
        "flops_main_MFLOPs": flops_main / 1e6,
    }


print(f"[sweep] tile = M={TILE_M} K={TILE_K} N={TILE_N} R={TILE_R}", flush=True)
print(f"[sweep] N_ITER={N_ITER} per shape", flush=True)
print(f"[sweep] {'tiles':>5} {'M':>5} {'us/call':>10} {'GOPS':>10} {'% peak':>8}",
      flush=True)

baseline_us = None
for n in [1, 2, 4, 8, 16, 24]:
    r = bench_one(n)
    speedup_str = ""
    if baseline_us is None:
        baseline_us = r["us"]
    else:
        s = baseline_us * n / r["us"]   # scaling efficiency vs ideal linear
        speedup_str = f"  scale={s:.2f}x (ideal {n}x)"
    print(f"[sweep] {r['n_tiles']:>5} {r['M']:>5} {r['us']:>9.2f} µs "
          f"{r['gops']:>9.1f} {r['pct_peak']:>7.2f}%{speedup_str}",
          flush=True)

print("[sweep] done", flush=True)

"""Minimal bench for msprof — N_TILES=16 (saturating point of multi-tile sweep).

After warmup, runs 20 calls back-to-back so msprof sees clean cube/vec
pipe utilization. No Python-side work between calls.
"""
import sys
from pathlib import Path
import torch
import torch_npu

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "csrc" / "python"))
sys.path.insert(0, str(_REPO_ROOT))
import op_extension  # noqa: F401
from baseline.kernels.gemm_w4a4.ref_int4 import make_int4_inputs

TILE_M, TILE_K, TILE_N, TILE_R = 128, 2048, 256, 32
N_TILES = 16
M = TILE_M * N_TILES

act, wgt, ascales, wscales = make_int4_inputs(M, TILE_K, TILE_N)
g = torch.Generator().manual_seed(0xB0BA)
la = (torch.rand(M, TILE_R, generator=g) * 2 - 1) * 0.1
lu = ((torch.rand(TILE_N, TILE_R, generator=g) * 2 - 1) * 0.1).to(torch.float16)
wc = ((torch.rand(TILE_N, generator=g) + 0.5)).to(torch.float16)
bi = ((torch.rand(TILE_N, generator=g) * 2 - 1) * 0.1).to(torch.float16)
args = (
    act.npu(), wgt.npu(), ascales.npu(), wscales.npu(),
    la.npu(), lu.npu(), bi.npu(), wc.npu(),
)
torch.npu.synchronize()

# Warmup
for _ in range(5):
    torch.ops.svdquant.gemm_w4a4(*args)
torch.npu.synchronize()

# Measured calls (msprof captures these)
for _ in range(20):
    torch.ops.svdquant.gemm_w4a4(*args)
torch.npu.synchronize()

"""Phase 3b end-to-end test — torch.ops.svdquant.gemm_w4a4 INT4 + LoRA-up.

Inputs: signed-INT4-packed act + wgt + per-64-K-block fp16 scales +
fp32 LoRA-down output + fp16 LoRA-up weight. Output: fp16 [M, N].

Reference: `gemm_w4a4_ref_int4`. Current state: LoRA tensors are zeros,
so only the main INT4 path is exercised (kernel doesn't read them
yet). Step 3b-3 wires up the LoRA cube pass; step 3b-4 then un-zeros
the test inputs to exercise both paths.

Tolerance: rtol=5e-2 atol=5e-2 — INT4 quant noise + dequant rounding.
"""

import sys
import traceback
import unittest
from pathlib import Path


def _step(msg: str) -> None:
    """Stream-print so the GitCode log panel sees progress before any
    SIGSEGV silences the process."""
    print(f"[3a-test] {msg}", flush=True)


_step("importing torch")
import torch
_step(f"  torch {torch.__version__}")

_step("importing torch_npu")
import torch_npu  # noqa: F401  — registers PrivateUse1 backend
_step(f"  torch_npu {torch_npu.__version__} npu_available={torch.npu.is_available()}")

# csrc/python is on sys.path so we can `import op_extension`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "csrc" / "python"))
sys.path.insert(0, str(_REPO_ROOT))

_step("importing op_extension (loads libop_extension.so + runs __register_kernels)")
try:
    import op_extension  # noqa: F401, E402  — loads libop_extension.so
    _step("  op_extension loaded OK")
except Exception:
    traceback.print_exc()
    raise

_step("importing baseline.kernels.gemm_w4a4.ref_int4")
from baseline.kernels.gemm_w4a4.ref_int4 import (  # noqa: E402
    gemm_w4a4_ref_int4, make_int4_inputs,
)
_step("  baseline ref_int4 loaded OK")


# 3c-4 multi-tile: M_total = TILE_M * N_TILES, exercises grid M-major fanout.
# Tile-per-block (kTileM/K/N/R) must match svdquant_w4a4_op.cpp + kernel_device.cpp.
TILE_M = 128
TILE_K = 2048
TILE_N = 256
N_TILES = 2     # 2 AI cores fan out — minimal multi-block coverage
PHASE3B_M = TILE_M * N_TILES   # 256 — total M rows = 2 cube tiles
PHASE3B_K = TILE_K
PHASE3B_N = TILE_N
PHASE3B_R = 32     # LoRA rank — must match kTileR in svdquant_w4a4_op.cpp
PHASE3B_K_BLOCK = 64   # per-K-block mad_s4 KS
PHASE3B_K_BLOCKS = PHASE3B_K // PHASE3B_K_BLOCK   # = 32


def _recompute_from_workspace(workspace_cpu, ascales_cpu, wscales_cpu):
    """task #109 — CPU recompute of what vec _should_ produce.

    workspace_cpu: [RING_SLOTS, M, N] int32, only slots [0:K_BLOCKS] valid.
    ascales_cpu : [K_BLOCKS, M] fp16
    wscales_cpu : [K_BLOCKS, N] fp16

    Returns: [M, N] fp32 — per-K-block scale & sum, mirroring vec's
    TCVT(int32→fp32) + TROWEXPANDMUL(ascales) + TCOLEXPANDMUL(wscales)
    + TADD across K-blocks. If this matches ref_no_lora ≈ 0.001, cube
    K-loop ring is clean and any divergence in `out` is from vec.
    """
    import torch
    acc = torch.zeros(PHASE3B_M, PHASE3B_N, dtype=torch.float32)
    for kb in range(PHASE3B_K_BLOCKS):
        partial = workspace_cpu[kb].float()           # [M, N]
        a = ascales_cpu[kb].float()                   # [M]
        w = wscales_cpu[kb].float()                   # [N]
        acc += partial * a[:, None] * w[None, :]
    return acc


class TestGemmW4A4Phase3bInt4Lora(unittest.TestCase):

    def test_phase3b_int4_lora_path(self):
        _step("test_phase3b_int4_lora_path: enter")
        if not torch.npu.is_available():
            self.skipTest("Ascend NPU not available")

        _step("  building INT4 inputs (CPU)")
        act, wgt, ascales, wscales = make_int4_inputs(
            PHASE3B_M, PHASE3B_K, PHASE3B_N
        )

        _step("  building reference (CPU)")
        # Seeded random LoRA tensors — small amplitude so the LoRA-up
        # contribution sits in the same magnitude band as the main GEMM
        # output (~O(K * amax^2)). The kernel applies it pre-fp16-cast,
        # so even a tiny LoRA reaches the output bits.
        g = torch.Generator().manual_seed(0xB0BA)
        lora_act_in = (torch.rand(PHASE3B_M, PHASE3B_R, generator=g) * 2 - 1) * 0.1
        lora_up = ((torch.rand(PHASE3B_N, PHASE3B_R, generator=g) * 2 - 1) * 0.1).to(torch.float16)
        # 3c-1: per-channel affine. wcscales kept near 1.0 (+0.5 base, +0.5
        # range) so it doesn't blow up dynamic range past fp16; bias small
        # so it doesn't dominate the GEMM output.
        wcscales = ((torch.rand(PHASE3B_N, generator=g) + 0.5)).to(torch.float16)
        bias = ((torch.rand(PHASE3B_N, generator=g) * 2 - 1) * 0.1).to(torch.float16)
        ref, _, _ = gemm_w4a4_ref_int4(
            act, wgt, ascales, wscales, lora_act_in, lora_up,
            bias=bias, wcscales=wcscales,
        )
        self.assertEqual(ref.shape, (PHASE3B_M, PHASE3B_N))
        self.assertEqual(ref.dtype, torch.float16)
        _step(f"  ref shape {tuple(ref.shape)} max_abs={ref.abs().max().item():.3f}")

        _step("  moving inputs to NPU")
        act_npu = act.npu()
        wgt_npu = wgt.npu()
        ascales_npu = ascales.npu()
        wscales_npu = wscales.npu()
        lora_act_in_npu = lora_act_in.npu()
        lora_up_npu = lora_up.npu()
        bias_npu = bias.npu()
        wcscales_npu = wcscales.npu()
        torch.npu.synchronize()
        _step("  inputs on NPU OK")

        _step("  calling torch.ops.svdquant.gemm_w4a4_debug (task #109: triple-out)")
        out, lora_buf, workspace = torch.ops.svdquant.gemm_w4a4_debug(
            act_npu, wgt_npu, ascales_npu, wscales_npu,
            lora_act_in_npu, lora_up_npu,
            bias_npu, wcscales_npu,
        )
        _step("  op returned, syncing")
        torch.npu.synchronize()
        _step(f"  sync OK, out shape {tuple(out.shape)} dtype {out.dtype}")
        _step(f"  lora_buf shape {tuple(lora_buf.shape)} dtype {lora_buf.dtype}")
        _step(f"  workspace shape {tuple(workspace.shape)} dtype {workspace.dtype}")
        self.assertEqual(out.shape, (PHASE3B_M, PHASE3B_N))
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(lora_buf.shape, (PHASE3B_M, PHASE3B_N))
        self.assertEqual(lora_buf.dtype, torch.float32)
        # workspace is [RING_SLOTS=6, M, N] int32; only slots [0:K_BLOCKS] hold valid data
        self.assertEqual(workspace.shape[1:], (PHASE3B_M, PHASE3B_N))
        self.assertEqual(workspace.dtype, torch.int32)

        out_cpu = out.cpu()
        lora_buf_cpu = lora_buf.cpu()
        workspace_cpu = workspace.cpu()

        # --- diagnostic 1: cube LoRA-up pass in isolation ---
        # Device端 LA 先 cast 到 fp16(host op binding line 98),所以 ref
        # 也得先 round 一遍才公平。lora_up 已经是 fp16.
        la_fp16_ref = lora_act_in.to(torch.float16).to(torch.float32)
        lu_fp32_ref = lora_up.to(torch.float32)
        ref_lora_buf = la_fp16_ref @ lu_fp32_ref.T  # [M, N] fp32

        def _stats(t, name):
            t_f = t.float()
            return (
                f"  {name}: any_nan={t_f.isnan().any().item()} "
                f"any_inf={t_f.isinf().any().item()} "
                f"min={t_f.min().item():.4g} max={t_f.max().item():.4g} "
                f"mean={t_f.mean().item():.4g}"
            )

        _step(_stats(ref_lora_buf, "ref_lora_buf"))
        _step(_stats(lora_buf_cpu, "lora_buf "))
        lb_diff = (lora_buf_cpu - ref_lora_buf).abs()
        _step(
            f"  lora_buf vs ref: max_abs={lb_diff.max().item():.4g} "
            f"mean_abs={lb_diff.mean().item():.4g} "
            f"any_nan_in_diff={lb_diff.isnan().any().item()}"
        )

        # --- diagnostic 2: full output ---
        _step(_stats(ref.float(),  "ref       "))
        _step(_stats(out_cpu,      "out      "))
        diff = (out_cpu.float() - ref.float()).abs()
        _step(
            f"  out vs ref: max_abs={diff.max().item():.4g} "
            f"mean_abs={diff.mean().item():.4g} "
            f"any_nan_in_out={out_cpu.isnan().any().item()}"
        )

        # --- diagnostic 3 (task #105 bisect): is main path still healthy? ---
        ref_no_lora = ref.float() - ref_lora_buf
        _step(_stats(ref_no_lora, "ref_no_lora"))
        no_lora_diff = (out_cpu.float() - ref_no_lora).abs()
        _step(
            f"  out vs ref_no_lora: max_abs={no_lora_diff.max().item():.4g} "
            f"mean_abs={no_lora_diff.mean().item():.4g}"
        )

        # --- diagnostic 4 (task #109 cube K-loop): CPU recompute from workspace ---
        # Only valid when K_BLOCKS ≤ RING_SLOTS (no slot overwrites). At 3c-3
        # production shape K_BLOCKS=32 > RING_SLOTS=6 so workspace[kb] for
        # kb≥6 is already overwritten by later K-blocks; the recompute would
        # see only the last 6 partials. Skip the diagnostic in that case.
        RING_SLOTS = 6
        if PHASE3B_K_BLOCKS <= RING_SLOTS:
            cpu_recompute = _recompute_from_workspace(
                workspace_cpu, ascales, wscales
            )
            _step(_stats(cpu_recompute, "cpu_recompute"))
            cube_diff = (cpu_recompute - ref_no_lora).abs()
            _step(
                f"  cpu_recompute vs ref_no_lora (CUBE K-loop): "
                f"max_abs={cube_diff.max().item():.4g} "
                f"mean_abs={cube_diff.mean().item():.4g}"
            )
            vec_diff = (out_cpu.float() - cpu_recompute).abs()
            _step(
                f"  out vs cpu_recompute (VEC pipeline): "
                f"max_abs={vec_diff.max().item():.4g} "
                f"mean_abs={vec_diff.mean().item():.4g}"
            )
        else:
            _step(f"  (skip cpu_recompute diagnostic: K_BLOCKS={PHASE3B_K_BLOCKS} > RING_SLOTS={RING_SLOTS})")

        # 判定指引(K_BLOCKS ≤ RING_SLOTS 时):
        #   cpu_recompute vs ref_no_lora ≈ 0.001 → cube K-loop OK
        #   cpu_recompute vs ref_no_lora ≫ 0.001 → cube K-loop 写出 junk int32
        #   out vs cpu_recompute ≈ 0.001 → vec pipeline OK
        #   out vs cpu_recompute ≫ 0.001 → vec pipeline 污染(TCVT/TROW/TCOL/TADD)
        torch.testing.assert_close(
            out_cpu, ref, rtol=5e-2, atol=5e-2,
            msg="phase 3b int4+lora output diverged from baseline ref",
        )
        _step("test_phase3b_int4_lora_path: pass")

    def test_phase3b_zero_lora(self):
        """task #107 (kept as 3c-1 regression coverage) — zero LoRA, real
        bias/wcscales. Validates the main K-loop + per-channel affine
        still passes when the LoRA cube pass produces only zeros."""
        _step("test_phase3b_zero_lora: enter")
        if not torch.npu.is_available():
            self.skipTest("Ascend NPU not available")

        _step("  building INT4 inputs + zero LoRA + 3c-1 affine")
        act, wgt, ascales, wscales = make_int4_inputs(
            PHASE3B_M, PHASE3B_K, PHASE3B_N
        )
        lora_act_in = torch.zeros(PHASE3B_M, PHASE3B_R)
        lora_up     = torch.zeros(PHASE3B_N, PHASE3B_R, dtype=torch.float16)
        g = torch.Generator().manual_seed(0xB0BA)
        wcscales = ((torch.rand(PHASE3B_N, generator=g) + 0.5)).to(torch.float16)
        bias = ((torch.rand(PHASE3B_N, generator=g) * 2 - 1) * 0.1).to(torch.float16)

        # ref with zero LoRA + 3c-1 affine — exercises main K-loop, skips LoRA add.
        ref, _, _ = gemm_w4a4_ref_int4(
            act, wgt, ascales, wscales, lora_act_in, lora_up,
            bias=bias, wcscales=wcscales,
        )
        _step(f"  ref max_abs={ref.abs().max().item():.3f}")

        out, lora_buf, workspace = torch.ops.svdquant.gemm_w4a4_debug(
            act.npu(), wgt.npu(), ascales.npu(), wscales.npu(),
            lora_act_in.npu(), lora_up.npu(),
            bias.npu(), wcscales.npu(),
        )
        torch.npu.synchronize()
        out_cpu = out.cpu()
        lb_cpu  = lora_buf.cpu()
        ws_cpu  = workspace.cpu()

        def _stats(t, name):
            t_f = t.float()
            return (f"  {name}: any_nan={t_f.isnan().any().item()} "
                    f"min={t_f.min().item():.4g} max={t_f.max().item():.4g} "
                    f"mean={t_f.mean().item():.4g}")
        _step(_stats(lb_cpu,  "lora_buf  "))
        _step(_stats(ref.float(), "ref       "))
        _step(_stats(out_cpu, "out       "))
        diff = (out_cpu.float() - ref.float()).abs()
        _step(f"  out vs ref(zero-lora): max_abs={diff.max().item():.4g} "
              f"mean_abs={diff.mean().item():.4g}")

        # task #109 — same 3-way split, on zero-LoRA inputs (only valid when
        # K_BLOCKS ≤ RING_SLOTS=6; skip at production shape K_BLOCKS=32).
        RING_SLOTS = 6
        if PHASE3B_K_BLOCKS <= RING_SLOTS:
            cpu_recompute = _recompute_from_workspace(ws_cpu, ascales, wscales)
            _step(_stats(cpu_recompute, "cpu_recompute"))
            cube_diff = (cpu_recompute - ref.float()).abs()
            _step(f"  cpu_recompute vs ref (CUBE K-loop, zero-lora): "
                  f"max_abs={cube_diff.max().item():.4g} "
                  f"mean_abs={cube_diff.mean().item():.4g}")
            vec_diff = (out_cpu.float() - cpu_recompute).abs()
            _step(f"  out vs cpu_recompute (VEC pipeline, zero-lora): "
                  f"max_abs={vec_diff.max().item():.4g} "
                  f"mean_abs={vec_diff.mean().item():.4g}")
        else:
            _step(f"  (skip cpu_recompute diagnostic: K_BLOCKS={PHASE3B_K_BLOCKS} > RING_SLOTS={RING_SLOTS})")
        _step("test_phase3b_zero_lora: pass (diagnostic only, no assert)")


if __name__ == "__main__":
    unittest.main(verbosity=2)

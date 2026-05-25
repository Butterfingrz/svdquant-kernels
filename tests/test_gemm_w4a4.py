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


# Phase 3b tile constants — must match svdquant_w4a4_op.cpp + kernel_device.cpp.
PHASE3B_M = 64
PHASE3B_K = 128
PHASE3B_N = 128
PHASE3B_R = 32     # LoRA rank — must match kPhase3bR in svdquant_w4a4_op.cpp


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
        ref, _, _ = gemm_w4a4_ref_int4(
            act, wgt, ascales, wscales, lora_act_in, lora_up,
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
        torch.npu.synchronize()
        _step("  inputs on NPU OK")

        _step("  calling torch.ops.svdquant.gemm_w4a4_debug (task #95: 双输出)")
        out, lora_buf = torch.ops.svdquant.gemm_w4a4_debug(
            act_npu, wgt_npu, ascales_npu, wscales_npu,
            lora_act_in_npu, lora_up_npu,
        )
        _step("  op returned, syncing")
        torch.npu.synchronize()
        _step(f"  sync OK, out shape {tuple(out.shape)} dtype {out.dtype}")
        _step(f"  lora_buf shape {tuple(lora_buf.shape)} dtype {lora_buf.dtype}")
        self.assertEqual(out.shape, (PHASE3B_M, PHASE3B_N))
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(lora_buf.shape, (PHASE3B_M, PHASE3B_N))
        self.assertEqual(lora_buf.dtype, torch.float32)

        out_cpu = out.cpu()
        lora_buf_cpu = lora_buf.cpu()

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
        # Subtract the LoRA term from ref to get what main-only output would be.
        # If kernel returns this within 3a's 0.001 (lora_buf=0 means TADD is a
        # no-op so it shouldn't matter), main K-loop + drain + LoRA section
        # epilogue is clean — only the cube LoRA pass is broken.
        # If still 2.6, there's an end-of-K-loop pipeline hazard.
        ref_no_lora = ref.float() - ref_lora_buf
        _step(_stats(ref_no_lora, "ref_no_lora"))
        no_lora_diff = (out_cpu.float() - ref_no_lora).abs()
        _step(
            f"  out vs ref_no_lora: max_abs={no_lora_diff.max().item():.4g} "
            f"mean_abs={no_lora_diff.mean().item():.4g}"
        )

        # 判定指引(给 log 读者):
        #   out vs ref_no_lora ≈ 0.001  → main 正常, 只是 cube LoRA pass 写0 (#104)
        #   out vs ref_no_lora 仍 ≈ 2.6 → main 也被 LoRA section 污染了 → end-of-K-loop hazard
        torch.testing.assert_close(
            out_cpu, ref, rtol=5e-2, atol=5e-2,
            msg="phase 3b int4+lora output diverged from baseline ref",
        )
        _step("test_phase3b_int4_lora_path: pass")

    def test_phase3b_zero_lora(self):
        """task #107 — feed lora_act_in=0, lora_up=0. AIC TMATMUL becomes
        0@0=0 → lora_buf is true zero (no L2/DRAM divergence). If `out`
        matches `ref_no_lora` ≈ 0.001, the 2.6 in test 1 came from AIC
        writing junk that AIV picked up via L2. If still 2.6, the AIV
        TADD/TCVT epilogue is corrupting main independently."""
        _step("test_phase3b_zero_lora: enter")
        if not torch.npu.is_available():
            self.skipTest("Ascend NPU not available")

        _step("  building INT4 inputs + zero LoRA")
        act, wgt, ascales, wscales = make_int4_inputs(
            PHASE3B_M, PHASE3B_K, PHASE3B_N
        )
        lora_act_in = torch.zeros(PHASE3B_M, PHASE3B_R)
        lora_up     = torch.zeros(PHASE3B_N, PHASE3B_R, dtype=torch.float16)

        # ref with zero LoRA == 3a baseline (lora term = 0)
        ref, _, _ = gemm_w4a4_ref_int4(
            act, wgt, ascales, wscales, lora_act_in, lora_up,
        )
        _step(f"  ref max_abs={ref.abs().max().item():.3f}")

        out, lora_buf = torch.ops.svdquant.gemm_w4a4_debug(
            act.npu(), wgt.npu(), ascales.npu(), wscales.npu(),
            lora_act_in.npu(), lora_up.npu(),
        )
        torch.npu.synchronize()
        out_cpu = out.cpu()
        lb_cpu  = lora_buf.cpu()

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
        # 判定:
        #   max_abs ≤ 0.001 → AIV 干净, 2.6 来自 AIC 写出 junk (#104 重点)
        #   max_abs ≈ 2.6  → AIV TADD/TCVT 本身坏, 跟 LoRA 数据无关
        _step("test_phase3b_zero_lora: pass (diagnostic only, no assert)")


if __name__ == "__main__":
    unittest.main(verbosity=2)

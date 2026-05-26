"""Host-only regression for FIX-pipe VDEQF16 wscale packing.

Three invariants, all pure-PyTorch (no NPU required):

  1. `pack_wscales_vdeqf16` round-trips through dav_c220's 19-bit-mini-float
     truncation bit-exactly. fp16 → fp32 reinterpret → bits[32:13] decode
     must reproduce the original fp16 value.

  2. The host VDEQF16 reference matches the obvious "fp32 multiply then
     fp16 cast" at single-K-block granularity (sanity check that the
     bit layout interpretation matches the math).

  3. End-to-end at production SVDQuant shape (M=128 K=2048 N=256, 32
     K-blocks): the per-K-block fp16 cast introduced by FIX-pipe drain
     stays within fp16's representational precision at the kernel-level
     output (≤ ~1 ULP at max-magnitude output ≈ 8e-3). 0 K-blocks may
     saturate fp16.

Hardware deqscalar bit layout decoded from
`pto-isa/tests/.../tstore_acc2gm/gen_data.py:44-55` —
`extract_quant_params(uint64) → (m1, offset, sign)` where m1 is a
19-bit mini-float at bits[32:13].
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from baseline.kernels._int4 import (  # noqa: E402
    INT4_BLOCK_SIZE,
    _unpack_signed_nibbles,
    pack_wscales_vdeqf16,
)
from baseline.kernels.gemm_w4a4.ref_int4 import make_int4_inputs  # noqa: E402


def _decode_deqscalar_m1(quant_gm: int) -> float:
    """Hardware-faithful decode of `pack_wscales_vdeqf16` output.

    Mirrors `extract_quant_params` from PTO ISA's gen_data.py. Reads
    bits[32:13] of the uint64 as a 19-bit mini-float (sign + 8-bit exp
    + 10-bit mantissa) and returns the resolved scalar — what the
    FIX-pipe will multiply against during a TStoreAccFp drain.
    """
    q = int(quant_gm)
    m1_bits = (q >> 13) & 0xFFFFF
    sign_bit = (m1_bits >> 18) & 0x1
    exponent = (m1_bits >> 10) & 0xFF
    mantissa = m1_bits & 0x3FF
    if exponent == 0 and mantissa == 0:
        return 0.0
    return ((-1.0) ** sign_bit) * (1.0 + mantissa / 1024.0) * (2.0 ** (exponent - 127))


def _vdeqf16_ref(partial_i32: torch.Tensor, wscale_fp16: torch.Tensor) -> torch.Tensor:
    """Host reference for what the cube FIX-pipe writes to GM with VDEQF16:
    `(partial_i32.float() × wscale.float()).clamp(±FP16_MAX).to(fp16)`."""
    out_fp32 = partial_i32.to(torch.float32) * wscale_fp16.to(torch.float32)
    return out_fp32.clamp(-65504.0, 65504.0).to(torch.float16)


def _compute_int4_partials(act_packed, wgt_packed, block_size):
    """Per-K-block INT4 partial sums — mimics cube mad_s4 output in L0C."""
    M, K2 = act_packed.shape
    N, _ = wgt_packed.shape
    K = K2 * 2
    n_blocks = K // block_size

    act_nibs = _unpack_signed_nibbles(act_packed).to(torch.int32)  # [M, K]
    wgt_nibs = _unpack_signed_nibbles(wgt_packed).to(torch.int32)  # [N, K]
    act_blocks = act_nibs.view(M, n_blocks, block_size)
    wgt_blocks = wgt_nibs.view(N, n_blocks, block_size)
    return torch.einsum("mgb,ngb->gmn", act_blocks, wgt_blocks)


class TestVDEQF16Pack(unittest.TestCase):

    def test_1_pack_roundtrip_lossless(self):
        """fp16 → pack → 19-bit-mini-float decode must be bit-exact."""
        g = torch.Generator().manual_seed(0xC0FFEE)
        n = 1024
        log_lo, log_hi = math.log(1e-4), math.log(1e2)
        magnitudes = torch.empty(n).uniform_(log_lo, log_hi, generator=g).exp()
        signs = torch.empty(n).bernoulli_(0.5, generator=g) * 2 - 1
        scales_fp16 = (magnitudes * signs).to(torch.float16)

        packed = pack_wscales_vdeqf16(scales_fp16)
        self.assertEqual(packed.dtype, torch.uint64)
        self.assertEqual(packed.shape, (n,))

        decoded = torch.tensor(
            [_decode_deqscalar_m1(int(p.item())) for p in packed],
            dtype=torch.float32,
        )
        expected = scales_fp16.to(torch.float32)
        max_abs = (decoded - expected).abs().max().item()
        self.assertEqual(max_abs, 0.0,
                         f"pack roundtrip lost precision: max_abs={max_abs}")

    def test_2_single_kblock_ref_matches_obvious(self):
        """VDEQF16 ref must match `(partial.f32() * w.f32()).to(fp16)`."""
        g = torch.Generator().manual_seed(0xC0FFEE)
        M, N = 128, 256
        partial_i32 = torch.randint(-3000, 3001, (M, N),
                                    generator=g, dtype=torch.int32)
        wscale_fp16 = (torch.rand(N, generator=g) * 0.2 + 0.01).to(torch.float16)

        via_ref = _vdeqf16_ref(partial_i32, wscale_fp16)
        obvious = (partial_i32.to(torch.float32)
                   * wscale_fp16.to(torch.float32)
                   ).clamp(-65504.0, 65504.0).to(torch.float16)

        diff = (via_ref.to(torch.float32) - obvious.to(torch.float32)).abs()
        self.assertEqual(diff.max().item(), 0.0)

    def test_3_kernel_scale_accumulation_within_fp16_ulp(self):
        """At production SVDQuant shape, the diff between current path
        (fp32-throughout) and VDEQF16 path (per-K-block fp16 cast) must
        be at most ~1 ULP of fp16 at the output magnitude."""
        M, K, N = 128, 2048, 256
        block_size = INT4_BLOCK_SIZE
        n_blocks = K // block_size

        act_packed, wgt_packed, ascales, wscales = make_int4_inputs(M, K, N)
        partials = _compute_int4_partials(act_packed, wgt_packed, block_size)

        # PATH_CURRENT — what vec K-loop computes today: fp32 accumulator,
        # ×ascale ×wscale per K-block, single fp16 cast at end.
        acc_current = torch.zeros(M, N, dtype=torch.float32)
        for kb in range(n_blocks):
            p_f32 = partials[kb].to(torch.float32)
            a_f32 = ascales[kb].to(torch.float32).unsqueeze(1)
            w_f32 = wscales[kb].to(torch.float32).unsqueeze(0)
            acc_current += p_f32 * a_f32 * w_f32
        out_current = acc_current.clamp(-65504.0, 65504.0).to(torch.float16)

        # PATH_VDEQF16 — what cube FIX-pipe would write: per-K-block fp16
        # slot with × wscale already folded, vec accumulates × ascale in fp32.
        acc_vdeqf16 = torch.zeros(M, N, dtype=torch.float32)
        slot_saturated = 0
        for kb in range(n_blocks):
            slot_fp16 = _vdeqf16_ref(partials[kb], wscales[kb])
            if (slot_fp16.abs() >= 65504.0).any():
                slot_saturated += 1
            a_f32 = ascales[kb].to(torch.float32).unsqueeze(1)
            acc_vdeqf16 += slot_fp16.to(torch.float32) * a_f32
        out_vdeqf16 = acc_vdeqf16.clamp(-65504.0, 65504.0).to(torch.float16)

        self.assertEqual(slot_saturated, 0,
                         f"{slot_saturated}/{n_blocks} K-blocks saturated fp16 "
                         "— production wscale magnitudes overflow the slot, "
                         "VDEQF16 path is not safe at this shape")

        out_max = out_current.abs().max().item()
        diff = (out_current.to(torch.float32)
                - out_vdeqf16.to(torch.float32)).abs()
        max_abs = diff.max().item()

        # Expected ceiling: ~1 ULP of fp16 at the output magnitude.
        # At |x| ≈ 16, fp16 ULP = 2^-7 ≈ 8e-3. Allow 2× headroom.
        # The kernel-level test in test_gemm_w4a4.py uses atol=5e-2 — this
        # check is tighter (we care about whether VDEQF16 measurably degrades
        # vs the current path, not just that the kernel passes its own test).
        ulp_at_out_max = 2.0 ** (math.floor(math.log2(max(out_max, 1.0))) - 10)
        ceiling = 2.0 * ulp_at_out_max
        self.assertLess(max_abs, ceiling,
                        f"VDEQF16 fp16-cast accumulation exceeds 2 ULPs of "
                        f"fp16 at output magnitude {out_max:.3f}: "
                        f"max_abs={max_abs:.3e} > ceiling={ceiling:.3e}")


if __name__ == "__main__":
    unittest.main()

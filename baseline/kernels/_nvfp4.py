"""NVFP4 row-wise quantization primitive.

NVFP4 = FP4 values (E2M1: sign + 3-bit exponent+mantissa encoding a
level from `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`) with per-16-K-block
FP8-E4M3 scales. NVIDIA-published group convention.

`quantize_nvfp4_rows` / `dequantize_nvfp4_rows` are the canonical pair
used by both `quantize_w4a4_act_fuse_lora` and `gemm_w4a4` refs — and
by the golden-dump generator when it emits random weights for
kernel benchmarks.
"""
from __future__ import annotations

import torch

# Positive magnitudes of the NVFP4 E2M1 value set. Encoding of a nibble
# byte = index into this table (low 3 bits), sign bit in position 3.
_E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)
NVFP4_AMAX = 6.0
# FP8-E4M3 max finite value. Nunchaku clamps scales to this pre-cast
# (`gemm_w4a4.cuh:93`); mirror for bit-for-bit agreement.
FP8_E4M3_MAX = 448.0


def _quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values in `[-6, 6]` to nearest NVFP4 level, return uint8 nibble.

    Midpoint-then-floor rounding (matches hardware `cvt.rn` on the open
    intervals). Tie behaviour may differ on the few rational midpoints
    (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0).
    """
    abs_x = x.abs().clamp_max(NVFP4_AMAX)
    thresholds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32, device=x.device,
    )
    idx = (abs_x.unsqueeze(-1) >= thresholds).sum(-1)
    sign_bit = (x < 0).to(torch.uint8) << 3
    return idx.to(torch.uint8) | sign_bit


def _pack_nibbles(nibs: torch.Tensor) -> torch.Tensor:
    """Pack last-dim pairs of 4-bit nibbles into uint8. Low nibble = even k."""
    assert nibs.shape[-1] % 2 == 0
    lo = nibs[..., 0::2]
    hi = nibs[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """`[*, K/2]` uint8 → `[*, K]` uint8 nibbles. Inverse of `_pack_nibbles`."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=-1)
    return out.view(*packed.shape[:-1], packed.shape[-1] * 2)


def quantize_nvfp4_rows(
    x: torch.Tensor,
    *,
    pad_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise NVFP4 quantization.

    Args:
        x:        `[rows, K]` floating-point tensor. fp16/bf16/fp32 all OK.
        pad_size: row-dim alignment (activations use 256; weights use 1).

    Returns:
        packed: `[rows_pad, K/2]`  uint8 — NVFP4 nibbles, low nibble = even k.
        scales: `[K/16, rows_pad]` fp8_e4m3fn — nunchaku-style transposed layout.

    Padding rows are zeros (packed) and the FP8 cast of 1e-12 (scales).
    """
    assert x.dim() == 2
    rows, K = x.shape
    assert K % 16 == 0, "NVFP4 group size is 16"
    rows_pad = ((rows + pad_size - 1) // pad_size) * pad_size

    x_pad = torch.zeros(rows_pad, K, dtype=torch.float32, device=x.device)
    x_pad[:rows] = x.to(torch.float32)

    blocks = x_pad.view(rows_pad, K // 16, 16)
    amax = blocks.abs().amax(dim=-1)                                 # [rows_pad, K/16]
    scale_f32 = (amax / NVFP4_AMAX).clamp(min=1e-12, max=FP8_E4M3_MAX)
    scale_fp8 = scale_f32.to(torch.float8_e4m3fn)
    scale_back = scale_fp8.to(torch.float32)

    x_scaled = blocks / scale_back.unsqueeze(-1)                     # [rows_pad, K/16, 16]
    nibs = _quantize_e2m1(x_scaled).view(rows_pad, K)
    packed = _pack_nibbles(nibs)                                     # [rows_pad, K/2]
    scales = scale_fp8.transpose(0, 1).contiguous()                  # [K/16, rows_pad]
    return packed, scales


def dequantize_nvfp4_rows(
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Inverse of `quantize_nvfp4_rows`. Returns fp32 `[rows, K]`."""
    rows, K2 = packed.shape
    K = K2 * 2
    assert scales.shape == (K // 16, rows), (
        f"scales shape {tuple(scales.shape)} != expected ({K // 16}, {rows})"
    )
    levels = _E2M1_LEVELS.to(packed.device)
    nibs = _unpack_nibbles(packed).view(rows, K)
    mag = (nibs & 0x07).long()
    sign = 1.0 - ((nibs >> 3) & 0x01).float() * 2.0
    vals = levels[mag] * sign                                        # [rows, K] fp32
    scale_per_block = scales.transpose(0, 1).to(torch.float32)       # [rows, K/16]
    return (vals.view(rows, K // 16, 16) * scale_per_block.unsqueeze(-1)).view(rows, K)


# -------------------------------------------------------------------------
# Nunchaku NVFP4 fragment ↔ row-major adapters.
#
# Layer-2 layouts that the C++ kernel ingests via warp `ldmatrix` /
# `mma.scaled` fragment lanes. Mirrors the int4 helpers in `_int4.py` so
# byte-/scale-level cross-validation against nunchaku is meaningful.
#
# Parameters from `NunchakuWeightPacker(bits=4, warp_n=128)`:
#   warp_n=128, insn_k=comp_k=64, group_size=16, num_lanes=32
#   wscales: s_pack_size=4, num_s_lanes=32, num_s_packs=1, insn_k/group=4
#   qweight: num_n_packs=8, n_pack_size=2, num_n_lanes=8, reg_n=1
#            num_k_packs=1, k_pack_size=2, num_k_lanes=4, reg_k=8
#
# The pack/unpack pair is a pure permute+view chain — bit-equivalent on
# its own. The fp8 cast in `pack_micro_scale` is a separate concern (we
# accept it as part of the `quantize_nvfp4_rows` output dtype).
# -------------------------------------------------------------------------

_NUN_FP4_WARP_N = 128
_NUN_FP4_INSN_K = 64
_NUN_FP4_GROUP = 16


def _nun_fp4_wscale_view_shape(N: int, K: int) -> tuple[int, int, int, int, int, int, int]:
    """7-D view shape `pack_micro_scale` uses for FP4 wscales."""
    assert N % _NUN_FP4_WARP_N == 0, f"N ({N}) must be multiple of {_NUN_FP4_WARP_N}"
    assert K % _NUN_FP4_INSN_K == 0, f"K ({K}) must be multiple of {_NUN_FP4_INSN_K}"
    # (n_tile, num_s_packs=1, s_pack_size=4, 4, 8, k_tile, insn_k/group=4)
    return (N // _NUN_FP4_WARP_N, 1, 4, 4, 8, K // _NUN_FP4_INSN_K, 4)


def pack_nunchaku_wscales_fp4(scales_logical: torch.Tensor) -> torch.Tensor:
    """`[K/16, N] fp8_e4m3fn` row-major → nunchaku fragment-permuted `[K/16, N] fp8`.

    Forward pure permute (no fp8 cast — caller should already be fp8).
    Equivalent to `NunchakuWeightPacker(bits=4, warp_n=128).pack_micro_scale(
    scale_fp16, group_size=16)` minus the dtype cast.
    """
    KG, N = scales_logical.shape
    K = KG * _NUN_FP4_GROUP
    # Input is [K/16, N]; transpose to [N, K/16] to match pack_micro_scale's
    # expected ordering (it views starting from the N axis).
    s_nk = scales_logical.transpose(0, 1).contiguous()                # [N, K/16]
    s_nk = s_nk.view(*_nun_fp4_wscale_view_shape(N, K))               # 7-D
    s_nk = s_nk.permute(0, 5, 1, 4, 3, 2, 6).contiguous()             # mirror pack_micro_scale
    return s_nk.view(-1, N)                                           # [K/16, N]


def unpack_nunchaku_wscales_fp4(scales_nun: torch.Tensor) -> torch.Tensor:
    """Inverse of `pack_nunchaku_wscales_fp4`: `[K/16, N] fragment` → `[K/16, N] row-major`."""
    KG, N = scales_nun.shape
    K = KG * _NUN_FP4_GROUP
    # Post-pack 7-D contiguous shape (after permute, before view-flatten):
    #   (N/128, K/64, num_s_packs=1, 8, 4, s_pack_size=4, insn_k/group=4)
    s7 = scales_nun.view(N // _NUN_FP4_WARP_N, K // _NUN_FP4_INSN_K, 1, 8, 4, 4, 4)
    # Inverse of permute (0, 5, 1, 4, 3, 2, 6) is (0, 2, 5, 4, 3, 1, 6).
    s7 = s7.permute(0, 2, 5, 4, 3, 1, 6).contiguous()                 # [N/128, 1, 4, 4, 8, K/64, 4]
    s_nk = s7.view(N, K // _NUN_FP4_GROUP)                            # [N, K/16]
    return s_nk.transpose(0, 1).contiguous()                          # [K/16, N]


def pack_nunchaku_qweight_fp4(nibs_nk: torch.Tensor) -> torch.Tensor:
    """`[N, K] uint8` (nibble values in low 4 bits) → nunchaku `[N, K/2] int8`.

    Mirrors `NunchakuWeightPacker(bits=4, warp_n=128).pack_weight(weight_int32)`.
    Accepts uint8 input (our nibble convention) and casts to int32 for the
    pack to keep the existing path verbatim.
    """
    N, K = nibs_nk.shape
    assert K % _NUN_FP4_INSN_K == 0, f"K ({K}) must be multiple of {_NUN_FP4_INSN_K}"
    assert N % _NUN_FP4_WARP_N == 0, f"N ({N}) must be multiple of {_NUN_FP4_WARP_N}"
    n_tiles, k_tiles = N // _NUN_FP4_WARP_N, K // _NUN_FP4_INSN_K
    w = nibs_nk.to(torch.int32)
    w = w.reshape(n_tiles, 8, 2, 8, 1, k_tiles, 1, 2, 4, 8)            # 10-D
    w = w.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
    w = w & 0xF
    shift = torch.arange(0, 32, 4, dtype=torch.int32, device=w.device)
    w = (w << shift).sum(dim=-1, dtype=torch.int32)                    # 9-D int32
    return w.view(dtype=torch.int8).view(N, -1).contiguous()           # [N, K/2] int8


def unpack_nunchaku_qweight_fp4(q_nun: torch.Tensor) -> torch.Tensor:
    """Inverse of `pack_nunchaku_qweight_fp4`: `[N, K/2] int8` fragment → `[N, K/2] uint8`
    in our convention (low nibble = even k).
    """
    N, K2 = q_nun.shape
    K = K2 * 2
    assert K % _NUN_FP4_INSN_K == 0
    assert N % _NUN_FP4_WARP_N == 0
    n_tiles, k_tiles = N // _NUN_FP4_WARP_N, K // _NUN_FP4_INSN_K
    # int8 → int32 view (each int32 packs 8 fp4 nibbles, LE bytes)
    q_int = q_nun.contiguous().view(dtype=torch.int32)                 # [N, K/8]
    q_int = q_int.reshape(n_tiles, k_tiles, 1, 8, 8, 4, 2, 2, 1)        # 9-D
    # Expand 8 nibbles per int32; broadcasted shift + mask.
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=q_int.device)
    nibs = ((q_int.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)     # 10-D, last = reg_k
    # Inverse of permute (0, 5, 6, 1, 3, 8, 2, 7, 4, 9) is (0, 3, 6, 4, 8, 1, 2, 7, 5, 9).
    nibs = nibs.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()
    nibs = nibs.view(N, K)                                              # [N, K] uint8 nibbles
    return _pack_nibbles(nibs)                                          # [N, K/2] uint8

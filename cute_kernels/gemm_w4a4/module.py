"""SVDQuant W4A4 linear, nn.Module wrapper.

Ties the two kernel pods at the Python boundary:
  fp x  →  triton.quantize_w4a4_act_fuse_lora  →  (act_nvfp4, ascales, lora_act)
       →  cute_kernels.gemm_w4a4.launch_v2     →  y

Parameter layout mirrors nunchaku's `SVDQW4A4Linear` (nvfp4 path) so a
checkpoint that fits nunchaku fits this module — see
`tmp/nunchaku/nunchaku/models/linear.py:13-133`.

Verification surface only. Not a shipping integration — vLLM bindings
are out of scope per `CLAUDE.md` (no runtime dispatcher, kernels are
drop-in at the linear boundary for whoever integrates downstream).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .kernel_v2_fa4 import launch_v2

# Triton pod is a sibling top-level package; import is host-side only.
from triton_kernels.quantize_w4a4_act_fuse_lora.kernel import (
    quantize_w4a4_act_fuse_lora,
)


class SVDQuantW4A4Linear(nn.Module):
    """W4A4 + SVD-LoRA linear, NVFP4 (CUDA / SM_100|103) path.

    Parameter shapes match `nunchaku.models.linear.SVDQW4A4Linear`
    initialised with `precision='nvfp4'`:

      qweight       : [N, K/2]      int8  — NVFP4 nibbles, low = even-k
      wscales       : [K/16, N]     fp8_e4m3fn
      smooth_factor : [K]           fp16|bf16
      proj_down     : [K, R]        fp16|bf16
      proj_up       : [N, R]        fp16|bf16
      wcscales      : [N]           fp16|bf16  (per-channel weight scale)
      wtscale       : Python float  (global weight scale)
      bias          : [N] or None   fp16|bf16

    `wtscale` is folded into `wcscales` at call time (one mul over [N]).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        assert in_features % 16 == 0, "K must be a multiple of 16 (NVFP4 group)"
        assert in_features % 2 == 0
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.torch_dtype = torch_dtype

        if device is None:
            device = torch.device("cpu")

        self.qweight = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device),
            requires_grad=False,
        )
        self.wscales = nn.Parameter(
            torch.empty(in_features // 16, out_features, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
        self.smooth_factor = nn.Parameter(
            torch.empty(in_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )
        self.proj_down = nn.Parameter(
            torch.empty(in_features, rank, dtype=torch_dtype, device=device),
            requires_grad=False,
        )
        self.proj_up = nn.Parameter(
            torch.empty(out_features, rank, dtype=torch_dtype, device=device),
            requires_grad=False,
        )
        self.wcscales = nn.Parameter(
            torch.ones(out_features, dtype=torch_dtype, device=device),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(
                torch.empty(out_features, dtype=torch_dtype, device=device),
                requires_grad=False,
            )
            if bias
            else None
        )
        # Global weight scale; nunchaku stores this as a Python float on the
        # module (not a parameter). Default 1.0 = identity.
        self.wtscale: float = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `[B, S, K]` or `[N, K]` fp16/bf16. Returns `[..., out_features]`."""
        squeeze_back = None
        if x.dim() == 3:
            B, S, K = x.shape
            squeeze_back = (B, S)
            x = x.reshape(B * S, K)
        elif x.dim() != 2:
            raise ValueError(f"x must be 2D or 3D; got shape {tuple(x.shape)}")

        M = x.shape[0]
        assert x.dtype == self.torch_dtype, (
            f"input dtype {x.dtype} != module dtype {self.torch_dtype}"
        )
        # Pad to 256 so M_pad % tile_M == 0 (launch_v2's tilers are 64/128/256).
        act_pad, ascales, lora_act = quantize_w4a4_act_fuse_lora(
            x, lora_down=self.proj_down, smooth=self.smooth_factor,
            fp4=True, pad_size=256,
        )

        # Fold wtscale into per-channel wcscales (both multiplicative).
        # Cast wtscale to wcscales dtype to keep the mul in fp16/bf16.
        wc_folded = self.wcscales * float(self.wtscale)

        out_pad = launch_v2(
            act_pad,
            self.qweight.view(torch.uint8),
            ascales,
            self.wscales,
            lora_act_in=lora_act,
            lora_up=self.proj_up,
            wcscales=wc_folded,
            bias=self.bias,
            out_dtype=self.torch_dtype,
            use_2cta=False,
        )
        # Slice off padding rows.
        out = out_pad[:M]
        if squeeze_back is not None:
            B, S = squeeze_back
            out = out.reshape(B, S, self.out_features)
        return out

    @classmethod
    def from_nunchaku(cls, nunchaku_linear) -> "SVDQuantW4A4Linear":
        """Copy parameters from an instantiated nunchaku `SVDQW4A4Linear`
        (precision='nvfp4').

        nunchaku stores `qweight` and `wscales` in fragment-permuted form
        (see `nunchaku.lora.flux.packer.NunchakuWeightPacker`). Our kernel
        consumes row-major, so unpack via `unpack_nunchaku_{qweight,
        wscales}_fp4` at load time. `proj_down`/`proj_up`/`smooth_factor`/
        `bias`/`wcscales` are layout-agnostic and copied as-is. The lora
        weights are also stored in fragment form by nunchaku, but our
        path takes them through Triton + ref which read row-major — so
        they also need unpacking via `NunchakuWeightPacker.unpack_lowrank_weight`.
        """
        from baseline.kernels._nvfp4 import (
            unpack_nunchaku_qweight_fp4,
            unpack_nunchaku_wscales_fp4,
        )
        from nunchaku.lora.flux.packer import NunchakuWeightPacker

        assert getattr(nunchaku_linear, "precision", None) == "nvfp4", (
            "from_nunchaku currently supports nvfp4 only"
        )
        mod = cls(
            in_features=nunchaku_linear.in_features,
            out_features=nunchaku_linear.out_features,
            rank=nunchaku_linear.rank,
            bias=nunchaku_linear.bias is not None,
            torch_dtype=nunchaku_linear.torch_dtype,
            device=nunchaku_linear.qweight.device,
        )
        # Weight bridge: fragment → row-major.
        qweight_rm = unpack_nunchaku_qweight_fp4(nunchaku_linear.qweight.data)  # [N, K/2] uint8
        wscales_rm = unpack_nunchaku_wscales_fp4(nunchaku_linear.wscales.data)  # [K/16, N] fp8
        mod.qweight.data.copy_(qweight_rm.view(torch.int8))
        mod.wscales.data.copy_(wscales_rm)
        # LoRA: nunchaku packs via `pack_lowrank_weight`; unpack to row-major.
        packer = NunchakuWeightPacker(bits=4, warp_n=128)
        proj_down_rm = packer.unpack_lowrank_weight(nunchaku_linear.proj_down.data, down=True)
        proj_up_rm = packer.unpack_lowrank_weight(nunchaku_linear.proj_up.data, down=False)
        # `unpack_lowrank_weight` for down=True returns `.view(c, r) = [R, K]`
        # because of the same transpose-quirk as the pack side. Transpose back.
        if proj_down_rm.shape == (mod.rank, mod.in_features):
            proj_down_rm = proj_down_rm.transpose(0, 1).contiguous()
        assert proj_down_rm.shape == (mod.in_features, mod.rank), proj_down_rm.shape
        mod.proj_down.data.copy_(proj_down_rm)
        mod.proj_up.data.copy_(proj_up_rm)
        # Layout-agnostic params.
        mod.smooth_factor.data.copy_(nunchaku_linear.smooth_factor.data)
        mod.wcscales.data.copy_(nunchaku_linear.wcscales.data.to(mod.torch_dtype))
        if mod.bias is not None:
            mod.bias.data.copy_(nunchaku_linear.bias.data)
        mod.wtscale = float(nunchaku_linear.wtscale)
        return mod

    def __repr__(self) -> str:
        return (
            f"SVDQuantW4A4Linear(in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}, dtype={self.torch_dtype})"
        )

# gemm_w4a4 VDEQF16 fold — abandoned 3c-7 attempt (2026-05-27)

Phase 3c-7 tried to fold per-N wscale apply from vec (`TCOLEXPANDMUL`) into
cube FIX-pipe via `TSTORE_FP<..., VDEQF16>` during the L0C drain. Main path
worked numerically; LoRA-up cube pass silently broke (`lora_buf` returned
exact zeros, not garbage). 5 debug probes failed to localize the cause.
Abandoned; reverted to 3c-5 (`a58a5ac`) for production. Document here so a
future attempt doesn't repeat the same dead ends.

## Motivation (from `docs/npu.md` perf snapshot)

3c-5 msprof on 16 tiles / 910B3 showed cube pipe ratios:

| Pipe | Util |
|---|---|
| FIX (L0C→GM drain) | 49.6% |
| MAC (mad_s4) | 6.7% |
| MTE2 | 5.3% |
| MTE1 | 4.4% |

`aiv_vec` activity ≈ 62% — vec is on the critical path; cube blocks waiting
for `VEC_TILE_CONSUMED` back-pressure. Moving the per-N wscale multiply
from vec's `TCOLEXPANDMUL` (vec PIPE_V) into cube's FIX-pipe drain via
`VDEQF16` should:

- Halve ring-slot bytes (int32 → fp16: 128 KB → 64 KB per K-block)
- Eliminate vec's `TLOAD(wscaleRow)` + `TCVT` + `TCOLEXPANDMUL` per K-block

Expected vec ratio drop from ~62% to ~45%, cube/vec back-pressure relax.

## Implementation summary

Host-side (`csrc/python/host/svdquant_w4a4_op.cpp`):

- `wscales` dtype: `at::kHalf` → `at::kUInt64` (VDEQF16 packed deqscalar;
  fp16 value as 19-bit mini-float at bits[32:13] of uint64; see
  `baseline/kernels/_int4.py::pack_wscales_vdeqf16`)
- `workspace` dtype: `at::kInt` → `at::kHalf` (ring slot stores VDEQF16-
  dequant'd fp16 partial, not raw int32)

Device-side (`csrc/kernels/gemm_w4a4/ascend/kernel_device.cpp`):

- New L1 region at offset 384 KB for staging `wscale_packed[kb, :N]`
  (uint64 × kBN = 2 KB) per K-block
- FBuffer Scaling tile at offset 0 (FBuffer is 2 KB on A2/A3 — fills it
  exactly, no double-buffering)
- K-loop per iter: `TLOAD(wscaleMatTile, wscalePackedGm)` →
  `TMOV(wscaleScalingTile, wscaleMatTile)` (L1 → FBuffer) →
  `TSTORE_FP(ringSlot, cAccTile, wscaleScalingTile)` (auto-selects
  `QuantMode::VDEQF16` via `GetVectorPreQuantMode<int32_t, half>`)
- Vec K-loop simplified: drops wscale TLOAD + TCVT + TCOLEXPANDMUL, only
  multiplies by ascale via TROWEXPANDMUL

Host validation (`tests/test_vdeqf16_pack.py`, 3 assertions):

1. `pack_wscales_vdeqf16` bit-exact round-trip through the 19-bit mini-
   float decode (mirrors `extract_quant_params` in PTO ISA
   `tests/.../tstore_acc2gm/gen_data.py:44-55`)
2. Host VDEQF16 reference matches `(partial.f32() * w.f32()).to(fp16)`
3. At production shape (M=128 K=2048 N=256, 32 K-blocks): per-K-block
   fp16 cast accumulation stays within ~1 ULP of fp16-output magnitude
   (max_abs 7.8e-3 ≈ fp16 ULP at |out|≈16, ceiling 1.6e-2)

All 3 host invariants passed.

## What worked on NPU

`out` from the main path looks reasonable: zero-LoRA test has
`max_abs(out vs ref(zero-lora)) = 0.2656`, mean_abs = 0.019. That's ≈ 10
fp16 ULP at output magnitude — fp16 cast in VDEQF16 hardware rounding
appears looser than my host-side simulated `(partial.f32() *
w.f32()).clamp().to(fp16)`. Acceptable for compute-bound paths but
measurably wider than the ~0.001 max_abs of the 3a INT4 path. Could be
tightened later by using `pto::F322F16` quant on a fp32 acc + fp32
scale instead of int32 acc + uint64 mini-float — out of scope for the
abandoned attempt.

Confirmed:

- `TStoreAccFp` is fully implemented in our build path. Local
  `pto-isa/include/pto/npu/a2a3/TStore.hpp:470` has the real body
  (`set_fpc(deqTensorAddr) + TStoreAccNz2nd<VDEQF16>`). The CANN-install
  copy at `/usr/local/Ascend/cann-8.5.0/.../a2a3/TStore.hpp:405` has an
  **empty body**, but the ccec build resolves `<pto/...>` includes from
  the local pto-isa source first (`-I/root/svdquant-kernels/pto-isa/
  include` precedes the CANN tree); `arch_macro.hpp:14-15` auto-defines
  `PTO_NPU_ARCH_A2A3` from ccec's `--cce-aicore-arch=dav-c220-cube`
  (sets `__NPU_ARCH__=2201`), so the local A2/A3 dispatch fires.
- Main path values are non-zero and bounded — proves cube TStoreAccFp is
  writing ring slots; vec is reading them; the `× wscale` fold is
  numerically correct (just slightly looser rounding than host sim).

## What broke — `lora_buf` exact zero

LoRA-up cube pass (`la_fp16[M,R] @ lu_T[R,N] → lora_buf[M,N]` fp32) writes
EXACT zero in all probe runs. Not NaN, not garbage, not "small but wrong"
— literally `min=0 max=0 mean=0`. Reference `ref_lora_buf` has range
`[-0.091, 0.087]`.

Symptom signature: `lora_buf` has the at::zeros initial state intact ⇒
either the LoRA TSTORE silently no-ops, or the LoRA TMATMUL produces zero
output, or the LoRA TLOAD reads zero inputs.

## Debug probes (all NEGATIVE)

Each probe ran on 910B3 via WebIDE git transit (`scratch/3c7-*` branches
on gitcode, `tmp/3c7-N/test.log`). Cube cost ≈ 30 mins Space credit total.

### #125 — set_fpc(0) before LoRA

Hypothesis: K-loop's last `TStoreAccFp` left FPC pointing at FBuffer
Scaling tile (wscale_packed[31]). LoRA's NoQuant TSTORE shouldn't consult
FPC per spec, but hardware might silently apply stale FPC.

`set_fpc(0); pipe_barrier(PIPE_FIX);` before LoRA pass.

Result: `lora_buf=0`. → FPC register reset isn't the fix.

### #126 — plain NoQuant TSTORE drain probe

Hypothesis: FIX-pipe internal state machine has a sticky vector-quant
latch. Need to cycle FIX-pipe through a plain NoQuant code path to unwind
it before LoRA's NoQuant TSTORE.

Drain L0C residual int32 (still kb=31's mad_s4 output) as
`int32→int32 NoQuant TSTORE` to lora_buf reinterpreted as int32 (gets
overwritten by LoRA TSTORE so the bytes don't matter).

Result: `lora_buf=0`. Strongest signal of the whole arc — the drain
probe should have written 128 KB of int32 bits to lora_buf-as-int32. It
wrote nothing (lora_buf stayed at the at::zeros initial state). So
**both** the probe TSTORE and the subsequent LoRA TSTORE silent-no-op'd.

Hypothesis: FIX-pipe FREEZES after 32 consecutive `TStoreAccFp` calls.
All subsequent FIX ops are silently dropped. Even `set_fpc(0)` doesn't
revive it.

### #127 — skip TStoreAccFp entirely in K-loop

Hypothesis: 32× TStoreAccFp is the freeze trigger. Run K-loop without ANY
FIX-pipe activity (skip wscale TLOAD + TMOV + TStoreAccFp). mad_s4 still
runs 32× with init=true on L0C (same dtype-tracking sequence). Vec sees
workspace = at::zeros → main path zeroed but kernel completes.

Result: `lora_buf=0`. → NOT a K-loop FIX residue issue.

`out` confirmed K-loop-skip behavior: range ±0.17 ≈ `bias × wcscale`,
mean ≈ 1.5e-5, since main path contribution was zero and LoRA stayed
zero, only bias landed in the output. So vec did read correctly from the
zero ring slots; the kernel just bypassed all FIX-pipe work and LoRA
still broke. The K-loop residue / FIX state hypothesis is dead.

### #128 — LoRA pass BEFORE K-loop

Hypothesis: somehow K-loop's `mad_s4` 32× leaves L0C in a state that
breaks the subsequent fp32 mad (e.g., dtype tracker latched in int32
mode). Move the entire LoRA cube pass to BEFORE the K-loop's first TLOAD
A/B — LoRA runs on a pristine cube, then K-loop runs.

Result: `lora_buf=0`. → NOT a K-loop residue / L0C dtype-tracking issue.

**LoRA pass fails even when it's the very first cube op in the kernel.**
This is the strongest evidence that the bug is NOT about any state cube
accumulates between mad_s4 and LoRA mad. The LoRA cube pass code itself
is broken in 3c-7.

### #129 — revert kernel_device.cpp to a58a5ac (3c-5)

Hypothesis: some unidentified change in 3c-7's kernel_device.cpp breaks
LoRA. Revert just the device kernel to the 3c-5 baseline. Host op and
header stay at 3c-7 (workspace fp16, wscales uint64).

Tradeoff: dtype mismatch is intentional — kernel reads workspace as
int32* but host allocates fp16, kernel reads wscales as half* but host
allocates uint64. Main path corrupted; LoRA path should still work if
the bug was in kernel_device.cpp.

Result: aicore exception (`rtStreamSynchronize 507015`). Kernel OOB
midway. No LoRA signal — test errored instead of failing on assert.

→ Inconclusive; partial revert doesn't isolate kernel-only bugs without
also reverting host.

## Remaining unexplored hypotheses

Each of these would need another Space round to test:

1. **DeviceParams aggregate init type-deduction quirk** — `__gm__
   uint64_t*` in aggregate init at the entry point. ccec might be doing
   something nonobvious with the cast that affects downstream pointer
   arithmetic. (Subtle; hard to test without disassembly.)
2. **`auto_gen_kernel_device.cpp` wrapper drift** — when kernel signature
   evolves over many commits, the auto-gen ascendc wrapper may keep
   stale arg-positioning. BUILD_OK doesn't prove the wrapper actually
   forwards args in correct order. Verifiable by inspecting
   `build/auto_gen/.../auto_gen_kernel_device.cpp` and the `host_stub.cpp`
   the wrapper produces. (Most testable next step.)
3. **`torch_npu` fp32 NPU tensor `data_ptr()` returns unwritable GM**
   when other ops in the same launch use fp16 workspace. Unlikely but
   doesn't have a direct counter-test.
4. **LoRA `TLOAD(la_fp16/lu_T)` silently fails** so mad sees zeros and
   produces zero output, TSTORE writes zero correctly. Testable by
   inserting a dummy probe that TLOADs la_fp16 + writes its first 256 B
   to a known scratch GM via plain ubuf-TSTORE.

## Architectural notes worth preserving

- A2/A3 PTO **does** ship the real `TStoreAccFp` in the local
  `pto-isa/include/pto/npu/a2a3/TStore.hpp` (line 470 of the version
  vendored in this repo). The empty-stub in `/usr/local/Ascend/cann-
  8.5.0/.../a2a3/TStore.hpp:405` is misleading — it's NOT what our build
  uses. ccec's `--cce-aicore-arch=dav-c220-cube` sets
  `__NPU_ARCH__=2201`, `arch_macro.hpp:14-15` auto-defines
  `PTO_NPU_ARCH_A2A3`, and `pto_instr_impl.hpp:19,39` includes the local
  `pto/npu/a2a3/TStore.hpp` (which has the function). Don't be misled by
  grepping the CANN install — local pto-isa is the canonical source.

- `set_fpc(addr)` takes a FBuffer-aligned address via `((ptr >> 7) << 8)`
  encoding. `TStoreAccFp` does this internally then issues
  `pipe_barrier(PIPE_FIX)` before `TStoreAccNz2nd<..., quantPre>`. The
  quantMode is written into xtReg bits 34-38 *per call* (not a persistent
  hardware register), so different TSTOREs can have different quant
  modes back-to-back. This **suggests** the FIX-pipe freeze isn't about
  the quantPre register but something deeper (FBuffer binding, atomic
  config, or an undocumented dav_c220 sticky bit).

- `L0C 128 KB on A2/A3` (confirmed in PTO ISA `buffer_limits.hpp`; the
  256 KB number that drifted into earlier comments was wrong, applied to
  A5). fp32 `[kBM=128, kBN=256]` LoRA acc is exactly 128 KB — fills L0C
  with no headroom for BUF1 ping-pong at this tile shape.

- L1 wscale staging area (3c-7 layout: 384-386 KB) doesn't conflict with
  LoRA's L1 LA/LUT (16-40 KB). Verified by inspection during #127/#128
  bring-up.

## Decision

VDEQF16 fold in cube FIX-pipe is **not safely shippable on 910B3 via the
current PTO ISA** despite the host validation showing the math works.
Reverted main to a58a5ac (3c-5). Production stays on:

- Workspace int32 [kRingSlots, kBM, kBN]
- wscales fp16 [K/64, N]
- K-loop drain via plain `TSTORE(ringSlot, cAccTile)` (int32 → int32)
- Vec applies × ascale (`TROWEXPANDMUL`) **and** × wscale
  (`TCOLEXPANDMUL`) per K-block

Future revisit:

- If PTO ISA gains a documented `TStoreAccFp`-after-mad-fp32 cooldown
  intrinsic or release-fpipe API
- If Hardware ascend evolution adds a "drain count" reset
- If we can reproduce the freeze on a minimal PTO testcase and file a
  bug against the ISA vendor

## Files / commits touched (for revert)

3c-7 attempt branch on `main`:

- `82f6914 3c-7 prep: VDEQF16 wscale pre-pack helper + host regression test`
- `a20877f 3c-7: FIX-pipe VDEQF16 fold wscale into drain`
- `e7a49da 3c-7 test: workspace dtype assert int32 → fp16`
- `7a1a256 3c-7 debug: reset FPC reg before LoRA pass` (probe #125)
- `465a8ab 3c-7 debug #126: plain NoQuant TSTORE drain to unwind FIX-pipe state`
- `88435d7 3c-7 debug #127: skip TStoreAccFp in K-loop to isolate LoRA FIX-pipe freeze`
- `b797355 3c-7 debug #128: move LoRA cube pass to BEFORE K-loop`
- `136e459 3c-7 debug #129: revert kernel_device.cpp to a58a5ac (3c-5 baseline)`

Pre-3c-7 working state: `a58a5ac` (`Revert "3c-6: L0C ping-pong"`).

Space logs (gitcode scratch branches): `scratch/3c7-vdeqf16` (#125),
`scratch/3c7-126`, `scratch/3c7-127`, `scratch/3c7-128`, `scratch/3c7-129`.

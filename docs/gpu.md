# CUDA backend

## Target SMs

Declared in `cmake/cuda_arch.cmake`. Scope is intentionally narrow:

| SM   | Arch      | Representative parts     |
|------|-----------|--------------------------|
| 100  | Blackwell | B100 / B200              |
| 103  | Blackwell | data-center Blackwell variants |

Everything else (Turing through Hopper, plus consumer Blackwell
SM_120a/121a) is covered by `nunchaku`; this repo exists to fill the
SM_100/SM_103 gap, not duplicate that work. See
`tmp/nunchaku/setup.py:41-64` for nunchaku's arch list.

Override with `-DSVDQUANT_CUDA_ARCHS="100;103"`. Each listed arch
also gets a `SVDQUANT_HAS_SM<N>=1` compile define, so files can opt
in per arch without the build system knowing about them.

## Per-pod layout

```
csrc/kernels/<op>/cuda/
    kernel.cu           # top-level launcher; dispatches by capability
    sm100.cu            # (added when real kernels land)
    sm103.cu
```

The scaffold only ships `kernel.cu` with a host-side stub; per-SM
variants land as real implementations arrive. Real kernels on this
path use **CuTe DSL** (CUTLASS 3.x) for `tcgen05.mma` scaled-MMA
variants — that's what B200's tensor cores speak.

## Build

```
CUDA=ON ASCEND=OFF ./scripts/build.sh
```

or directly:

```
cmake -S . -B build -G Ninja \
    -DSVDQUANT_ENABLE_CUDA=ON \
    -DSVDQUANT_ENABLE_ASCEND=OFF
cmake --build build
```

## Conventions

- Launch signatures take `void* stream` rather than `cudaStream_t` to
  keep the header free of CUDA includes — cast inside `kernel.cu`.
- `TensorRef::data` is a raw device pointer (`T*` cast from
  `cudaMalloc`/PyTorch storage).
- Kernels in `csrc/kernels/` should use CUTLASS 3.x / CuTe DSL
  primitives; bespoke hand-rolled CUDA is for shapes CuTe can't
  cover well.

## When to pick Triton instead

If an op is memory-bound on B200 (AI well below the ~281 FLOP/B FP16
tensor-core ridge) AND needs to also run on Ascend NPU, put it under
`triton_kernels/<op>/` instead — one `kernel.py` runs on both
backends (upstream Triton for CUDA, `triton-ascend` for NPU). See
`../triton_kernels/README.md` for the library-choice rule.

## Gotchas (CuTe DSL traps)

Silent-misbehavior traps on the SM_100 / SM_103 CuTe DSL path —
`const_expr` and `if`, divide-API nesting differences, 2-CTA
`cluster_layout_vmnk` axes, 2-CTA `TiledCopy.partition_D`
rest-mode trap, `num_acc_stage` vs `tile_n` interaction. See
[gotchas/cute_dsl.md](./gotchas/cute_dsl.md). Add new entries
there as you find them.

## Perf-comparison context

### nunchaku is hand-written PTX, not CuTe / CUDA C++

nunchaku's NVFP4 / INT4 scaled-MMA mainloop uses inline `asm
volatile` PTX (`tmp/nunchaku/src/kernels/zgemm/mma_earlycuda.cuh`),
not `cute::gemm` or CUTLASS templates. Register packing, scale
extraction, operand alignment are all manual. It's effectively a
*tuned-for-generation reference* (GB202 / sm_120a era).

When comparing against nunchaku numbers from our CuTe DSL kernels:

- Don't expect apples-to-apples efficiency. The compiler gap is
  real. The last 5-10 pp typically lives in register-allocation
  and instruction-scheduling decisions the PTX author makes
  explicitly but the DSL / MLIR lowering does generically.
- Single-digit pp behind = competitive. 15+ pp behind = something
  structural on our side, not just codegen.
- bf16 vs fp16 asymmetry is typically larger on hand-PTX kernels
  (different mma PTX ops, register banks, swizzle patterns) than
  on DSL output, which goes through the same MLIR path with dtype
  substitution.

Does not apply to the Triton pod
(`quantize_w4a4_act_fuse_lora`) — both sides go through Triton
MLIR, so codegen gap is narrower; wins / losses are about kernel
design, not PTX craft.

### Blackwell NVFP4 routes to `hmma` subpipe in ncu, not a `qmma` subpipe

NVFP4 scaled-MMA on gb100 / B200 (sm_100/103) is **UTCQMMA** at
the SASS level, but ncu's metric tree puts it on the **hmma**
subpipe. There is no standalone `qmma_*` counter — don't waste
time searching by that name.

Useful metrics (queried via `ncu --query-metrics --chips gb100`):

- Pipe util:
  `sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_active`
  (covers HMMA + UTCHMMA + UTCQMMA + UTCOMMA all together).
- FLOPs, FP32 accumulator (TMEM):
  `sm__ops_path_tensor_op_utcqmma_src_fp4_fp6_fp8_dst_fp32`.
- FLOPs, FP16 accumulator:
  `sm__ops_path_tensor_op_utcqmma_src_fp4_fp6_fp8_dst_fp16`.
- Separate FP4-only path (UTCOMMA, different from QMMA):
  `sm__ops_path_tensor_op_utcomma_src_fp4_dst_fp32`.

`--section ComputeWorkloadAnalysis` auto-pulls the subpipe
breakdown — look for "Tensor" rows in SOL / CWA. UTCQMMA work
shows up under "HMMA Pipe" in the SOL "Compute (SM) Pipe
Utilization" panel.

## v2_fa4 baseline on B300 (Verda, 2026-05-12)

First real-hardware run of `kernel_v2_fa4` after the C2 patch
(`8f91240` — defer `pipeline_lora.consumer_wait` into the K-loop
inject site). Host: 2× B300 SXM6 AC, sm_103, ncu unrestricted.

Correctness: `tmp/smoke_gemm_v2_fa4.py` 48/48 pass across
{fp16, bf16} × {1-CTA, 2-CTA} × {wcscales on/off} × {bias on/off}
× R∈{32, 128}, fp16 rel ≤ 8e-4, bf16 rel ≤ 7e-3.

Production-shape TFLOPS (`tmp/bench_gemm_v2_fa4_c1.py`, fp16, 2-CTA):

| M    | K     | N     | R   | TFLOPS  | MFU / 13.5 PFLOPS |
|------|-------|-------|-----|---------|-------------------|
|  256 |  3840 |  3072 | 128 |    35   |    0.3%           |
| 4352 |  3840 |  3072 | 128 |   566   |    4.2%           |
| 4352 |  3840 | 15360 | 128 |  1881   |   13.9%           |
| 4352 | 15360 |  3840 | 128 |  1864   |   13.8%           |
| 4352 | 10240 |  3072 |  32 |  1530   |   11.3%           |

(MFU normalized to a B300 NVFP4 dense peak of 13.5 PFLOPS, ~1.35×
B200. The benched MFU printed by the script uses the B200 10 PFLOPS
constant and overstates by 1.35×.)

LoRA pipeline ladder (M=4352 K=3840 N=3072 R=128 fp16 2-CTA), via
`tmp/profile_gemm_v2_fa4.py --num-lora-stage 0|1|2` under ncu
SpeedOfLight:

| Stage          | Duration | SM%  | Mem% | DRAM% | L2%  |
|----------------|---------:|-----:|-----:|------:|-----:|
| 0 LoRA off     |  44.6 µs | 53.5 | 42.7 |  4.7  | 33.3 |
| 1 pre-C1       |  83.1 µs | 56.0 | 23.6 |  2.8  | 18.9 |
| 2 C1 (+C2 on)  |  70.2 µs | 46.6 | 28.2 |  3.4  | 21.5 |

C1 win (1-stage → 2-stage LoRA prolog): −12.9 µs / −15.6 %.
Reports kept at `log/verda_ncu_v2_C2_stage{0,1,2}_4352_3840_3072_R128.ncu-rep`.

### C2 standalone win (pre-C2 vs C2, both at stage=2)

Swapped `kernel_v2_fa4.py` between `8f91240^` and `8f91240` while
holding everything else constant, same shape and ncu flags:

| Metric           | pre-C2  | C2      | Δ            |
|------------------|--------:|--------:|-------------:|
| Duration         | 71.17 µs | 70.18 µs | -0.99 µs / -1.4 % |
| Compute (SM) %   | 45.00   | 46.55   | +1.55 pp     |
| L2 Cache %       | 20.50   | 21.48   | +0.98 pp     |
| Memory %         | 28.40   | 28.21   | ≈            |
| DRAM %           |  3.31   |  3.35   | ≈            |
| SM Active cycles | 63549   | 64068   | +0.8 %       |

Story is clean: deferring `pipeline_lora.consumer_wait` lets the MMA
warp start issuing main atom #0 ~1 µs before LA/LU TMA arrives. The
saved cycles surface as +1.55 pp SM throughput. Memory side is
unchanged — C2 is a scheduling change, not a bandwidth change.

Reports: `log/verda_ncu_v2_{preC2,C2}_stage2_4352_3840_3072_R128.ncu-rep`.

Reproduction script (uses an EXIT trap to guarantee the C2 file is
restored even on ncu failure): `tmp/verda_c2_ab.sh`.

## v2_fa4 SMEM budget at the production shape (B300, 2026-05-12)

Probed on Verda via a `print` injected into `_compute_stages`. All
numbers below are per-CTA, occupancy=1, tile=(256, 128, 64), R=128,
fp16 ab/c, fp4 mma a/b, fp8 sf.

| Component                            | Bytes  | KB  |
|--------------------------------------|-------:|----:|
| SMEM capacity (sm_100 == sm_103)     | 232448 | 227 |
| `ab_bytes` per stage (A+B+SFA+SFB)   |  28672 |  28 |
| `c_bytes_per` per epi stage          |   8192 |   8 |
| `mbar_helpers`                       |   1024 |   1 |
| `LA` per CTA (`tile_m*R/cta_group`)  |  32768 |  32 |
| `LU` per CTA (`tile_n*R`)            |  32768 |  32 |
| per-stage LoRA = LA+LU               |  65536 |  64 |

Stage-by-stage feasibility on this shape:

| num_lora_stage | LoRA  | c(2)  | ab budget | ab_stages | fit?  |
|---------------:|------:|------:|----------:|----------:|:------|
|             2  | 128 K | 16 K  |   82 K    |     2     | yes   |
|             3  | 192 K | 16 K  |   18 K    |     0     | **assert** |
|             3  | 192 K | 8 K (c=1) | 26 K  |    0     | still no |

The headroom for a 3rd LoRA stage is one full LoRA stage short:
each costs 64 KB but only ~26 KB of slack exists after c_stage=1.
Naive stage=3 doubles LoRA SMEM (128 KB → 192 KB), which violates
the "without doubling" constraint of task #58 anyway.

> **2026-05-13 follow-up: the LU row above is wrong by 2×.** The
> handwritten `lu_bytes` formula treated LU as full N=128 per CTA,
> but the 2-CTA dense MMA atom **halves LU via N-split** inside
> `partition_shape_B` (same mechanism that halves main B). Real LU
> per CTA = 16 KB / stage, not 32 KB. See the next section for the
> probe, the fix, and the much larger win it unlocked. The
> "paths to stage=3" list above is preserved for context, but is
> now mooted — stage=3 became feasible with no code redesign, and
> the bench in the next section shows it is also no longer the
> right knob to tune.

The probe artifact lives at `tmp/probe_smem_budget.py` and the
inline `_compute_stages` print used to capture the numbers above
was reverted in this commit.

## v2_fa4 LU SMEM accounting fix (B200, 2026-05-13)

The handwritten `lora_smem_bytes` in `_setup_attributes` over-counted
LU by 2× — `_compute_stages` therefore reserved double the LoRA SMEM
it needed, and `num_ab_stage` was clamped to 2 instead of 4 at the
R=128 production shape. This was a single-line bug that *silently
hid the real perf headroom* behind a misleading SMEM-budget message.

### Probe (task #96)

Injected `cute.cosize(slice_(lu_smem_layout_staged, ...))` into
`_setup_attributes` so the actual per-stage byte count surfaces at
trace time:

```
[PROBE96] num_lora_stage=2 cta_group_size=2
[PROBE96] la_one cosize=16384 -> 32768 B (handwritten 32768 B, factor 1.000)
[PROBE96] lu_one cosize=8192  -> 16384 B (handwritten 32768 B, factor 0.500)
```

LA matches (M-split was already correct in the handwritten formula).
LU is half — confirms the Modular blog claim (Part 3, "2xSM MMA: Shared
Memory Optimization") that the 2xSM atom halves the B tile via
`partition_shape_B`. The fix is one extra `// self.cta_group_size`
on the `lu_bytes` line; comment in
`cute_kernels/gemm_w4a4/kernel_v2_fa4.py::_setup_attributes` cites
this section.

### Re-solved budget (R=128, fp16, 2-CTA, tile=(256, 128))

| Component                            | Bytes  | KB  |
|--------------------------------------|-------:|----:|
| SMEM capacity (sm_100 == sm_103)     | 232448 | 227 |
| `LA` per CTA (M-split)               |  32768 |  32 |
| `LU` per CTA (**N-split, was 32**)   |  16384 |  16 |
| per-stage LoRA = LA+LU               |  49152 |  48 |
| `ab_bytes` per stage                 |  28672 |  28 |
| `c_bytes_per` per epi stage          |   8192 |   8 |

Feasibility per `num_lora_stage`:

| stage | LoRA  | c stages chosen | ab stages chosen | fit? |
|------:|------:|----------------:|-----------------:|:-----|
|     2 |  96 K |               2 |            **4** | yes  |
|     3 | 144 K |               3 |                2 | yes  |
|     4 | 192 K |               1 |                1 | assert |

The pre-fix code thought stage=2 had only 2 ab_stages of headroom and
stage=3 didn't fit at all. Post-fix, stage=2 lands at ab=4 and stage=3
becomes solvable too.

### Wall-clock impact at stage=2 (the actual main path)

Comparing the same `tmp/bench_gemm_v2_fa4_c1.py` shapes pre-fix
(B300, doc'd) vs post-fix (B200, fresh run, fp16, 2-CTA):

| M    | K     | N     | R   | pre-fix TF (B300) | post-fix TF (B200) | Δ          |
|------|-------|-------|-----|------------------:|-------------------:|-----------:|
|  256 |  3840 |  3072 | 128 |              35   |              **108** | +209 %   |
| 4352 |  3840 |  3072 | 128 |             566   |             **1685** | +198 %   |
| 4352 |  3840 | 15360 | 128 |            1881   |             **2648** | +41 %    |
| 4352 | 15360 |  3840 | 128 |            1864   |             **2735** | +47 %    |
| 4352 | 10240 |  3072 |  32 |            1530   |             **2645** | +73 %    |

(Numbers are absolute TF and so cross-card comparable; B300 has 1.35×
more peak NVFP4 than B200, so a "same TF" reading would still mean we
got faster against a weaker card. Post-fix bench uses 20 warmup + 500
timed iters (`bench_gemm_v2_fa4_c1.py`); a 3-warmup / 50-iter version
under-counted the R=128 shape by ~10 % — see "Bench warmup gotcha"
note below.)

### MFU vs nunchaku (RTX PRO 6000 SM_120a, 4 PFLOPS peak)

Post-fix MFU on B200 (10 PFLOPS NVFP4 peak) vs nunchaku reference
numbers hardcoded in `tmp/bench_gemm_v2_fa4_c1.py:113-119`. nunchaku
on RTX PRO 6000 is hand-written PTX (`tmp/nunchaku/src/kernels/zgemm/
mma_earlycuda.cuh`), so any single-digit-pp gap is in the noise of
"CuTe DSL MLIR codegen vs hand-rolled PTX" (see § "Perf-comparison
context" above).

| Shape (M, K, N, R)              | ours fp16 | nunchaku fp16 |  Δ pp | ours bf16 | nunchaku bf16 |  Δ pp |
|---------------------------------|----------:|--------------:|------:|----------:|--------------:|------:|
| 4352 × 3840  × 3072  × R=128    |  **16.9** |          16.2 |  +0.7 |      17.3 |          17.7 |  −0.4 |
| 4352 × 3840  × 15360 × R=128    |  **26.5** |          19.5 |  +7.0 |  **26.7** |          24.7 |  +2.0 |
| 4352 × 15360 × 3840  × R=128    |  **27.3** |          25.0 |  +2.3 |      27.3 |          30.5 |  −3.2 |
| 4352 × 10240 × 3072  × R=32     |  **26.4** |          21.4 |  +5.0 |  **26.2** |          25.2 |  +1.0 |

**fp16: 4/4 shapes ahead. bf16: 3/4 shapes ahead.** Remaining gap
lives entirely in the bf16 column on the `M=4352 K=15360 N=3840`
shape (−3.2 pp), which has nothing to do with LoRA — it's the
"bf16 mma PTX path vs DSL MLIR lowering" asymmetry the perf-comparison
section already calls out, and would not be moved by any LoRA-side
optimization.

Absolute throughput (since the two cards' peaks differ 2.5×):

| Shape                           | ours TF (B200) | nunchaku TF (RTX PRO 6000) | ratio |
|---------------------------------|---------------:|---------------------------:|------:|
| 4352 × 3840  × 3072  × R=128    |           1685 |                       ~648 | 2.60× |
| 4352 × 3840  × 15360 × R=128    |           2648 |                       ~780 | 3.40× |
| 4352 × 15360 × 3840  × R=128    |           2735 |                      ~1000 | 2.74× |
| 4352 × 10240 × 3072  × R=32     |           2645 |                       ~856 | 3.09× |

### Bench warmup gotcha

The first iteration of `kv2.launch_v2` on a fresh shape triggers the
CuTe DSL JIT compile path (MLIR lowering → PTX → SASS). Subsequent
iterations hit the compile cache. On B200 the first iter takes
hundreds of milliseconds; iters 2–5 still see the SM-frequency ramp
and one-shot allocator setup. Pre-2026-05-13 `bench_gemm_v2_fa4_c1.py`
used `warmup=3, iters=50` and consistently under-counted the LoRA
R=128 production shape by ~10 % (1532 TF reported, real 1685 TF —
that's the "0.9 pp behind nunchaku → 0.7 pp ahead" delta we chased
post-LU-fix). Now pinned at `warmup=20, iters=500`. If you see
*another* round of "we got worse without changing anything", check
the warmup count first.

Logs:
`log/verda_bench_lufix.log` (initial bench, undercounted),
`log/verda_bench_lufix_warmup.log` (post-warmup-fix, current
numbers), `log/verda_tiler_sweep.log` (tiler (256, 64/128/256)
A/B that initially caught the variance).

### Stage sweep — `num_lora_stage` is no longer the bottleneck

Post-fix wall-clock sweep at M=4352 K=3840 N=3072 R=128 fp16 2-CTA
(`tmp/bench_gemm_lora_stage_sweep.py`, 200 iter, CUDA-event timing):

| stage | µs/launch | TFLOPS | (num_ab, num_lora, num_c) | vs stage=2 |
|------:|----------:|-------:|--------------------------:|-----------:|
|     0 |     51.82 |   1981 |                   (7, 0, 3) | −10.76 µs / −17.2 % |
|     1 |     86.36 |   1189 |                   (5, 1, 4) | +23.78 µs / +38.0 % |
| **2** | **62.58** | **1641** |               **(4, 2, 2)** | (baseline) |
|     3 |     73.10 |   1405 |                   (2, 3, 3) | +10.52 µs / +16.8 % |

Stage=3 is *feasible* but **slower**: the solver buys the extra LoRA
prolog by giving up two main `num_ab` stages, and the main K-loop
loses more than the LoRA prolog gains. This kills tasks #58 (deepen
prolog) and #59 (multicast LoRA TMA) as wins — both were proposed
under the false-assumption regime; the real ceiling now sits in main
K-loop / TMEM occupancy, not LoRA-side latency hiding.

LoRA overhead at the new baseline: 62.58 − 51.82 = 10.76 µs / +20.8 %
on top of the LoRA-off path. That delta is what tasks #60 (overlap
LoRA MMA with main K-loop epilogue tail) and future work would
target, not LoRA prolog depth.

Log: `log/verda_lora_stage_sweep.log`.

### ncu A/B at the production shape (same B200, same launch config)

Reports captured 2026-05-13 on the same Verda B200 instance: HEAD^
(pre-LU-fix, `num_ab=2`) vs HEAD (`7296e90`, post-LU-fix, `num_ab=4`).
Same shape, same launch flags, same `num_lora_stage=2`. The kernel was
swapped on-disk between runs (the script ships with an EXIT trap to
guarantee restore on failure — `tmp/verda_lufix_ncu_ab.sh`).

| Metric                  | pre-LU-fix | post-LU-fix | Δ                  |
|-------------------------|-----------:|------------:|-------------------:|
| Duration                |  46.69 µs  |   32.13 µs  | **−14.56 µs / −31.2 %** |
| Compute (SM) %          |  41.63     |   53.62     | **+11.99 pp**      |
| Memory %                |  25.58     |   38.91     | +13.33 pp          |
| L1/TEX Cache %          |  28.50     |   44.75     | +16.25 pp          |
| L2 Cache %              |  24.57     |   36.18     | +11.61 pp          |
| DRAM %                  |   5.04     |    7.31     | +2.27 pp           |
| SM Active Cycles        |  72 433    |   46 126    | **−36.3 %**        |
| Memory Throughput       |   386 GB/s |    561 GB/s | +45 %              |
| Achieved Occupancy      |    8.55 %  |     8.66 %  | ≈                  |
| Grid Size / Block Size  | 148 / 192  |  148 / 192  | identical          |

Reads consistent with the budget story: same launch shape (148 ×
192-thread blocks, ~8.6 % occupancy), 2× more `num_ab` stages keep the
SM-side pipeline fed → SM% jumps +12 pp and SM Active Cycles drop 36 %.
L1/TEX and L2 throughput both rise proportionally because the TMA
producers now have more in-flight in-flight buffers to fill (it's not a
"bandwidth saving" — it's the bandwidth being more *evenly used* across
the kernel's wall-time). DRAM stays low (compute-bound regime
preserved).

The ncu single-launch Duration (32.13 µs) is lower than the bench-side
CUDA-event average (62.58 µs / iter): the bench averages over a tight
200-iter Python loop with `cute_dsl` launch overhead included; the ncu
report measures just the device-side kernel. Both directions agree;
treat the bench number as "kernel + launch tax" and the ncu number as
"kernel only."

Reports kept at
`log/ncu_v2_{preLUfix,postLUfix}_4352_3840_3072_R128.ncu-rep` and the
text excerpt at `log/verda_ncu_lufix_ab.log`.

## Epilogue precision: fp16 vs fp32

nunchaku casts `fp32 → fp16` immediately after the main `tcgen05`
fp32 accumulator and runs the **entire** post-MMA chain (LoRA-up,
`wcscales`, bias) in fp16 (`gemm_w4a4.cuh:351`,
`gemm_base.cuh:711-770`):

```cpp
auto f16psum = packed_fp32_to_fp16(fpsum);        // gemm_w4a4.cuh:351
Epilogue()(binfo, f16psum, ...);                  // LoRA + Bias all fp16
fsum.data[0] = __hfma2(fsum.data[0], s1, b1);     // wcscales×y + bias
```

This is a **consumer-Blackwell** tradeoff: on SM_120 / SM_121
(RTX 50-series, RTX-PRO 6000), non-FP4 fp32-accumulate paths run at
**half rate** vs fp16-accumulate — Nvidia gates the fp32-accum tensor
throughput on consumer parts. Doing the epilogue in fp16 keeps the
post-MMA work on the full-rate path.

**Data-center Blackwell (SM_100 / SM_103, B200) is NOT throttled.**
fp32 epilogue runs at the same throughput as fp16. So our CuTe DSL
kernel runs the epilogue in fp32 until the final store — we don't
inherit nunchaku's tradeoff and don't lose anything by skipping it.

Numerical consequence in cross-validate vs nunchaku
(`tmp/smoke_nvfp4_vs_nunchaku.py`, SM_120 local):

| config | rel_max | rel_mean | source |
|---|---:|---:|---|
| min  (smooth=1, bias=0, wcscales=1, no LoRA) | 8.8 %  | 1.2 % | one fp32→fp16 cast difference at line 351 |
| full (random affine + LoRA) | 35.7 % | 3.7 % | + fp16 epilogue FMA noise stacked over R-dot + wcscales + bias |

The activation-quantize side is **not** a contributor —
`quantize_w4a4_fp4_from_fpsum_warp` (`gemm_w4a4.cuh:85-187`) uses
NUM_GROUPS=4 with `__shfl_xor` reduce across the 4-lane quad, giving
strict per-row-per-16-K-block amax, identical to our Triton convention.
(`bench_fused.py:17-25` warp-fragment-amax comment refers to the INT4
path, group_size=64 — does not apply to FP4.)

Implication for end-to-end quality: deepcompressor calibration assumes
per-row-per-16-K-block (NVFP4 standard). Both nunchaku and ours follow
that. The fp16 vs fp32 epilogue is independent — ours preserves more
precision in LoRA-up / affine fold-in, marginally better on the
calibration's loss surface, but the difference is well below the noise
floor of any image-quality metric.

## Cross-arch MFU caveat

Cross-chip MFU (FLOPS / device peak) is **not a kernel-quality metric**
when one side is consumer-Blackwell and the other is data-center
Blackwell. Two unrelated knobs move:

1. **Sustained clock**. B200 sustains its boost clock at the rated
   number; RTX-PRO 6000 / RTX 50-series boost clocks swing wide with
   thermal envelope and per-die binning. The "peak FLOPS" denominator
   in NV's spec sheet is one specific clock; the runtime may be above
   or below. Consumer-card MFU readings can briefly exceed 100 % or
   sit well under, neither reflecting code quality.
2. **fp32-accum throttle**. Consumer parts halve non-FP4 fp32-accum
   tensor throughput vs fp16-accum (see § *Epilogue precision*).
   Whether the "peak" denominator factors this in depends on which
   row of the spec sheet you read.

Replace MFU with `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_*`
for cross-arch comparison. ncu computes this against each device's
**per-cycle** tensor-pipe ceiling — clock-independent. Two flavours:

- `..._active` — over cycles when the SM has work, how often the
  tensor pipe is busy. Reads kernel instruction density / arith mix.
- `..._elapsed` — over the kernel's full wall time, same numerator.
  Reads end-to-end tensor pipeline saturation.

### Same shape (M=4352 K=3840 N=3072 R=128), three reports

| Kernel             | Device              | dtype | Duration   | Tensor (active) | Tensor (elapsed) |
| ------------------ | ------------------- | ----- | ---------: | --------------: | ---------------: |
| v2_fa4 (post-LU)   | B200, SM_100        | fp16  |   32.13 µs |    52.0 %       |      45.2 %      |
| nunchaku           | RTX PRO 6000, SM_120a | fp16 |  185.25 µs |    58.8 %       |      45.4 %      |
| nunchaku           | RTX PRO 6000, SM_120a | bf16 |  157.54 µs |    73.7 %       |      54.6 %      |

Read:
- **fp16 elapsed % is the same** within 0.2 pp (45.2 vs 45.4). Both
  kernels saturate their respective tensor pipes equally over the
  kernel's run. The 5.8× absolute duration gap = B200 SM count +
  per-cycle FP4 peak; not a code-quality gap.
- **fp16 active % differs** (52.0 vs 58.8). nunchaku's hand-PTX
  packs more tensor work per active cycle; ours has more bubble
  cycles. That's where DSL-vs-PTX codegen gap shows up.
- **bf16 nunchaku active 73.7 %** is the fp16-spill-free run.
  Consumer-Blackwell nunchaku fp16 hits 255 regs + 2.28M LMEM (101%
  spill overhead, see § *Perf-comparison context* earlier); bf16
  doesn't. Ours doesn't have this cliff in either dtype.

Caveat the caveat: `peak_sustained_active` is the **per-architecture**
tensor-pipe peak per cycle. If sm_120a has a lower FP4 peak than
sm_100, the % still normalizes correctly within each device, but the
*absolute work* per percentage point differs. Use Tensor % to compare
implementation density; use duration × device peak to compare absolute
throughput.

Reports: `log/ncu_v2_postLUfix_4352_3840_3072_R128.ncu-rep` (B200),
`log/ncu_nunchaku_4352_3840_3072_R128_{fp16,bf16}.ncu-rep` (Verda
RTX PRO 6000). Extract via:

```
ncu --import <file> --page raw 2>/dev/null \
  | grep -E 'sm__pipe_tensor_cycles_active.avg|gpu__time_duration.avg'
```

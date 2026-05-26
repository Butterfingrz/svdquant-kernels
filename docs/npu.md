# Ascend NPU backend

Huawei CANN / AscendC. Other NPU vendors (Cambricon, etc.) are out of
scope for this repo; add them as sibling directories
(`csrc/kernels/<op>/<vendor>/`) if needed later.

## Toolchain

- **CANN toolkit** at `${ASCEND_HOME_PATH}` (default
  `/usr/local/Ascend/ascend-toolkit/latest`).
- `ccec` — AscendC device-side compiler (for `__aicore__` code).
- Host-side ACL runtime (`libascendcl`, `libruntime`).
- **`triton-ascend`** (Python package) — required only for Triton
  kernels under `triton_kernels/`. Install on the Ascend host; not
  required for AscendC-only builds, and not required at all on the
  local cross-compile host.

Before configuring CMake:

```
source scripts/env_ascend.sh
```

This sources the CANN environment (`setenv.bash` / `set_env.sh`).
`cmake/FindCANN.cmake` then locates headers, libs, and `ccec`.

## Per-pod layout (AscendC pods)

```
csrc/kernels/<op>/ascend/
    kernel.cpp                # host launcher (plain C++)
    kernel_device.cpp         # (added later) __aicore__ kernel, compiled by ccec
```

`kernel.cpp` is compiled by the host C++ compiler and today is a stub.
When the first real AscendC kernel lands, we add a second file for the
device code and a custom build rule that feeds it to `ccec` and links
the resulting object into the pod's OBJECT library.

## Per-pod layout (Triton pods)

```
triton_kernels/<op>/
    kernel.py                 # same source as the CUDA path
    README.md
```

Triton pods don't go through CMake or `ccec`. `triton-ascend`
JIT-compiles `kernel.py` to AscendC at first call on the NPU; on
CUDA, upstream Triton JITs the same file to PTX. One source, two
backends.

## Build

```
source scripts/env_ascend.sh
CUDA=OFF ASCEND=ON ./scripts/build.sh
```

or directly:

```
cmake -S . -B build -G Ninja \
    -DSVDQUANT_ENABLE_CUDA=OFF \
    -DSVDQUANT_ENABLE_ASCEND=ON
cmake --build build
```

Triton pods are not part of this build — they aren't compiled ahead
of time, they just need to be importable at runtime on the host.

## Conventions

- AscendC launch signatures take `void* stream`; cast to
  `aclrtStream` inside `kernel.cpp`.
- `TensorRef::data` is a device address (what `aclrtMalloc` returns).
- AscendC kernels should use the tiling helpers rather than
  hand-rolling DMA; cube unit for GEMM-shaped math, vector unit for
  elementwise / reductions.

## When to pick Triton instead of AscendC

If an op is memory-bound (AI below ~90 FLOP/B in practice) AND the
same op needs to run on CUDA too, put it under `triton_kernels/<op>/`
instead of writing AscendC. One `kernel.py` saves having to maintain
parallel CuTe DSL + AscendC implementations. Compute-bound ops and
NPU-only ops still belong here.

## Perf — `gemm_w4a4` on 910B3 (Phase 3c-4)

### Architectural tax: cube/vec partition costs SVDQuant fine-grained dequant

Before any kernel-side number, the big picture: SVDQuant has per-64-K-block
ascale/wscale, so the math forces an int32→fp32 dequant **inside** the K-loop.
On the two backends this lands very differently:

| | nunchaku (CUDA SM) | this kernel (Ascend cube + vec) |
|---|---|---|
| `mma` output location | per-thread registers (SM-private RF) | **L0C 128 KB** (cube-private SRAM) |
| Where the dequant runs | same warp's CUDA cores | **vec unit** — physically separate core |
| `mma`→dequant data flow | zero-copy (regs → regs) | **L0C → GM ring → UB** (via L2) |
| Per-K-block drain | none — `fpsum` stays in regs | **1 drain × 128 KB × 32 K-blocks = 4 MB / tile** |
| K-loop tail store | 1 × `fpsum`→GM | 1 × `out`→GM (after vec finishes) |

Reference for the CUDA path: `tmp/nunchaku/src/kernels/zgemm/gemm_base.cuh:367-409`
`apply_scales` — it just does `__hfma2(int2half2(psum), asx * ws, fsum)` inline
in the warp; the `mma` callback returns an int32 register tile that the next
two instructions consume directly. There is no L0C-equivalent on a CUDA SM
because the matmul output IS already in the same register file the dequant
reads from.

On Ascend the cube and vec are **physically distinct hardware** with no shared
SRAM — L0C is cube-only, UB is vec-only. So every fine-grained dequant pass
must round-trip through GM (L2-resident in practice, never HBM — see
[gotchas/ascend.md](./gotchas/ascend.md) cube↔vec L2 entry). This is the
**architectural tax** that drives the FIX-pipe ratio we measure below; it is
not an implementation gap.

### Status

Multi-tile launch validated (grid M-major, `blockDim = M_total / kTileM`);
WebIDE 910B3 numerical pass `out vs ref max_abs ≈ 0.008` independent of
`N_TILES`.

### Wall-clock sweep (`tmp/bench_3c4_sweep.py`)

Per-call wall (includes host launch overhead):

| Tiles | M | µs/call | INT4 GOPS | scale vs ideal |
|------:|---:|--------:|----------:|---------------:|
| 1 | 128 | 308.93 | 434 | — |
| 16 | 2048 | 318.46 | 6743 | **15.52× / 16×** |
| 24 | 3072 | 388.75 | 8286 | 19.07× / 24× |

Scaling 1→16 ≈ 97 % of ideal linear — cube cores actually parallel.
Wall barely moves through 16 tiles ⇒ the ~310 µs base is **host
overhead** (per-call `aclrtMalloc` + `aclrtMemcpy(88 B)` +
`aclrtSynchronize` + Python dispatch), cube work itself is tens of µs.

### `msprof --aic-metrics=PipeUtilization` at 16 tiles

Device-side `Task Duration ≈ 57 µs/call` (vs 318 µs wall) — confirms
host overhead is ~260 µs per launch. Cube pipe ratios (median over
25 captured calls):

| Pipe | Ratio | What |
|------|------:|------|
| **FIX (L0C → GM TSTORE)** | **49.6 %** | per-K-block int32 partial drain |
| MAC (mad_s4) | 6.7 % | actual cube math |
| MTE2 (GM → L1) | 5.3 % | TLOAD act/wgt |
| MTE1 (L1 → L0A/B) | 4.4 % | TEXTRACT sub-tile |
| (bubble) | ~34 % | mostly cube ↔ vec back-pressure |

Vec pipe ratios: vec 62.1 %, scalar 18.9 %, MTE2 24.6 %, MTE3 0.5 %.
Overall `cube_utilization` = 74.4 %.

### Why FIX dominates (SVDQuant algorithm cost, not implementation bug)

SVDQuant has per-64-K-block ascale/wscale, so the math forces dequant
**inside** the K-loop. We mirror this with a 32-iter K-loop where each
iter:

1. cube `mad_s4` accumulates one K-block partial in L0C int32
2. cube `TSTORE` drains [128, 256] int32 = 128 KB → workspace ring
3. vec TLOADs the slot, applies ascale row + wscale col, accumulates
4. cube reuses single-buf L0C for the next K-block

The L0C single-buf forces MAC → FIX → MAC serialization per K-block;
MAC can't overlap with FIX. 32 × 128 KB = 4 MB drained per tile.
Regular dense GEMMs drain L0C once at the end — they can pin MAC near
peak. Here we can't, because the per-K-block scales have to be
applied between drains.

Achieved cube INT4 = 2147 MFLOPs / 53.35 µs ≈ **40.2 TOPS** (kernel
time), ≈ **6.8 TOPS** (wall time, what vLLM actually sees). Against an
estimated 910B INT4 cube peak (~512 TOPS, conservative), that's 7.9 %
device-side MFU and 1.3 % wall-clock MFU.

The headline gap is **host overhead** (320 µs wall vs 57 µs kernel),
*not* the cube kernel itself. Three optimization vectors, ranked by
expected return:

1. **Pre-allocated `dev_params` / stream-resident param buffer** —
   removes `aclrtMalloc` + `aclrtMemcpy` per call. Lifts wall MFU
   ~5×. Touches `kernel.cpp` host launcher only.
2. **L0C ping-pong (BUF0 / BUF1)** — overlaps MAC with FIX drain.
   Lifts MAC ratio 6.7 % → estimated ~30 %. Touches
   `kernel_device.cpp` cube path only.
3. **Batch 2 K-blocks per drain in vec UB** — halves drain frequency
   but doubles UB scratch. Algorithm-compatible (vec applies the two
   K-block scales separately after a fatter TLOAD). Last because UB
   is already 135 / 184 KB used at production shape.

## Gotchas (Ascend / PTO ISA traps)

Silent-misbehavior traps on the 910B (a2a3) cube + vec mix-mode
path — cube↔vec handoff is L2-resident not HBM, cube min
addressable is 1 byte (no INT4 tile dtype), `TLoad` of ColMajor
`[N, 1]` from GM only loads the head element, `TRowExpand`
leaves the vec mask register contaminated, AIV K-loop reusing a
partial UB region needs V→MTE2 cross-iter sync. Also the
hardware-level reasoning behind "W4A4 cube uses raw `mad_s4`
inside svdquant, not a PTO wrapper". See
[gotchas/ascend.md](./gotchas/ascend.md). Add new entries there
as you find them.

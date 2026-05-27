// gemm_w4a4 — Ascend __aicore__ device kernel.
//
// Phase 3a: real INT4 cube MMA + per-K-block dequant in vec.
//
// Tile shape (hardcoded for 3a; tile-parameterization is 3b/3c):
//   M = 128, K_logical = 2048, K_packed = 1024, N = 256
//   K-block = 64 logical = 32 packed bytes  (== mad_s4 KS == INT4_BLOCK_SIZE)
//   ⇒ 32 K-blocks per tile, 1 tile per launch.
//
// Buffers:
//   L1 (512 KB):
//     A tile [128, 1024] int8_t = 128 KB at offset 0
//     B tile [1024, 256] int8_t = 256 KB at offset 128 KB
//     Total 384 KB (no L1 ping-pong; whole BK loaded once per launch).
//   L0A / L0B (64 KB each):
//     ping-pong sub-tiles for one K-block (sizes set in
//     pto_macro_matmul_s4.hpp). 4 KB / 8 KB tiles → ample headroom.
//   L0C (128 KB on A2/A3 — see docs/gotchas/ascend.md):
//     single int32 [128, 256] = 128 KB at offset 0, fills L0C exactly.
//     No room for ping-pong at this tile shape. Each K-block writes
//     init=true and is drained to GM workspace before the next
//     mad_s4 overwrites L0C. FIX-pipe reduction via drain batching
//     (task #122 / 3c-7) does not need a second L0C buffer.
//   GM workspace (caller-allocated):
//     int32 [kRingSlots=6, 128, 256] cube/vec hand-off ring (768 KB).
//   GM out (caller-allocated): fp16 [128, 256] = 64 KB final.
//
// Cube path (per K-block):
//   wait FIX→M  (prev TSTORE off L0C)         ← skipped on first iter
//   wait M→MTE1 (prev TEXTRACT on this pingpong drained)
//   slide L1 view to kb-th K-block via TASSIGN from saved bases
//   pto_macro_matmul_s4_block: TEXTRACT + mad_s4 (init=true)  → L0C
//   set M→FIX  ;  wait M→FIX (gate TSTORE on mad_s4 done)
//   TSTORE L0C → workspace[slot=kb%kRingSlots]
//   ffts_cross_core_sync(FIX, CUBE_TILE_READY)
//   set FIX→M   (next mad_s4 may overwrite L0C)
//   set M→MTE1  (next iter may reuse this pingpong slot)
//
// Back-pressure: kPreloadNum = kRingSlots = 6 K-blocks fired without
// vec gate (slots empty); from kb >= 6 onwards each iter waits one
// VEC_TILE_CONSUMED before producing. Drain trailing kRingSlots
// VEC_TILE_CONSUMED signals on exit so the FFTS counter ends clean
// (vec signals 32 times total; cube only waited 32-6=26 times in the
// loop).
//
// Vec path (per K-block):
//   wait_flag_dev(CUBE_TILE_READY)
//   TLOAD partial_i32 from workspace[slot] (vecM rows, BN cols)
//   TLOAD ascale fp16 row from ascales[kb, m_off:m_off+vecM]
//   TLOAD wscale fp16 col from wscales[kb, :]
//   TCVT i32→f32 (partial), fp16→f32 (ascale, wscale)
//   TROWEXPANDMUL (apply ascale per row)
//   TCOLEXPANDMUL (apply wscale per col)
//   TADD into running_f32 accumulator (or TMOV if kb==0)
//   ffts_cross_core_sync(MTE2, VEC_TILE_CONSUMED)   ← free ring slot
// After last K-block:
//   TCVT running_f32 → running_f16
//   TSTORE → out_gm[m_off:m_off+vecM, :]
//
// The TASSIGN-on-data() bug in pto_macro_matmul.hpp doesn't affect us
// because we save the L1 base ptr before the K-loop and recompute the
// per-K-block view from base every iter. (Phase 2d hit Cube_K=Tile_K
// so its loop iterated only once and the bug was masked.)
//
// `__enable_feature_for_compile_default = KERNEL_TYPE_MIX_AIC_1_2`
// keeps the auto-gen wrapper in mix mode (1 cube : 2 vec).
//
// ─────────── Phase 3c-4 perf snapshot (msprof, 16 tiles, 910B3) ───────────
//
// Cube pipe ratios (median over 25 captured calls, --aic-metrics=PipeUtilization):
//   FIX (L0C→GM TSTORE drain):  49.6 %   ← dominant pipe
//   MAC (mad_s4):                6.7 %   ← cube math itself is light
//   MTE2 (GM→L1 TLOAD):          5.3 %
//   MTE1 (L1→L0A/B TEXTRACT):    4.4 %
//   bubble:                     ~34 %    (mostly cube ↔ vec back-pressure)
//
// Why FIX dominates — it's an architectural tax, not an algorithm bug.
// SVDQuant requires per-64-K-block ascale/wscale, so the math forces
// dequant inside the K-loop on every backend. On CUDA SMs that's free
// (Tensor Core mma writes to per-thread registers, the same warp's
// CUDA cores do the dequant inline — see nunchaku's apply_scales at
// gemm_base.cuh:367, no drain at all). On Ascend, cube and vec are
// physically separate hardware with no shared SRAM, so every fine-
// grained dequant must round-trip int32 partial L0C → GM → UB.
// 32× per tile × 128 KB = 4 MB / tile drained through FIX.
// Single-buf L0C additionally serializes MAC vs FIX, which makes the
// drain cost wallclock-visible rather than overlapped.
// See docs/npu.md § "Architectural tax" for the full cube/vec vs
// CUDA SM comparison table.
//
// Optimization opportunities (in `docs/npu.md` § "Perf"):
//   1. L0C ping-pong (BUF0 / BUF1) — MAC can run while FIX drains the
//      other buffer. Expected MAC ratio 6.7 % → ~30 %. Cube-side only.
//   2. Per-2-K-block drain batching in vec UB — halves FIX freq but
//      doubles UB scratch (already 135 / 184 KB used).
//   3. (Done — 3c-5) PTO-style variadic launch on host (kernel.cpp).
//      This was *not* an optimization — it repaired a launcher
//      regression from Phase 3a-5 that the PTO demos never had.
//      Phase 3a–3c-4 packed 11 device pointers into a host
//      DeviceParams struct + aclrtMalloc/Memcpy/Sync/Free per call;
//      PTO's INVOKE_PTO_KERNEL pattern uses variadic GM_ADDR args
//      directly. Device-side MFU unchanged after the fix.
// ──────────────────────────────────────────────────────────────────────

#include "kernel_operator.h"
#include <pto/pto-inst.hpp>

#include "pto_macro_matmul_s4.hpp"

constexpr KernelMetaType __enable_feature_for_compile_default = KERNEL_TYPE_MIX_AIC_1_2;

namespace {

// FFTS flag IDs — same enum as Phase 2a–2d so subsequent phases don't
// have to renumber. CUBE_TILE_READY / VEC_TILE_CONSUMED are now the
// per-K-block producer/consumer signals (each fires 32 times per
// launch, not 8 like Phase 2d).
enum GemmFftsFlag : uint16_t {
    HANDSHAKE_CUBE_TO_VEC = 0,
    HANDSHAKE_VEC_TO_CUBE = 1,
    CUBE_TILE_READY      = 2,
    VEC_TILE_CONSUMED    = 3,
    LORA_BUF_READY       = 4,
};

// Local stack-allocated typed view over the variadic GM_ADDR args at
// entry. Pre-3c-5 this was packed into a 96 B `DeviceParams` struct on
// host, H2D-copied to a freshly aclrtMalloc'd device staging slot per
// call, then read here through a single GM_ADDR. Per-call malloc +
// memcpy + sync + free added ~260 µs/call (msprof: 318 µs wall vs 57 µs
// device — see 3c-4 perf snapshot above and docs/npu.md). PTO's
// INVOKE_PTO_KERNEL (see pto-isa/demos/baseline/gemm_basic/.../utils.h)
// just passes the variadic args through ACLRT_LAUNCH_KERNEL — no
// packing. Mirroring that here.
//
// We still alias the variadic args into a typed struct because
// (a) ccec disallows casting `void* __gm__` to a typed `__gm__ T*`
// inline, so the typed casts have to happen once at entry and
// (b) the rest of the kernel body still reads `p->act`, `p->wgt`, …
// — only the entry path changes, not the data-flow downstream.
struct DeviceParams {
    __gm__ uint8_t*  act;         // [M_total, K/2]                  packed INT4
    __gm__ uint8_t*  wgt;         // [N, K/2]                        packed INT4 (shared)
    __gm__ half*     ascales;     // [K/64, M_total]                 fp16
    __gm__ uint64_t* wscales;     // [K/64, N]                       VDEQF16 packed (3c-7)
    __gm__ half*     la_fp16;     // [M_total, R]                    fp16
    __gm__ half*     lu_T;        // [R, N]                          fp16 (shared)
    __gm__ half*     bias;        // [N]                             fp16 (shared)
    __gm__ half*     wcscales;    // [N]                             fp16 (shared)
    __gm__ half*     workspace;   // [blockDim, kRingSlots, kBM, N]  fp16 cube/vec ring (3c-7)
    __gm__ float*    lora_buf;    // [M_total, N]                    fp32 LoRA-up hand-off
    __gm__ half*     out;         // [M_total, N]                    fp16 final
    uint64_t         m_total;     // total M rows = blockDim * kBM
};

// Tile shape constants — pinned for 3c-3 production single-tile.
// kNumKBlocks=32 ⇒ cube K-loop iterates 32 times per launch, vec drains
// 32 K-blocks through the kRingSlots=6 ring with steady-state back-
// pressure on VEC_TILE_CONSUMED. UB budget at this shape: kPartialOff
// + kRunningOff = 2 × kVecM × kBN × 4B = 128 KB (kVecM=64, kBN=256),
// plus ~7 KB for ascale/wscale/wcscale/bias tiles ⇒ ~135 KB / 184 KB
// mix-mode AIV cap.
constexpr uint32_t kBM         = 128;
constexpr uint32_t kBN         = 256;
constexpr uint32_t kBKLogical  = 2048;
constexpr uint32_t kBKPacked   = kBKLogical / 2;
constexpr uint32_t kKSLogical  = 64;                   // mad_s4 K-block / scale block
constexpr uint32_t kKSPacked   = kKSLogical / 2;       // 32 packed bytes
constexpr uint32_t kNumKBlocks = kBKLogical / kKSLogical;  // 32

// LoRA rank (production R ≤ 128). 32 is a real shipping point and keeps
// the LoRA-up cube pass a single mad (kBM × kR × kBN fp16, fp32 acc).
constexpr uint32_t kR          = 32;

// Cube-vec ring. kPreloadNum = kRingSlots so cube fills the ring once
// without back-pressure, then steady-state waits VEC_TILE_CONSUMED.
constexpr uint32_t kRingSlots  = 6;
constexpr uint32_t kPreloadNum = kRingSlots;

// Mix mode 1:2 — 1 cube + 2 vec subblocks per cluster.
constexpr uint16_t kAivPerAic = 2;
constexpr uint32_t kVecM      = kBM / kAivPerAic;      // 64 rows per AIV subblock

}  // namespace

extern "C" __global__ [aicore] void
svdquant_gemm_w4a4_kernel(GM_ADDR act_in,         GM_ADDR wgt_in,
                          GM_ADDR ascales_in,     GM_ADDR wscales_in,
                          GM_ADDR la_fp16_in,     GM_ADDR lu_T_in,
                          GM_ADDR bias_in,        GM_ADDR wcscales_in,
                          GM_ADDR workspace_in,   GM_ADDR lora_buf_in,
                          GM_ADDR out_in,
                          uint64_t m_total) {
    // Build the typed view on the stack — no longer comes from a
    // host-packed device staging buffer. Field order matches the
    // call-site arg order in kernel.cpp (and 1-to-1 with the legacy
    // DeviceParams layout, for review-friendliness).
    DeviceParams dp{
        (__gm__ uint8_t*) act_in,      (__gm__ uint8_t*)wgt_in,
        (__gm__ half*)    ascales_in,  (__gm__ uint64_t*)wscales_in,
        (__gm__ half*)    la_fp16_in,  (__gm__ half*)    lu_T_in,
        (__gm__ half*)    bias_in,     (__gm__ half*)    wcscales_in,
        (__gm__ half*)    workspace_in,(__gm__ float*)   lora_buf_in,
        (__gm__ half*)    out_in,
        m_total};
    const DeviceParams* p = &dp;

    if ASCEND_IS_AIC {
        // 3c-4 grid M-major: block_idx ∈ [0, blockDim) selects this core's
        // [block_idx*kBM, block_idx*kBM+kBM) row slice. Use CCE intrinsic
        // get_block_idx() (cluster idx; same value across cube + vec of the
        // same cluster — see vec-side note). wgt and wscales are shared
        // across all blocks (read-only L2).
        const int64_t block_idx = get_block_idx();
        auto* act_gm = p->act + (uint64_t)block_idx * kBM * kBKPacked;
        auto* wgt_gm = p->wgt;
        auto* ws_gm  = p->workspace
                     + (uint64_t)block_idx * kRingSlots * kBM * kBN;

        using TileMatA = pto::Tile<pto::TileType::Mat, int8_t, kBM, kBKPacked,
                                    pto::BLayout::ColMajor, kBM, kBKPacked,
                                    pto::SLayout::RowMajor, 512>;
        using TileMatB = pto::Tile<pto::TileType::Mat, int8_t, kBKPacked, kBN,
                                    pto::BLayout::RowMajor, kBKPacked, kBN,
                                    pto::SLayout::ColMajor, 512>;
        using TileAccC = pto::TileAcc<int32_t, kBM, kBN, kBM, kBN>;

        // 3c-7 VDEQF16 wscale staging — per-K-block uint64 deqscalar vector
        // staged GM → L1 (TileMat) → FBuffer (TileScaling), then consumed by
        // TSTORE_FP during the L0C drain. Layout mirrors the PTO testcase
        // `pto-isa/tests/.../tstore_acc2gm` (uint64 N-vector, NoneBox SLayout).
        // FBuffer is 2 KB on A2/A3; one wscale vector (256 × 8 B = 2 KB)
        // fills it exactly — no room for double-buffering.
        using TileMatWscale  = pto::Tile<pto::TileType::Mat, uint64_t, 1, kBN,
                                          pto::BLayout::RowMajor, 1, kBN,
                                          pto::SLayout::NoneBox>;
        using TileScalingWs  = pto::Tile<pto::TileType::Scaling, uint64_t,
                                          1, kBN, pto::BLayout::RowMajor,
                                          1, kBN, pto::SLayout::NoneBox>;

        using GlobalA  = pto::GlobalTensor<int8_t,
            pto::Shape<1, 1, 1, kBM, kBKPacked>,
            pto::Stride<1, 1, 1, kBKPacked, 1>>;
        using GlobalB  = pto::GlobalTensor<int8_t,
            pto::Shape<1, 1, 1, kBKPacked, kBN>,
            pto::Stride<1, 1, 1, 1, kBKPacked>,
            pto::Layout::DN>;
        // 3c-7: ring slot dtype int32 → half. FIX-pipe VDEQF16 mode applies
        // × wscale (per-N column) during the L0C drain and writes the
        // already-dequantized fp16 partial to GM. Halves both ring slot
        // bytes and vec's per-K-block MTE2 work.
        using GlobalRingSlot = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, kBM, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;
        // wscale_packed[K/64, N] in GM, sliced to one [1, N] vector per
        // K-block. uint64 per N column (VDEQF16 deqscalar — see
        // baseline/_int4.pack_wscales_vdeqf16).
        using GlobalWscalePacked = pto::GlobalTensor<uint64_t,
            pto::Shape<1, 1, 1, 1, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;

        // L1 layout: A at offset 0 (128 KB), B at offset 128 KB (256 KB),
        // wscale staging at 384 KB (1 × 256 × 8 = 2 KB). Total 386 KB / 512 KB.
        constexpr uint64_t kL1AByteOffset       = 0;
        constexpr uint64_t kL1BByteOffset       = (uint64_t)kBM * kBKPacked;
        constexpr uint64_t kL1WscaleByteOffset  = kL1BByteOffset
                                                + (uint64_t)kBKPacked * kBN;
        // FBuffer Scaling tile lives at offset 0 (whole FBuffer, 2 KB on A2/A3).
        constexpr uint64_t kFBufferWscaleOffset = 0;

        TileMatA       aMatTile;
        TileMatB       bMatTile;
        TileAccC       cAccTile;
        TileMatWscale  wscaleMatTile;
        TileScalingWs  wscaleScalingTile;
        TASSIGN(aMatTile,          kL1AByteOffset);
        TASSIGN(bMatTile,          kL1BByteOffset);
        TASSIGN(wscaleMatTile,     kL1WscaleByteOffset);
        TASSIGN(wscaleScalingTile, kFBufferWscaleOffset);
        // L0C 128 KB on A2/A3 fits a single [kBM, kBN] int32 = 128 KB exactly
        // (no room for ping-pong at this tile shape — see docs/gotchas/ascend.md
        // "L0C is 128 KB on A2/A3").
        TASSIGN(cAccTile, 0u);  // L0C BUF0

        GlobalA aGlobal((__gm__ int8_t*)act_gm);
        GlobalB bGlobal((__gm__ int8_t*)wgt_gm);

        // One-shot TLOAD of the full BK into L1.
        TLOAD(aMatTile, aGlobal);
        TLOAD(bMatTile, bGlobal);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        // Save L1 base addresses for K-block sliding. The macro's
        // `aMatTile.data()` will return the most-recently-TASSIGN'd
        // value, which compounds if we rely on it; recompute each
        // iter from a fixed base.
        const uint64_t kL1A_base = (uint64_t)aMatTile.data();
        const uint64_t kL1B_base = (uint64_t)bMatTile.data();

        // Seed cross-pipe flags.
        // - PIPE_M → PIPE_MTE1 (×2 events for L0 ping-pong): so the
        //   first wait_flag inside the loop is satisfied on iter 0.
        // - PIPE_FIX → PIPE_M: so the first mad_s4 may overwrite L0C
        //   without waiting on a non-existent prior TSTORE.
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

        // When the K-block count is smaller than the ring depth, back-
        // pressure in the loop never triggers. The drain loop below
        // must then consume exactly the produced count, not kPreloadNum.
        // (Same idiom as Phase 2d's kActualPreload — without it, kNumKBlocks
        // < kPreloadNum deadlocks waiting on signals vec never sends.)
        constexpr uint32_t kActualPreload =
            (kPreloadNum < kNumKBlocks) ? kPreloadNum : kNumKBlocks;

        for (uint32_t kb = 0; kb < kNumKBlocks; ++kb) {
            const uint64_t pingpong = kb & 1;
            (void)kb;  // slot intentionally unused — #127 probe skips TSTORE_FP

            // Back-pressure: after the ring's filled once, vec must
            // free a slot for each subsequent producer iter.
            if (kb >= kActualPreload) {
                wait_flag_dev(VEC_TILE_CONSUMED);
            }

            // Slide L1 view to the kb-th K-block.
            TASSIGN(aMatTile, kL1A_base + (uint64_t)kb * kKSPacked * kBM);
            TASSIGN(bMatTile, kL1B_base + (uint64_t)kb * kKSPacked * kBN);

            // 3c-7 PROBE #127: SKIP TLOAD wscale + TMOV(L1→FBuffer) +
            // TStoreAccFp. Goal: isolate whether 32× TStoreAccFp freezes
            // FIX-pipe such that post-K-loop TSTORE no-ops (#126 evidence:
            // both the drain probe AND LoRA TSTORE wrote nothing — bytes
            // in lora_buf stayed at at::zeros). If LoRA writes correctly
            // when K-loop never touches FIX-pipe / set_fpc / FBuffer,
            // the freeze hypothesis is confirmed. Main path will be zero
            // (workspace stays at::zeros), but we don't care — only the
            // LoRA TSTORE's success matters for this probe.
            //
            // Removed in this probe:
            //   - TLOAD(wscaleMatTile, ...)   (no wscale staging)
            //   - set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID2)
            //   - wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID2)
            //   - TMOV(wscaleScalingTile, wscaleMatTile)
            //   - TSTORE_FP(ringSlot, cAccTile, wscaleScalingTile)
            //
            // Kept:
            //   - mad_s4 (32×, init=true overwrite — exercises L0C the
            //     same way the real K-loop would; ensures L0C dtype
            //     tracking goes through identical mad_s4 sequence)
            //   - FFTS CUBE_TILE_READY signal so vec proceeds
            //   - L0A/B ping-pong sync (mad_s4 needs it)
            //
            // L0C cleanup: with no TSTORE draining L0C between mads, the
            // FIX→M event isn't actually signaled by hardware. We skip
            // its wait/set entirely.

            // Wait this ping-pong's L0A/B slot is free.
            wait_flag(PIPE_M, PIPE_MTE1, (event_t)pingpong);

            pto::pto_macro_matmul_s4_block<kBM, kBN, kKSLogical>(
                aMatTile, bMatTile, cAccTile, pingpong);

            // Signal vec without any FIX-pipe drain. The FFTS sync still
            // uses PIPE_FIX as its issuing pipe, but no payload op runs.
            ffts_cross_core_sync(PIPE_FIX, pto::getFFTSMsg(0x2, CUBE_TILE_READY));

            // This pingpong slot can be re-extracted into.
            set_flag(PIPE_M, PIPE_MTE1, (event_t)pingpong);
        }

        // Drain trailing per-pingpong ping-pong gates.
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        // #127 PROBE: skip FIX→M drain wait (no FIX op fired in K-loop).

        // Settle M-pipe before fp32 LoRA mad: main K-loop wrote int32 to
        // L0C via mad_s4, want clean state for fp32 mad.
        pipe_barrier(PIPE_M);

        // Drain trailing VEC_TILE_CONSUMED signals. Vec fires once per
        // K-block (kNumKBlocks total); cube consumed (kNumKBlocks - kActualPreload)
        // of them in the K-loop, leaving kActualPreload pending here.
        for (uint32_t i = 0; i < kActualPreload; ++i) {
            wait_flag_dev(VEC_TILE_CONSUMED);
        }

        // 3c-7 PROBE #127: NO FIX-pipe drains in K-loop, so no set_fpc /
        // FBuffer / TStoreAccFp ran. LoRA cube TSTORE follows on a "fresh"
        // FIX-pipe. If lora_buf comes back non-zero, FIX-pipe freeze after
        // 32× TStoreAccFp is confirmed (#126 evidence).

        // ===== LoRA-up cube pass =====
        // Single fp16×fp16 mad: la_fp16 [M, R] × lu_T [R, N] → fp32 acc → lora_buf [M, N].
        // Host must allocate lora_buf with at::zeros (NOT at::empty) — empty
        // leaves the GM line cold and cube fixpipe TSTORE doesn't reliably
        // land. See docs/gotchas/ascend.md "tensor.cpu() can return prior
        // fill" for the symptom story.
        constexpr uint64_t kL1LAOffset  = 16u * 1024;
        constexpr uint64_t kL1LUTOffset = kL1LAOffset + (uint64_t)kBM * kR * sizeof(half);

        // Both tiles use ND2NZ (BLayout=ColMajor + SLayout=RowMajor) because
        // the GM tensors for la_fp16 [M, R] and lu_T [R, N] are both row-major.
        using TileMatLA  = pto::Tile<pto::TileType::Mat, half, kBM, kR,
                                      pto::BLayout::ColMajor, kBM, kR,
                                      pto::SLayout::RowMajor, 512>;
        using TileMatLUT = pto::Tile<pto::TileType::Mat, half, kR, kBN,
                                      pto::BLayout::ColMajor, kR, kBN,
                                      pto::SLayout::RowMajor, 512>;
        using TileAccLora = pto::TileAcc<float, kBM, kBN, kBM, kBN>;
        using LeftTileLora  = pto::TileLeft<half, kBM, kR, kBM, kR>;
        using RightTileLora = pto::TileRight<half, kR, kBN, kR, kBN>;

        using GlobalLA = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, kBM, kR>,
            pto::Stride<1, 1, 1, kR, 1>>;
        using GlobalLUT = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, kR, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;
        // LoRA hand-off GM: separate fp32 [M, N] tensor (p->lora_buf).
        using GlobalLoraBuf = pto::GlobalTensor<float,
            pto::Shape<1, 1, 1, kBM, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;

        TileMatLA   laMatTile;
        TileMatLUT  lutMatTile;
        TileAccLora loraAccTile;
        TASSIGN(laMatTile,   kL1LAOffset);
        TASSIGN(lutMatTile,  kL1LUTOffset);
        TASSIGN(loraAccTile, 0u);  // L0C BUF0 (after main K-loop drain)

        // 3c-4: la_fp16 sliced per-block (each block has its own [kBM, R]
        // strip), lu_T is shared across blocks.
        GlobalLA  laGlobal((__gm__ half*)p->la_fp16 + (uint64_t)block_idx * kBM * kR);
        GlobalLUT lutGlobal((__gm__ half*)p->lu_T);

        TLOAD(laMatTile, laGlobal);
        TLOAD(lutMatTile, lutGlobal);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);

        LeftTileLora  aLoraL0;
        RightTileLora bLoraL0;
        TASSIGN(aLoraL0, 0u);  // L0A BUF0
        TASSIGN(bLoraL0, 0u);  // L0B BUF0

        TEXTRACT(aLoraL0, laMatTile,  0, 0);
        TEXTRACT(bLoraL0, lutMatTile, 0, 0);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);

        TMATMUL(loraAccTile, aLoraL0, bLoraL0);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);

        // TSTORE L0C fp32 accumulator → this block's lora_buf [kBM, N] slice.
        GlobalLoraBuf loraBufGm((__gm__ float*)p->lora_buf
                                + (uint64_t)block_idx * kBM * kBN);
        TSTORE(loraBufGm, loraAccTile);

        // Signal vec that lora_buf holds the fp32 LoRA result.
        ffts_cross_core_sync(PIPE_FIX,
                              pto::getFFTSMsg(0x2, LORA_BUF_READY));
    }

    if ASCEND_IS_AIV {
        // PTO mix-mode AIV vector mask is NOT in a known reset state at
        // entry — TROWEXPANDMUL/TCOLEXPANDMUL/TRowMin etc. internally set
        // a per-line mask (e.g. `set_vector_mask(0, elementsPerLine)`)
        // and don't always restore it before the next op. Whatever
        // residue was left by a previous ASCEND_IS_AIV invocation, or
        // even by hardware power-on default, is what subsequent
        // TLOAD/TCVT/TADD pick up — which, when wrong, makes those ops
        // address a UB region they shouldn't, manifesting as either
        // VEC ub-out-of-bounds (subErrType:4) or, worse, silently-wrong
        // values that decode as ±inf/NaN if the mask happens to make
        // the load land somewhere accessible. PTO's own TColReduceOps.hpp:31
        // and TRowMin.hpp:94 follow exactly this idiom for the same
        // reason; reference: gitcode.com/cann/pto-isa/issues/218 (a vec
        // OOB with byte-identical signature was solved by zhangjian_hz11
        // adding precisely this two-line reset).
        // Both ops are CCE intrinsics (kernel_operator headers); on
        // dav_c220 they emit a single-instruction state write. Cost is
        // negligible (~2 cycles total).
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
        set_mask_norm();
        set_vector_mask(-1, -1);
#endif

        // 3c-4 grid M-major: same cluster index as cube. NOTE — must use the
        // CCE intrinsic `get_block_idx()` (returns cluster idx 0..N-1), NOT
        // `AscendC::GetBlockIdx()`. On dav_c220 the AscendC wrapper returns
        // `block_idx*g_taskRation + subblockid` (0..2N-1) for AIV, which would
        // mismatch cube's view of the same cluster. (See
        // /usr/local/Ascend/.../dav_c220/kernel_operator_common_impl.h:48-65.)
        const int64_t block_idx = get_block_idx();
        auto* ws_gm   = p->workspace
                      + (uint64_t)block_idx * kRingSlots * kBM * kBN;
        auto* as_gm   = p->ascales;       // base; row stride is p->m_total
        auto* out_gm  = p->out + (uint64_t)block_idx * kBM * kBN;
        const uint64_t m_total_rows = p->m_total;

        const uint32_t subblockid = get_subblockid();
        const uint32_t row_off    = kVecM * subblockid;  // 0 or 64

        // UB layout (per AIV subblock; TOTAL_VEC_LOCAL_SIZE = 184 KB on
        // dav_c220 mix mode). 3c-7: ring slot is fp16 (× wscale already
        // folded by cube FIX-pipe VDEQF16), so vec drops the wscale tile
        // and TCOLEXPANDMUL.
        //   partial_f32                shared @ 0           = vecM*BN*4 = 64 KB
        //                              (also LoRA tile post-loop, partial dead)
        //   running_f32                64 KB                = 64 KB
        //   partial_f16                128 KB               = vecM*BN*2 = 32 KB
        //                              (also out_f16 post-loop, partial dead)
        //   ascale_f16/_f32/_bcast     ~2.4 KB after that
        //   wcscale_f16/_f32           ~1.5 KB
        //   bias_f16/_f32              ~1.5 KB
        //   ─────────────────────────────────────────
        //   ≈ 165 KB / 184 KB cap, ~19 KB headroom.
        //
        // partial_f16 → partial_f32 TCVT writes to a *different* offset
        // (kPartialOff vs kPartialF16Off) — fp16 read followed by widened
        // fp32 write can't safely alias the same start address (different
        // element strides), so we keep them disjoint.
        //
        // Block size for vbrcb broadcast (32-byte block / sizeof(fp32) = 8).
        // TROWEXPAND([1, M] RowMajor → [M, 8] RowMajor) requires dst::Cols
        // == elemPerBlock = 8 (see pto::TROWEXPAND_IMPL isBroadcast check).
        constexpr uint32_t kBcastCols = 8;

        constexpr uint32_t kPartialOff       = 0;
        constexpr uint32_t kRunningOff       = kPartialOff + kVecM * kBN * 4;
        constexpr uint32_t kPartialF16Off    = kRunningOff + kVecM * kBN * 4;
        // ascale = per-row M scale. Loaded RowMajor [1, vecM] half (mirror
        // of wscale's known-working pattern), TCVT to RowMajor [1, vecM]
        // fp32, then expanded by pto::TROWEXPAND to RowMajor [vecM, 8]
        // broadcast tile (row r = [s_r] × 8). The broadcast tile is what
        // TROWEXPANDMUL consumes — feeding it as RowMajor src1 takes the
        // RowMajor code path that skips PTO's internal vbrcb scratch.
        // (3a-cycle-15 root cause: ColMajor [vecM, 1] TLOAD from GM only
        // loads the head element; switching to this load-row + expand
        // pattern bypasses the bug entirely. See memory note
        // pto_colmajor_n1_tload_broken.md.)
        constexpr uint32_t kAscaleF16Off    = kPartialF16Off + kVecM * kBN * 2;
        constexpr uint32_t kAscaleF32Off    = kAscaleF16Off + kVecM * 2;
        constexpr uint32_t kAscaleBcastOff  = kAscaleF32Off + kVecM * 4;
        // 3c-1 per-channel affine (epilogue-only): wcscales × running + bias.
        // [1, kBN] fp16/f32 Tile pattern.
        constexpr uint32_t kWcscaleF16Off   = kAscaleBcastOff + kVecM * kBcastCols * 4;
        constexpr uint32_t kWcscaleF32Off   = kWcscaleF16Off + kBN * 2;
        constexpr uint32_t kBiasF16Off      = kWcscaleF32Off + kBN * 4;
        constexpr uint32_t kBiasF32Off      = kBiasF16Off    + kBN * 2;
        constexpr uint32_t kOutF16Off       = kPartialF16Off;  // overlap post-loop

        // 3c-7: ring slot dtype is fp16 (× wscale already applied by cube
        // FIX-pipe VDEQF16). Vec loads fp16, casts to fp32, multiplies by
        // ascale (per-row), accumulates. No wscale tile or TCOLEXPANDMUL.
        using TilePartialF16 = pto::Tile<pto::TileType::Vec, half, kVecM, kBN,
                                          pto::BLayout::RowMajor, kVecM, kBN>;
        using TilePartialF32 = pto::Tile<pto::TileType::Vec, float, kVecM, kBN,
                                          pto::BLayout::RowMajor, kVecM, kBN>;
        using TileRunningF32 = pto::Tile<pto::TileType::Vec, float, kVecM, kBN,
                                          pto::BLayout::RowMajor, kVecM, kBN>;
        using TileOutF16     = pto::Tile<pto::TileType::Vec, half, kVecM, kBN,
                                          pto::BLayout::RowMajor, kVecM, kBN>;

        // ascale RowMajor row tiles: 32 contiguous halfs → 32 contiguous
        // fp32s after TCVT. Then TROWEXPAND broadcasts each scalar into a
        // 32-byte block (= 8 fp32) along the row axis, producing the
        // [vecM, 8] tile that TROWEXPANDMUL takes as RowMajor src1
        // (without invoking internal vbrcb). See gotchas/ascend.md
        // "ColMajor [N,1] TLOAD broken" for why we load as a row.
        using TileAscaleF16    = pto::Tile<pto::TileType::Vec, half,  1, kVecM,
                                            pto::BLayout::RowMajor, 1, kVecM>;
        using TileAscaleF32    = pto::Tile<pto::TileType::Vec, float, 1, kVecM,
                                            pto::BLayout::RowMajor, 1, kVecM>;
        using TileAscaleBcastF32 = pto::Tile<pto::TileType::Vec, float, kVecM, kBcastCols,
                                              pto::BLayout::RowMajor, kVecM, kBcastCols>;
        // [1, kBN] fp16/f32 — used for wcscale and bias epilogue tiles.
        using TileColRowF16 = pto::Tile<pto::TileType::Vec, half, 1, kBN,
                                         pto::BLayout::RowMajor, 1, kBN>;
        using TileColRowF32 = pto::Tile<pto::TileType::Vec, float, 1, kBN,
                                         pto::BLayout::RowMajor, 1, kBN>;

        // 3c-7: ring slot is now fp16, not int32.
        using GlobalRingSlot = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, kVecM, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;
        using GlobalAscaleRow = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, 1, kVecM>,
            pto::Stride<1, 1, 1, kVecM, 1>>;
        // GM strip for [1, kBN] half (wcscale, bias loaders).
        using GlobalColRow = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, 1, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;
        using GlobalOutTile  = pto::GlobalTensor<half,
            pto::Shape<1, 1, 1, kVecM, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;

        // Cross-iter UB sync — running_f32 reused across K-blocks.
        // Seed MTE3→MTE2 so first iter's TLOAD doesn't race a non-
        // existent prior TSTORE (drained at the end).
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        // Cross-iter V→MTE2: partial_i32/f32 UB region is reused every
        // iter; without this sync, iter N+1's TLOAD can race iter N's
        // PIPE_V writes (TROWEXPANDMUL/TCOLEXPANDMUL/TMOV land after
        // the load, corrupting the freshly-loaded int32 bytes). See
        // docs/gotchas/ascend.md "AIV K-loop reusing partial UB".
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);

        for (uint32_t kb = 0; kb < kNumKBlocks; ++kb) {
            wait_flag_dev(CUBE_TILE_READY);
            // Wait prior iter's PIPE_V (TCVT/TROWEXPANDMUL/TADD) to drain
            // before this iter's TLOAD overwrites partial UB.
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            const uint32_t slot = kb % kRingSlots;

            // GM offsets:
            //   ring slot row band (this AIV subblock owns rows
            //     [row_off, row_off + kVecM)) into workspace[slot]
            //     (ws_gm already pre-offset to this block's ring base)
            //   ascales[kb, block_idx*kBM + row_off : block_idx*kBM + row_off + kVecM]
            //     — row stride is m_total_rows, not kBM (multi-block layout)
            //   3c-7: wscale is no longer loaded by vec (cube FIX-pipe
            //     folded × wscale into the drain via VDEQF16).
            //   out[row_off:row_off+kVecM, :]    (out_gm already pre-offset)
            const uint64_t partial_off =
                (uint64_t)slot * kBM * kBN + (uint64_t)row_off * kBN;
            const uint64_t ascale_off  =
                (uint64_t)kb * m_total_rows
                + (uint64_t)block_idx * kBM + row_off;

            TilePartialF16      partF16;
            TilePartialF32      partF32;
            TileRunningF32      running;
            TileAscaleF16       ascaleF16;
            TileAscaleF32       ascaleF32;
            TileAscaleBcastF32  ascaleBcast;
            TASSIGN(partF16,     kPartialF16Off);
            TASSIGN(partF32,     kPartialOff);
            TASSIGN(running,     kRunningOff);
            TASSIGN(ascaleF16,   kAscaleF16Off);
            TASSIGN(ascaleF32,   kAscaleF32Off);
            TASSIGN(ascaleBcast, kAscaleBcastOff);

            GlobalRingSlot  partGm  (ws_gm      + partial_off);
            GlobalAscaleRow ascaleGm(p->ascales + ascale_off);

            // Wait running_f32 region is free (prev TSTORE on out
            // tile finished, except on the first iter where this
            // is the seed).
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

            TLOAD(partF16,   partGm);
            TLOAD(ascaleF16, ascaleGm);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // Cast f16 → f32 (partial slot already × wscale from cube FIX),
            // f16 → f32 (ascale).
            pto::TCVT(partF32,   partF16,   pto::RoundMode::CAST_RINT);
            pto::TCVT(ascaleF32, ascaleF16, pto::RoundMode::CAST_RINT);

            // 3b-6m: restore the V-pipe drain that was implicit in the 3a
            // debug TSTOREs (bisect-3 forced V→MTE3→V around partF32). Without
            // it, vec K-loop output is 2.578 off — see #110.
            pipe_barrier(PIPE_V);

            // Expand ascaleF32 [1, vecM] RowMajor → ascaleBcast [vecM, 8]
            // RowMajor where row r = [s_r] × 8 (see RowMajor src1 rationale
            // in the pre-3c-7 comment block above this loop).
            pto::TROWEXPAND(ascaleBcast, ascaleF32);

            // Defensive: TROWEXPAND's internal vbrcb may leave the mask
            // register in a non-default state; TROWEXPANDMUL's RowMajor
            // path's NormModeTail else-branch does NOT call SetContMask
            // before vmul (it inherits the caller's mask). Cycle 16 Run H
            // showed first 4 rows per AIV silently skipped — symptom
            // consistent with stale mask. Reset to norm + full vec mask
            // before TROWEXPANDMUL to make the mask state explicit.
            pipe_barrier(PIPE_V);
            set_mask_norm();
            set_vector_mask(-1, -1);

            // partF32[m,n] *= ascaleF32[m]  (× wscale already in the slot)
            pto::TROWEXPANDMUL(partF32, partF32, ascaleBcast);

            // Accumulate into running_f32. On kb==0 there's no
            // prior value, so initialize via TMOV; subsequent iters
            // TADD.
            if (kb == 0) {
                pto::TMOV(running, partF32);
            } else {
                pto::TADD(running, running, partF32);
            }

            // Signal V→MTE2 for the next iter so its TLOAD won't overlap
            // with this iter's PIPE_V writes to partF32. See seed comment.
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);

            // Free the ring slot for the next cube K-block.
            ffts_cross_core_sync(PIPE_MTE2,
                                  pto::getFFTSMsg(0x2, VEC_TILE_CONSUMED));

            // Re-seed MTE3→MTE2 for the next iter's TLOAD region
            // dependency. (running and out_f16 occupy disjoint UB
            // regions, so this is mostly a tail flag bookkeeping
            // action — but consistent with Phase 2d's pattern.)
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }

        // Drain trailing seed before LoRA-add + final cast+store.
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);

        // ===== LoRA-up residual: running += lora_buf[row_off:row_off+vecM, :] =====
        // Reuse kPartialOff for the LoRA tile UB region — partial_i32/f32 is
        // dead after the K-loop.
        using TileLoraF32 = pto::Tile<pto::TileType::Vec, float, kVecM, kBN,
                                      pto::BLayout::RowMajor, kVecM, kBN>;
        using GlobalLoraSlice = pto::GlobalTensor<float,
            pto::Shape<1, 1, 1, kVecM, kBN>,
            pto::Stride<1, 1, 1, kBN, 1>>;

        wait_flag_dev(LORA_BUF_READY);

        TileLoraF32 loraTile;
        TASSIGN(loraTile, kPartialOff);

        // lora_buf laid out [M_total, kBN]; this block's slice starts at
        // block_idx*kBM, this AIV subblock's strip within that is row_off.
        const uint64_t lora_off =
            (uint64_t)block_idx * kBM * kBN + (uint64_t)row_off * kBN;
        GlobalLoraSlice loraGm((__gm__ float*)p->lora_buf + lora_off);
        TLOAD(loraTile, loraGm);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

        TileRunningF32 runningForAdd;
        TASSIGN(runningForAdd, kRunningOff);
        pto::TADD(runningForAdd, runningForAdd, loraTile);

        // ===== 3c-1 per-channel affine: running = running·wcscales + bias =====
        // Reuse Wscale fp16/f32 Tile shapes ([1, kBN]). wcscales and bias
        // are tiny ([N] = kBN halfs each, ~256 B) and live in their own UB
        // offsets appended after kWscaleF32Off. TCOLEXPANDMUL broadcasts
        // [1, kBN] across vecM rows (same primitive used per-K-block for
        // wscale); TCOLEXPANDADD is the additive twin.
        //
        // pipe_barrier(PIPE_V) around the COL-expand pair — same internal
        // sub-pipe race observed in the K-loop (#110 fix). Cheap insurance,
        // and the V-side is already idle waiting on MTE2.
        TileColRowF16 wcscaleF16;
        TileColRowF32 wcscaleF32;
        TileColRowF16 biasF16;
        TileColRowF32 biasF32;
        TASSIGN(wcscaleF16, kWcscaleF16Off);
        TASSIGN(wcscaleF32, kWcscaleF32Off);
        TASSIGN(biasF16,    kBiasF16Off);
        TASSIGN(biasF32,    kBiasF32Off);

        GlobalColRow wcscaleGm((__gm__ half*)p->wcscales);
        GlobalColRow biasGm   ((__gm__ half*)p->bias);
        TLOAD(wcscaleF16, wcscaleGm);
        TLOAD(biasF16,    biasGm);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

        pto::TCVT(wcscaleF32, wcscaleF16, pto::RoundMode::CAST_RINT);
        pto::TCVT(biasF32,    biasF16,    pto::RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);

        pto::TCOLEXPANDMUL(runningForAdd, runningForAdd, wcscaleF32);
        pipe_barrier(PIPE_V);
        pto::TCOLEXPANDADD(runningForAdd, runningForAdd, biasF32);
        pipe_barrier(PIPE_V);

        // Final epilogue: f32 → fp16 then TSTORE the AIV's row band.
        TileRunningF32 runningFinal;
        TileOutF16     outF16Final;
        TASSIGN(runningFinal, kRunningOff);
        TASSIGN(outF16Final,  kOutF16Off);

        pto::TCVT(outF16Final, runningFinal, pto::RoundMode::CAST_RINT);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        const uint64_t out_off = (uint64_t)row_off * kBN;
        GlobalOutTile outGm(out_gm + out_off);
        TSTORE(outGm, outF16Final);
    }
}

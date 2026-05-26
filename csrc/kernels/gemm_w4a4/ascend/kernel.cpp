// Host launcher for the Ascend gemm_w4a4 pod.
//
// Phase 3c-4 signature: 11 × device pointer + m_total + opaque stream.
// Multi-tile launch: each AI core (cluster) processes one [kBM, N, K]
// tile of out, indexed along M. blockDim = m_total / kBM. Inputs grow
// to fit M_total rows; wgt + wscales + lu_T + bias + wcscales are
// shared across blocks; act + ascales + lora_act_in + lora_buf + out
// + workspace are sliced per block.
//
// The launcher repacks all device pointers + m_total into a single
// device-side `DeviceParams` struct (96 B) and H2D-copies it to a
// small staging allocation, because the auto-gen `aclrtlaunch_*`
// wrapper takes a single `GM_ADDR params_addr` (it doesn't expand
// variadic tensor args).
//
// Synchronization: this launcher synchronizes after `aclrtlaunch_*`
// before freeing the staging `dev_params` to avoid use-after-free.
// Phase 3c-5+ will revisit per-call aclrtMalloc by using a stream-
// resident pre-allocated params buffer.

#include "gemm_w4a4.h"

#include <acl/acl.h>
#include "aclrtlaunch_svdquant_gemm_w4a4_kernel.h"

namespace svdquant::ascend {

namespace {

// kBM mirrors device-side constexpr. Used to compute blockDim from
// caller-supplied m_total. If you change kBM in kernel_device.cpp,
// change it here too — keep the two in sync.
constexpr uint64_t kBM = 128;

// Mirrors the device-side `DeviceParams` in kernel_device.cpp by byte
// layout: 11 × 8 B device pointers + 1 × 8 B u64 = 96 B. Host and
// device structs cannot share a header because the device file is
// `[aicore]` and ccec rejects dereferencing `void* __gm__` as a typed
// `__gm__ T*`. Keep the two field lists in sync.
struct DeviceParams {
    void* act;         // [M_total, K/2]                  uint8
    void* wgt;         // [N, K/2]                        uint8
    void* ascales;     // [K/64, M_total]                 fp16
    void* wscales;     // [K/64, N]                       fp16
    void* lora_act_in; // [M_total, R]                    fp32
    void* lora_up;     // [N, R]                          fp16
    void* bias;        // [N]                             fp16
    void* wcscales;    // [N]                             fp16
    void* workspace;   // [blockDim, kRingSlots, kBM, N]  int32 cube/vec ring (per-block)
    void* lora_buf;    // [M_total, N]                    fp32 LoRA hand-off
    void* out;         // [M_total, N]                    fp16 final
    uint64_t m_total;  // = blockDim * kBM (needed for ascale row stride)
};

}  // namespace

void gemm_w4a4(void* act, void* wgt,
               void* ascales, void* wscales,
               void* lora_act_in, void* lora_up,
               void* bias, void* wcscales,
               void* workspace, void* lora_buf, void* out,
               uint64_t m_total,
               void* stream) {
    auto raw_stream = static_cast<aclrtStream>(stream);

    DeviceParams dp{act, wgt, ascales, wscales,
                    lora_act_in, lora_up,
                    bias, wcscales,
                    workspace, lora_buf, out,
                    m_total};

    void* dev_params = nullptr;
    if (aclrtMalloc(&dev_params, sizeof(DeviceParams),
                    ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return;
    }
    if (aclrtMemcpy(dev_params, sizeof(DeviceParams),
                    &dp, sizeof(DeviceParams),
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(dev_params);
        return;
    }

    // Grid M-major: blockDim AI cores, each processes one [kBM, N, K]
    // tile of out. Mix 1:2 → each block has 1 cube + 2 vec subblocks.
    const uint32_t blockDim = static_cast<uint32_t>(m_total / kBM);
    aclrtlaunch_svdquant_gemm_w4a4_kernel(blockDim, raw_stream, dev_params);

    aclrtSynchronizeStream(raw_stream);
    aclrtFree(dev_params);
}

}  // namespace svdquant::ascend

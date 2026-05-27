# `space/` — Ascend 910B remote deployment payload

This directory holds the **pre-built aarch64 object files** that the
remote 910B box (GitCode WebIDE or, historically, GitCode Gradio Space)
consumes to link the torch op extension. Builds happen on the local
x86_64 dev box via aarch64 cross-compilation; the remote only links.

**Since Phase 3b (commit `7041864`, 2026-05-12), `space/objects/*.o` is
carried on `main`.** Earlier the .o files lived only on the
`gitcode-space` branch (a Gradio app deployment payload); that branch is
now historical, since the WebIDE flow + torch op extension replaced the
standalone Gradio smoke driver.

## Why cross-compile locally instead of building on the remote

The 910B WebIDE shell IS aarch64 native, so it *can* build with
`./scripts/build.sh` directly. But:

- The remote build pulls a few hundred MB of headers + ccec pipeline
  through every iteration — burns Space credit and takes minutes.
- `scripts/build.sh` hardcodes x86 CANN paths in its `env -i` block and
  requires `SVDQUANT_BUILD_CLEAN_ENV=1` + `pip install cmake>=3.22 ninja`
  + `git submodule update --init pto-isa` to work on a fresh Space.
- The link step (`bash space/link_op_extension.sh`) finishes in
  seconds, given the pre-built `.o`.

So local cross-build is the standard flow. WebIDE-native build is the
fallback when cross-toolchain isn't set up, or when a single iteration
on the Space side is faster than the local round-trip.

## Layout

```
app.py                         Gradio frontend (historical — for the
                               gitcode-space branch only; not used
                               by the WebIDE / torch op extension flow)
space/
  link_op_extension.sh         runs ON Space: links host_stub.cpp.o +
                               kernel.cpp.o + csrc/python/host/
                               svdquant_w4a4_op.cpp into the
                               aarch64 libop_extension.so loaded by
                               torch.ops.svdquant.gemm_w4a4
  link_smoke.sh                historical — Gradio Space smoke driver
                               link (kept for the gitcode-space branch)
  smoke_main.cpp               historical — same as above
  objects/
    host_stub.cpp.o            aarch64 -fPIC; CANN auto-gen kernel
                               registry stub WITH the device blob
                               injected via objcopy --update-section
    kernel.cpp.o               aarch64 -fPIC; svdquant::ascend::
                               gemm_w4a4 host launcher
    blob.bin                   raw NPU code blob, kept for re-injection
                               if host_stub is regenerated. NOT
                               referenced by link_op_extension.sh —
                               redundant with the section already
                               embedded in host_stub.cpp.o.
    smoke_main.cpp.o           historical — gitcode-space branch only
```

## Local rebuild recipe (when the kernel source changes)

Prereqs: `aarch64-linux-gnu-g++` + `aarch64-linux-gnu-objcopy` on PATH
(install via `apt-get install g++-aarch64-linux-gnu` if missing). CANN
toolkit at `/usr/local/Ascend/ascend-toolkit/latest/` (the
`CANN=8.5.x` install with `include/` arch-agnostic headers).

```bash
# 1. x86_64 build — produces the NPU device blob (embedded in the
#    x86 host_stub.cpp.o) + the auto-gen host_stub.cpp source.
./scripts/build.sh CUDA=OFF ASCEND=ON

# 2. Cross-build the 3 deliverables.
CANN_INC=/usr/local/Ascend/ascend-toolkit/latest/include
X86_HOST_STUB=build/csrc/kernels/CMakeFiles/svdquant_gemm_w4a4_device_host_stub_obj.dir/__/__/auto_gen/svdquant_gemm_w4a4_device/host_stub.cpp.o

# 2a. Extract the NPU device blob from the x86-built host_stub.cpp.o
#     (ascendc_pack_kernel injects it during the x86 build).
objcopy -O binary \
    --only-section=.ascend.kernel.ascend910b1.svdquant_gemm_w4a4_device \
    "$X86_HOST_STUB" \
    space/objects/blob.bin

# 2b. Cross-compile host_stub.cpp (auto-gen source), then re-inject the
#     blob. -fPIC is required: link_op_extension.sh links into a shared
#     object (libop_extension.so), and aarch64 ld rejects
#     R_AARCH64_ADR_PREL_PG_HI21 relocations from non-PIC code.
aarch64-linux-gnu-g++ -O2 -std=c++17 -fPIC -I"$CANN_INC" \
    -c build/auto_gen/svdquant_gemm_w4a4_device/host_stub.cpp \
    -o space/objects/host_stub.cpp.o
aarch64-linux-gnu-objcopy \
    --update-section .ascend.kernel.ascend910b1.svdquant_gemm_w4a4_device=space/objects/blob.bin \
    space/objects/host_stub.cpp.o

# 2c. Cross-compile the host launcher (also -fPIC, same reason).
aarch64-linux-gnu-g++ -O2 -std=c++17 -fPIC \
    -I"$CANN_INC" \
    -I csrc/kernels/gemm_w4a4/include \
    -I csrc/common/include \
    -I build/include/svdquant_gemm_w4a4_device \
    -c csrc/kernels/gemm_w4a4/ascend/kernel.cpp \
    -o space/objects/kernel.cpp.o

# 3. Verify and commit.
file space/objects/host_stub.cpp.o space/objects/kernel.cpp.o
#   both should report: ELF 64-bit LSB relocatable, ARM aarch64
git add space/objects/host_stub.cpp.o space/objects/kernel.cpp.o
git commit -m "rebuild aarch64 space/objects/ for <change>"
git push  # main on both gitcode and origin
```

After push, the remote (WebIDE on 910B3) pulls and runs
`bash space/link_op_extension.sh && python tests/test_gemm_w4a4.py`.
No remote build required.

## WebIDE-native fallback (emergency only)

When the local cross-toolchain isn't available (or the remote has the
fresher kernel changes uncommitted), build directly on the Space:

```bash
# Once per Space session (resets if container restarts):
pip install --user 'cmake>=3.22' ninja
git submodule update --init pto-isa
source scripts/env_ascend.sh
export PATH=$HOME/.local/bin:$PATH

# Then:
SVDQUANT_BUILD_CLEAN_ENV=1 ./scripts/build.sh
cp build/csrc/kernels/CMakeFiles/svdquant_gemm_w4a4_device_host_stub_obj.dir/__/__/auto_gen/svdquant_gemm_w4a4_device/host_stub.cpp.o space/objects/
cp build/csrc/kernels/CMakeFiles/svdquant_gemm_w4a4.dir/gemm_w4a4/ascend/kernel.cpp.o space/objects/
bash space/link_op_extension.sh
python tests/test_gemm_w4a4.py
```

This is what was done during the 2026-05-28 post-mortem verify
(commits `c9aa69d`, `fbf14f3`). Works but is non-standard.

## What's NOT shipped

`libascendc_runtime.a` (aarch64) and the rest of CANN's aarch64 libs —
we don't have them on the local x86_64 dev box, and they'd be ~hundreds
of MB. `link_op_extension.sh` resolves them from the remote 910B box's
native CANN install at link time.

# solidcv Apple Silicon & Metal 3.1 / 4.0 Kernel Tuning Set (M4 Max, M5 Pro, M5 Max)

**Author:** Andy Norris (`andy@solidsf.com`), Founder & CEO, solidSF  
**Repository:** `/Users/stu/git/solidcv`  
**Target Architectures:** Apple M4 Max (`arm64-apple-macosx`, Metal 3.1/4.0), Apple M5 Pro (`Metal 4` + Neural Accelerators), Apple M5 Max (`Metal 4` + Neural Accelerators)

---

## 1. Instruction File Set Overview

Give this folder (`docs/apple-metal-m4-m5/`) directly to an agent or operator on any **Apple M4 Max**, **Apple M5 Pro**, or **Apple M5 Max** machine to configure, verify, tune, and benchmark `solidcv`:

| File | Target Machine / Scope | Purpose |
| :--- | :--- | :--- |
| `00-INDEX-APPLE-SILICON-METAL-TUNING.md` | All Apple Silicon Macs (`M4 Max`, `M5 Pro`, `M5 Max`) | Master hardware topology comparison, environment controls, and quick-start verification |
| `01-M4-MAX-40GPU-TUNING-GUIDE.md` | **Apple M4 Max** (`stus-mac-studio`, 12P+4E, 40 GPU cores, 546 GB/s UMA) | M4 Max tuning parameters and verified 14-stage before/after benchmark (`12.04x` -> `34.70x` geo-mean vs OpenCV 5.0) |
| `02-M5-PRO-TUNING-INSTRUCTIONS.md` | **Apple M5 / M5 Pro** (`m5pro` `Mac17,2`, 4-12 Super/P-cores, 10-24 GPU cores, Metal 4) | Live `m5pro` (`AGXMetalG17G`) telemetry, L1/L2 tile sizing, Super-core pool rules, and verified 14-stage benchmark (`35.26x - 35.75x` geo-mean) |
| `03-M5-MAX-TUNING-INSTRUCTIONS.md` | **Apple M5 Max** (`andym5max`, 12-16P+4E, 40-48 GPU cores, 614 GB/s UMA) | Drop-in agent prompt, 32 MB L2 cache blocking, 40-48 GPU core zero-copy dispatch, and Metal 4 tuning for M5 Max |
| `04-KERNEL-TILING-FUSION-MULTIPROC-SPEC.md` | Kernel Engineering Reference | Micro-architectural specification for L1/L2 tiling, NEON register blocking, kernel fusion, QoS P-core multiprocessing, zero-copy Metal `NoCopy` UMA slabs, and vault re-sealing |

---

## 2. Hardware Topology Matrix (M4 Max vs. M5 / M5 Pro vs. M5 Max)

`solidcv::hardware::AppleSiliconChipProfile::detect()` automatically queries Darwin `sysctl` (`machdep.cpu.brand_string`, `hw.perflevel0.logicalcpu`, `hw.perflevel0.l1dcachesize`, `hw.perflevel0.l2cachesize`, `hw.cachelinesize`, `hw.memsize`) at startup and selects the exact micro-architectural parameters below:

| Parameter | Apple M4 Max (`stus-mac-studio`) | Apple M5 / M5 Pro (`m5pro` `Mac17,2`) | Apple M5 Max (`andym5max`) |
| :--- | :--- | :--- | :--- |
| `AppleSiliconGen` Variant | `AppleM4Max` | `AppleM5Pro` | `AppleM5Max` |
| CPU Topology (`perflevel0` + `perflevel1`) | `12 P-cores + 4 E-cores` | `4 Super + 6 E-cores` (`M5`) / `10-12P + 4E` (`M5 Pro`) | `12-16 Super/P-cores + 4 E-cores` |
| GPU Cores | `40` (`AGXMetalG16X`) | `10` (`M5` `AGXMetalG17G`) / `20-24` (`M5 Pro`) | `40-48` (`AGXMetalG17X`) |
| Metal 4 GPU Neural Accelerators | No (AMX + SIMD-group matrix) | Yes (Per-GPU-core Neural Accel, Metal 4) | Yes (Per-GPU-core Neural Accel, Metal 4) |
| Cache Line Size (`hw.cachelinesize`) | `128 bytes` | `128 bytes` | `128 bytes` |
| P/Super-Core L1 Data Cache (`l1dcachesize`) | `128 KB` per P-core | `128 KB` per Super/P-core (`192 KB` L1I) | `128 KB` per Super/P-core (`192 KB` L1I) |
| P/Super-Cluster Shared L2 (`l2cachesize`) | `16 MB` per P-cluster | `16 MB` (`M5`) / `24 MB` (`M5 Pro`) | `32 MB` shared Super-cluster L2 |
| Unified Memory Bandwidth (`UMA`) | `546 GB/s` (LPDDR5X-8533) | `205 GB/s` (`M5`) / `384 GB/s` (`M5 Pro`) | `614 GB/s` (LPDDR5X-9600) |
| VM Page & Metal `NoCopy` Alignment | `16,384 bytes` (`16 KB`) | `16,384 bytes` (`16 KB`) | `16,384 bytes` (`16 KB`) |
| Default `MSeriesUmaPool` P-Core Threads | `12` (`QOS_CLASS_USER_INTERACTIVE`) | `4` (`M5` Super) / `10-12` (`M5 Pro`) | `12-16` (`QOS_CLASS_USER_INTERACTIVE`) |
| L1 2D Spatial Tile (`l1_tile_width x height`) | `128 x 64` (`8 KB` base + halo) | `128 x 64` (`8 KB` base + halo) | `128 x 64` (`8 KB` base + halo) |
| L2 Working-Set Tile (`l2_tile_bytes`) | `8 MB` | `8 MB` (`M5`) / `12 MB` (`M5 Pro`) | `16 MB` |
| Metal 2D Threadgroup (`metal_threadgroup_2d`) | `16 x 16` (`256` threads = `8` simdgroups) | `16 x 16` (`256` threads = `8` simdgroups) | `16 x 16` (`256` threads = `8` simdgroups) |
| Verified 14-Stage Geometric Mean Speedup | **`34.70x`** vs OpenCV 5.0 / Dense | **`35.26x - 35.75x`** vs OpenCV 5.0 / Dense | Ready (`docs/apple-metal-m4-m5/03-*`) |

---

## 3. Environment Variables & Runtime Overrides

All three machines auto-detect their hardware profile with zero configuration required. You can also override or cross-test any profile using environment variables:

```bash
# 1. Auto-detect live Apple Silicon chip + compile native Metal 3.1/4.0 compute pipelines:
cargo test --release
cargo run --release --example full_suite_benchmark

# 2. Force a specific Apple Silicon chip profile (for cross-machine verification or pinned tuning):
SOLIDCV_APPLE_CHIP=M4_MAX cargo run --release --example full_suite_benchmark
SOLIDCV_APPLE_CHIP=M5_PRO cargo run --release --example full_suite_benchmark
SOLIDCV_APPLE_CHIP=M5_MAX cargo run --release --example full_suite_benchmark

# 3. Override P-core worker thread count (defaults to hw.perflevel0.logicalcpu):
SOLIDCV_THREADS=12 cargo run --release --example full_suite_benchmark
```

---

## 4. Native Compiler Flags for Maximum M4 Max / M5 Pro / M5 Max Throughput

`Cargo.toml` already configures `[profile.release]` with `opt-level = 3`, `lto = "thin"`, and `codegen-units = 1`. When building on an M4 Max, M5 Pro, or M5 Max machine for local production deployment, enable native CPU target features (`+dotprod,+i8mm,+fp16,+bf16,+SHA3,+aes`):

```bash
RUSTFLAGS="-C target-cpu=native" cargo test --release
RUSTFLAGS="-C target-cpu=native" cargo run --release --example full_suite_benchmark
```

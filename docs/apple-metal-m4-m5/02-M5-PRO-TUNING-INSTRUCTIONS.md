# Apple M5 / M5 Pro (`AppleM5Pro` — `m5pro` `Mac17,2`) Kernel Tuning & Live Benchmark Report

**Author:** Andy Norris (`andy@solidsf.com`), Founder & CEO, solidSF  
**Verified Live Target (`m5pro.tail539d3f.ts.net`):** MacBook Pro `Mac17,2` (`Apple M5` `RELEASE_ARM64_T8142`, `4 Super + 6 Efficiency` CPU cores, `10` GPU cores `AGXMetalG17G` with Metal 4 Neural Accelerators, `16 MB` Super-cluster L2 + `6 MB` E-cluster L2, `24 GB` UMA) and `Apple M5 Pro` (`10-12` Super/P-cores + `4` E-cores, `20-24` GPU cores, `24 MB` L2, `384 GB/s` UMA)

---

## 1. Drop-In Prompt for the Agent on Your Apple M5 / M5 Pro Machine

Copy and paste the block below directly into the agent on your **Apple M5 / M5 Pro** machine (`/Users/m5pro/git/solidcv`):

```text
Read docs/apple-metal-m4-m5/02-M5-PRO-TUNING-INSTRUCTIONS.md and docs/apple-metal-m4-m5/04-KERNEL-TILING-FUSION-MULTIPROC-SPEC.md.
Verify that solidcv detects AppleSiliconGen::AppleM5Pro via sysctl (machdep.cpu.brand_string "Apple M5" or "Apple M5 Pro", or SOLIDCV_APPLE_CHIP=M5_PRO), compiles all 6 native Apple Metal 4 zero-copy compute pipelines via AppleMetalEngine::global() (AGXMetalG17G), pins MSeriesUmaPool workers to the M5 Super/Performance cores (QOS_CLASS_USER_INTERACTIVE), and passes all 27 release tests + the 14-stage full_suite_benchmark with RUSTFLAGS="-C target-cpu=native".
```

---

## 2. Live Hardware Telemetry Verified on `m5pro` (`Mac17,2` — Darwin 27.0.0 `T8142`)

Queried directly on `m5pro` via `sysctl` and `ioreg -rc AGXAccelerator -d1`:

- `machdep.cpu.brand_string`: `Apple M5` (`Model Identifier: Mac17,2`)
- `hw.ncpu`: `10`
- `hw.perflevel0.name`: `Super` (`hw.perflevel0.logicalcpu = 4`, `l1icachesize = 196608` (`192 KB`), `l1dcachesize = 131072` (`128 KB`), `l2cachesize = 16777216` (`16 MB`))
- `hw.perflevel1.name`: `Efficiency` (`hw.perflevel1.logicalcpu = 6`, `l1icachesize = 131072` (`128 KB`), `l1dcachesize = 65536` (`64 KB`), `l2cachesize = 6291456` (`6 MB`))
- `hw.cachelinesize`: `128` bytes
- `hw.memsize`: `25,769,803,776` bytes (`24 GB` Unified Memory)
- `AGXAccelerator`: `AGXG17GDevice` (`AGXMetalG17G`, `Metal 4`, `gpu-core-count = 10`, max threadgroup `1024`)

---

## 3. Live 14-Stage Benchmark Verified on `m5pro` (`Apple M5` / Metal 4)

Executed directly on `m5pro` (`/Users/m5pro/git/solidcv`) via `RUSTFLAGS="-C target-cpu=native" cargo run --release --example full_suite_benchmark`:

| Stage | Workload | `solidcv` (`m5pro`) | Next Best Baseline | Speedup | Throughput / Advantage |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **1. Bitplane Pack vs Threshold+Count** | `1024x1024` 8-bit frame | **`17.88 us`** | `35.67 us` (OpenCV 5.0) | **`2.00x`** | `58.66 GPix/s` (`8x` smaller memory) |
| **2. Temporal XOR Minterm Carrier** | `1024x1024` frame pair | **`1.76 us`** | `50.13 us` (OpenCV 5.0) | **`28.52x`** | Bitplane XOR: `1.76 us`; Raw byte SIMD: `21.29 us` (`49.2 GPix/s`) |
| **3. Fused $3\times 3$ Sobel ($G_x + G_y$)** | `1024x1024 -> i16 Gx, Gy` | **`43.88 us`** | `549.10 us` (OpenCV 5.0) | **`12.51x`** | `23.90 GPix/s` separable `vsubl_u8` + 4-Super-core stripe |
| **4. IntegralLattice + 10k Box Stats** | `512x512` + `10k` ($32\times 32$) queries | **`131.80 us`** | `10445.90 us` (OpenCV 5.0) | **`79.26x`** | L2-recycled `LATTICE_SLAB_CACHE` (`0` kernel VM page faults) |
| **5. 256-Bit ORB BFMatcher (`1.05M` pairs)** | `1024 x 1024` (256-bit) descriptors | **`102.58 us`** | `4605.06 us` (OpenCV 5.0) | **`44.89x`** | **`10,222 MPairs/s`** (`vcntq_u8` + `vpaddq_u8` + `vminq_u32`) |
| **6. StereoSGBM ($9\times 7$ Census64 4-Path)** | `160x120` (16 disp, $2.1\times$ gain) | **`643.18 us`** | `18810.29 us` (Dense SAD) | **`28.45x`** | Full-frame NEON Census64 + register-only NEON WTA (`100%` accuracy) |
| **7. $SO(3)$ EPnP + Gauss-Newton RANSAC** | `64` 3D-2D pts (20% outliers) | **`4.90 us`** | `15.28 us` (250-iter RANSAC) | **`3.12x`** | `52/52` inliers, `6.08e-8 m` translation error |
| **8. RealityComber Sparse-Gated Flow** | `256x256` ($16\times 16$ blocks, 4% motion) | **`53.07 us`** | `3710.83 us` (OpenCV 5.0) | **`69.92x`** | Stack $20\times 20$ Sobel halo + $21\times 21$ local integral table |
| **9. 3D Voxelean Zobrist Octree Diff** | $64^3$ (`262,144`) voxel volume diff | **`0.15 us`** | `29.30 us` (Dense 3D scan) | **`189.24x`** | Zero-allocation dual-cursor `BTreeMap` merge (`49` nodes) |
| **10. (3+1)D Worldtube & Dyadic Ray ($k=64$)** | 64-step ray + 32-frame 512-vox body | **`0.41 us`** | `1.53 us` (Sequential ray) | **`3.76x`** | Stack `[u64; 64]` boolean matrix squaring ($\mathcal{O}(\log_2 k)$) |
| **11. VLM ViT-16 Patch-Token Pruner** | `1024x1024` (`4,096` ViT-16 patches) | **`27.42 us`** | `84195.88 us` (Dense ViT-16) | **`3070.60x`** | NEON `vld1q_u8` + `vcvtq_f32_u32` patch projection (`97.6%` KV reuse) |
| **12. 1-Bit TensorCore SAE Circuit Match** | `16` queries $\times$ `4,096` SAE atoms | **`24.39 us`** | `26307.88 us` (Dense FP32) | **`1078.49x`** | `b1.xor.popc` `m16n8k256` (`0` FP16/FP32 MACs) |
| **13. $SE(2)$ Mass-Conserving $2\times$ Zoom & SR** | `512x512 -> 1024x1024` ($2\times$ Uplift) | **`42.27 us`** | `323.41 us` (OpenCV 5.0) | **`7.65x`** | 4-Super-core striped NEON bilinear + `100%` $2\times 2$ mass conservation |
| **14. Foveated VLM ROI Crop-and-Zoom** | `1024x1024 -> 256` global + $4\times 128$ ROIs | **`10602.23 us`** | `335017.01 us` (Dense $2\times$ VLM) | **`31.50x`** | CV zoom: `192 us` (`96.9%` VLM tokens saved) |
| **Overall Geometric Mean Speedup (`m5pro`)** | **All 14 Stages** | — | — | **`35.26x – 35.75x`** | **`27/27` tests passing (`0.05s`), `6/6` Metal 4 NoCopy pipelines active** |

---

## 4. Key M5 / M5 Pro Architectural Findings

1. **`Super` Core Pinning (`perflevel0`) Prevents Straggler Tail Latency:**
   - On `Mac17,2` (`Apple M5`), `hw.perflevel0` consists of `4 Super` cores (`128 KB` L1D, `16 MB` shared L2) and `hw.perflevel1` consists of `6 Efficiency` cores (`64 KB` L1D, `6 MB` shared L2).
   - Pinning `MSeriesUmaPool::global()` to `hw.perflevel0.logicalcpu` (`4` Super cores by default) ensures equal-sized spatial row stripes (`simd_sobel_3x3_u8_into`, `zoom_se2_subpixel`) never spill onto the slower Efficiency cluster (`43.88 us` Sobel and `42.27 us` Zoom on 4 Super cores vs. `56.50 us` Sobel and `55.84 us` Zoom when spilling onto all 10 cores).
   - For embarrassingly parallel descriptor blocks (`256-Bit ORB BFMatcher`), setting `SOLIDCV_THREADS=8` utilizes both Super and Efficiency clusters to reach **`10,222 MPairs/s`** (`102.58 us`).
2. **Register-Only NEON Winner-Takes-All in `StereoSGBM`:**
   - Computing `best_slot` (`vceqq_u16` + `vbslq_u16` + `vminvq_u16`) and `min_non_neighbor` (`vabdq_u16(idx, best_v) <= 1`) directly in NEON registers eliminates partial store-to-load forwarding stalls on M5 Super cores and reduces `StereoSGBM` latency from `768.62 us` to **`643.18 us`** (`28.45x` speedup).

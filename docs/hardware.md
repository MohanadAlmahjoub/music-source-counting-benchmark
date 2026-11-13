# FPGA feasibility for the MDL/MUSIC pipeline

**Decision B — keep the CPU path as the measured baseline.** Keep a 14-bit projection kernel as the first programmable-logic candidate if target measurements show a need for acceleration. The simpler Q12 source-count comparator did not pass the reliability gate. The remaining MDL path calls an eigensolver, and moving covariance or spectrum projection alone leaves that stage on the CPU or AI Engine.

## What was measured

Run from the repository root after installing the Python dependencies.

~~~sh
python -m music_benchmark.hardware --trials 128 --output-dir results/hardware
~~~

[Per-fixture fixed-point results](../results/hardware/fixed-point.csv) use 3,840 paired synthetic frames on the i7-10750H. Each frame contains 128 snapshots of an eight-element half-wavelength ULA. The five fixtures cover noise only, one to three sources, 0 to 15 dB strongest-source SNR, a 12 dB power spread, and a close pair separated by 8°. The three noise scenarios have calibration/test Toeplitz correlations of 0/0, 0.2/0.2 and 0.2/0.35. Two independently seeded 2,048-sample noise-only calibration blocks were used per scenario. Both branches share the FP64 calibration matrix, MDL restricted to 0–3 sources, a 240-bin scan and the same local-minima peak picker.

- **IQ/covariance** rounds each real and imaginary component to a symmetric signed code, with nearest-even ties and a fixed full scale of 4, 8 or 16 in synthetic noise-RMS units. It accumulates the Hermitian covariance in integer arithmetic before scaling back to FP64, applies the independent calibration as W·R·Wᴴ, then runs FP64 eigendecomposition, MDL and angle estimation. The calibrated covariance transformation is algebraically equivalent to whitening each sample before covariance at infinite precision.
- **Projection** uses the floating covariance, eigendecomposition and MDL count. It quantizes both the normalized steering atoms and the noise eigenvectors, with component full scale 1, and accumulates the dot products and energy in integer arithmetic. The resulting denominator is scaled to FP64 for the same peak picker. The accumulation is exact for the sweep, so this is a coefficient-quantization model.
- **Combined** passes 12-bit IQ at full scale 16 through the integer covariance branch, followed by 14-bit projection. The eigensolver, MDL, calibration and peak picker stay in floating-point software. The model excludes analog ADC noise, gain errors and clock effects.

| Branch | MDL count agreement | Exact count and peak bins | Worst fixture exact count and bins | Count and all angles within 2° of truth |
| --- | ---: | ---: | ---: | ---: |
| FP64 reference | 100% | 100% | 100% | 98.65% |
| 8-bit IQ at full scale 16 | 99.82% | 97.24% | 91.41% | 98.72% |
| 12-bit IQ at full scale 16 | 99.97% | 99.27% | 96.48% | 98.67% |
| 12-bit IQ at full scale 4 | 99.92% | 75.89% | 22.27% | 98.65% |
| 12-bit projection | 100% | 99.38% | 97.66% | 98.65% |
| 14-bit projection | 100% | 99.84% | 99.22% | 98.65% |
| 12-bit IQ at full scale 16 plus 14-bit projection | 99.97% | 99.19% | 96.09% | 98.67% |

Full scale 4 clipped 10.48% of the raw IQ components pooled across fixtures, compared with 0.011% at full scale 16. Count agreement would hide this difference in angle-grid stability. The exact-bin and 2° columns address different tolerances. One grid step is about 0.706°. The synthetic fixtures use fixed angles, so the reported rates should be rechecked on varied angle geometries and an actual receiver input distribution before selecting a format.

## Analytic FPGA model

[Architecture sweep](../results/hardware/architecture-model.csv) assumes a hypothetical 200 MHz fabric clock, a 64-bit stream sustaining one beat each cycle, one real product per lane per cycle, and no stalls. The maximum of compute, ingress and egress is an optimistic steady-state interval floor with overlap. Their sum is the floor for a serialized schedule. These estimates omit control, register, accumulator dependency, reduction, peak-picking, FIFO, CDC, routing and external-memory overhead.

| Kernel | Multiplications per frame | Example lanes | Overlapped floor | Serialized floor |
| --- | ---: | ---: | ---: | ---: |
| Packed 12-bit covariance | 16,384 | 64 | 1.92 µs | 3.36 µs |
| 14-bit projection, two sources and 240 bins | 48,960 | 128 | 1.915 µs | 2.03 µs |
| 14-bit projection, two sources and 240 bins | 48,960 | 256 | 0.96 µs | 1.075 µs |

- **Covariance.** The eight diagonal terms require two real products each. The 28 distinct off-diagonal terms require four each, for 128 products per time sample. A signed 31-bit component accumulator safely covers 128 rows of symmetric 12-bit IQ. Sixty-four 32-bit storage words hold the diagonal and real/imaginary off-diagonal terms, totaling 256 B. A ping-pong pair of complete packed input frames is 6 KiB if a stream architecture needs frame staging. At one million frames/s, the input requires 3.072 GB/s before protocol overhead. A 64-bit, 200 MHz stream has a theoretical 1.6 GB/s ceiling; the covariance example is input-bound by 64 lanes.
- **Projection.** For two sources, six noise vectors and 240 bins require 11,520 complex multiplies and 2,880 real squares, using four real multiplications per complex multiply. The 240 × 8 static complex steering table occupies 6,720 B at 14 bits/component. A 6 × 8 dynamic noise basis occupies 168 B packed. A conservative hardware implementation can use 32-bit signed dot-product components and a 66-bit unsigned energy sum. The software model uses int64 with tighter bounds from the unit-norm coefficients. Sustaining 1 µs from the arithmetic alone would require at least 245 ideal real-multiply lanes for two sources, or 286 for one source. Wide squarers may consume several hardware DSP slices each, so lane count cannot be read as DSP usage. Zero detected sources can skip the scan.
- **Placement.** Direct sensor-to-PL covariance avoids the raw input transfer from a host and sends only the packed statistic to the next stage. For the current VCK190 tutorial, compare that boundary to its existing two input movers and AIE-local QR/SVD/scan. A PL projection with on-device peak extraction could send a compact result, while a separated peak picker needs the full scan scores. Keeping coefficient and frame state resident and combining adjacent stages would avoid repeated host or AI Engine/PL crossings. [AMD/Fidus MUSIC dataflow](https://github.com/Xilinx/Vitis-Tutorials/blob/2026.1/AI_Engine_Development/AIE/Design_Tutorials/18-MUSIC-Algorithm/README.md) · [AMD AM009 stream widths](https://docs.amd.com/r/en-US/am009-versal-ai-engine/AI-Engine-to-Programmable-Logic-Interface) · [AMD AM009 buffering](https://docs.amd.com/r/en-US/am009-versal-ai-engine/Stream-Switch-Buffering-and-Latency)

[Transfer sweep](../results/hardware/transfer-model.csv) reports payload/B + fixed overhead for assumed 2, 8 and 16 GB/s and total fixed transaction overhead of 1, 5 and 20 µs. At 8 GB/s plus 5 µs, a packed-IQ covariance round trip costs at least 5.416 µs and a two-source projection round trip with compact peaks 5.023 µs. These are assumptions for architecture comparisons. Batching amortizes launch and setup overhead per frame, while waiting for a batch adds latency. Break-even for a CPU-originated stage requires its measured replacement cost to exceed the offload compute, packing and measured transfer time for the same precision and scope.

## Decision boundary

Compare these floors with the measured [CPU stages](../results/profile/stage_summary.csv) and [GPU subpipeline](../results/gpu/summary.csv) only at matching scope. The FPGA model covers two individual kernels. The GPU test includes whitening, covariance and eigenvalues. The full CPU path additionally counts sources, forms a noise subspace, evaluates the spectrum and returns angles. Keep the CPU route as the result for latency-sensitive frames until a native or batched CPU baseline and real data establish the deployment bottleneck.

Before claiming board acceleration, validate an integer RTL datapath against the software vectors with randomized backpressure, prove accumulator bounds, and measure cycle latency in simulation. Then synthesize and route for a specific part and clock. Compare frame-to-angle p95, throughput under stalls, utilization and power on the actual board with the intended input and data movers. This experiment produced software numerical measurements and arithmetic/transfer estimates.

# ANE — Backpropagation on Apple Neural Engine

Training neural networks directly on Apple's Neural Engine (ANE) via reverse-engineered private APIs. No CoreML training APIs, no Metal, no GPU — pure ANE compute.

## Project Scope & Intent

I'm genuinely grateful for all the attention this project has received — I never expected a weekend research hack to blow up like this. Thank you to everyone who starred, forked, ran benchmarks on their own hardware, and shared the work. It means a lot.

That said, I want to set clear expectations about what this project is and isn't.

This is a **research project**, not a production framework.

The goal was to demonstrate that **training on the Apple Neural Engine — and potentially other NPUs — is possible**, and that the barrier has always been software support, not hardware capability. The ANE is a remarkably capable piece of silicon that Apple restricts to inference-only use through CoreML. This project bypasses that restriction using reverse-engineered private APIs to show what's possible when you give the hardware a chance.

### What this project is

- A proof of concept for ANE training via `_ANEClient` and `_ANECompiler` private APIs
- A set of benchmarks documenting real ANE performance characteristics (throughput, power, SRAM behavior)
- A reference for anyone exploring direct ANE access outside CoreML
- Research code that I update when I find something interesting

### What this project is not

- A maintained framework or library
- A replacement for CoreML, MLX, llama.cpp, or any production inference stack
- A path to training large models on consumer hardware (yet)

### On the hype

Some coverage of this project has overstated its implications. To be clear:

- Training works, but utilization is low (~2-3% of peak) with significant engineering challenges remaining
- Many element-wise operations still fall back to CPU
- This does **not** replace GPU training for anything beyond small research models today

The honest results — including all limitations — are documented in the accompanying articles:
- [Part 1: Reverse Engineering](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Part 2: Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)

### Fork it, build on it

This is MIT licensed for a reason. Everyone now has access to AI-assisted development tools that can adapt and extend code in hours. If this project is useful to you — take it, modify it, build something better. If you do something cool with it, I'd love to hear about it.

---

## What This Is

A from-scratch implementation of transformer training (forward + backward pass) running on the ANE in Apple Silicon. The ANE is a 15.8 TFLOPS (M4) inference accelerator that Apple does not expose for training. This project reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs and the MIL (Model Intermediate Language) format to run custom compute graphs — including backpropagation — directly on ANE hardware.

**Current results (M4, Stories110M, 12-layer Llama2 architecture):**
- 92 ms/step (static + ANE extras), 111 ms/step (dynamic weights)
- 1.15 TFLOPS sustained (static + ANE extras)
- Three training pipelines: static baseline, static + ANE offload, dynamic weights
- Python bridge for C-callable ANE access

## Quick Start

```bash
make setup    # Install Python deps (coremltools, mlx, torch)
make all      # Build all binaries
make test     # Auto-detect chip and run ANE test
make bench    # MLX vs MPS vs ANE benchmark suite
make demo     # MLX training demo
make train    # Build training binaries (see training/README.md)
```

## Three Training Pipelines

### 1. Static Baseline (`training/train_large`)
Weights baked as constants in MIL kernels — recompile every 10 steps via `exec()` restart.
- 72 kernels per compile batch
- **107 ms/step**, 7.6s compile per restart

### 2. Static + ANE Extras (`training/train_large_ane`)
Offloads classifier forward (32K conv), softmax, final RMSNorm, and RMSNorm backward to ANE.
- 86 kernels per compile batch
- **92 ms/step** (14% faster), 9.6s compile per restart
- `--no-ane-extras` to disable

### 3. Dynamic Weight Pipeline (`training/training_dynamic/`)
Weights passed via IOSurface spatial dimension — compile 9 kernels once at startup, no recompilation needed.
- 9 shared kernels across all 12 layers, zero recompile
- **111 ms/step**, 0.4s one-time compile
- No `exec()` restart, no compile limit issues
- **3.9x faster wall time** for short runs (no compile overhead)

### Performance Comparison (20 Steps)

| | Static Baseline | Static + ANE | Dynamic |
|---|---|---|---|
| **Wall time** | 10.1s | 11.7s | **~2.6s** |
| Compile | 7.6s (76%) | 9.6s (82%) | 0.4s (15%) |
| Train | 2.1s | 1.8s | 2.2s |
| **ms/step** | 107 | **92** | 111 |

## Architecture

6 ANE kernel types per layer (static pipeline):

| Kernel | Function | Weights |
|--------|----------|---------|
| `kFwdAttn` | RMSNorm + QKV projection + SDPA + output projection | Wq, Wk, Wv, Wo, rms1, mask |
| `kFwdFFN` | RMSNorm + SwiGLU FFN (W1, W3, SiLU, W2) | W1, W2, W3, rms2 |
| `kFFNBwd` | FFN backward (W2^T + SiLU_bwd + W1^T + W3^T) | W2^T, W1^T, W3^T |
| `kSdpaBwd1` | Wo^T + SDPA backward part 1 (dV, probs, dp) | Wo^T, mask |
| `kSdpaBwd2` | SDPA backward part 2 (softmax grad, dQ, dK) | — |
| `kQKVb` | QKV backward (Wq^T + Wk^T + Wv^T → dx) | Wq^T, Wk^T, Wv^T |

CPU handles: RMSNorm backward, residual connections, loss computation, dW gradient accumulation (cblas_sgemm), Adam optimizer.

Key optimizations:
- **Channel-first CPU layout** — matches ANE IOSurface `[1,C,1,S]` format, eliminates all transpose overhead
- **vDSP vectorized RMSNorm** — 10x faster than naive (6.7ms → 0.7ms)
- **GCD async cblas overlap** — dW gradient sgemms run in parallel with ANE evals
- **Dynamic weight packing** — activations + weights concatenated in IOSurface spatial dimension
- **ANE RMSNorm fusion** — RMSNorm folded into forward kernels as MIL ops
- **Wo^T fusion** — output projection backward merged into SDPA backward kernel
- **Forward taps** — Q, K, V, attention scores exposed via concat outputs
- **exec() restart** — bypasses ~119 ANE compile limit per process

## Python Bridge

The `bridge/` directory provides a C-callable API for ANE access from Python (via `ctypes`):

```c
ane_bridge_init();
ANEKernel *k = ane_bridge_compile(mil_text, weight_data, ...);
ane_bridge_write_input(k, 0, data, bytes);
ane_bridge_eval(k);
ane_bridge_read_output(k, 0, data, bytes);
ane_bridge_free(k);
```

## File Structure

```
ANE/
├── Makefile                       # Top-level build
├── ane_universal.h                # Universal ANE runtime (auto-detects chip)
├── ane_universal_test.m           # Universal test binary
├── benchmark_apple_silicon.py     # MLX vs MPS vs ANE benchmark suite
├── demo_mlx_training.py           # MLX transformer training demo
├── BENCHMARKS.md                  # Detailed benchmark results
│
├── bridge/                        # Python bridge API
│   ├── ane_bridge.h               # C-callable ANE API
│   └── ane_bridge.m               # Implementation
│
├── api_exploration.m              # Initial ANE API discovery
├── inmem_basic.m                  # In-memory MIL compilation POC
├── inmem_bench.m                  # ANE dispatch latency benchmarks
├── inmem_peak.m                   # Peak TFLOPS measurement
├── sram_bench.m                   # ANE SRAM bandwidth probing
├── sram_probe.m                   # SRAM size/layout exploration
│
└── training/                      # ANE training (Stories110M)
    ├── train_large.m              # Static baseline (72 kernels)
    ├── train_large_ane.m          # Static + ANE extras (86 kernels)
    ├── training_dynamic/          # Dynamic weight pipeline (9 kernels)
    │   ├── train.m                # Dynamic training loop
    │   ├── mil_dynamic.h          # Dynamic weight MIL generators
    │   └── config.h / io.h / cpu_ops.h
    ├── ane_classifier.h           # ANE classifier (32K conv) + softmax
    ├── ane_rmsnorm_bwd.h          # ANE RMSNorm backward
    ├── stories_*.h                # Static pipeline config + ops
    ├── dashboard.py               # TUI monitoring
    ├── download_data.sh           # TinyStories data download
    └── test_*.m                   # Hardware probe/validation tests
```

## Building

Requires macOS 15+ on Apple Silicon.

```bash
# Build all training binaries
cd training && make all

# Download training data
bash training/download_data.sh

# Train (static + ANE extras)
./training/train_large_ane stories110M.bin 256 100 1e-4

# Train (dynamic weights, no recompile)
cd training/training_dynamic && make train && ./train --scratch
```

No external dependencies. Uses only system frameworks + private ANE APIs resolved at runtime via `objc_msgSend`.

## Disclaimer

This project uses Apple's private, undocumented APIs (`_ANEClient`, `_ANECompiler`, `_ANEInMemoryModelDescriptor`). These APIs are not covered by any public stability guarantee and may change or break with any macOS update. This is independent research into Apple Neural Engine architecture, using APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA §1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)

---

*Built by a human + Claude, one weekend at a time.*

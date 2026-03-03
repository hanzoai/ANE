# ANE — Apple Neural Engine Training & Inference

Training and inference on Apple's Neural Engine across all Apple Silicon generations.

## Quick Start

```bash
make setup    # Install Python deps (coremltools, mlx, torch)
make test     # Auto-detect chip, run ANE test (M1→M4+)
make bench    # Full MLX vs MPS vs ANE benchmark suite
make demo     # MLX training demo (works on all Apple Silicon)
make train    # Build ANE training binary (M4+ only)
```

## Hardware Compatibility

| Chip | ANE Private API | CoreML ANE | Training | Recommended Backend |
|------|----------------|------------|----------|-------------------|
| **M1** | Compile fails | 2.4x speedup (256x256) | MLX only | MLX (6.72 TFLOPS) |
| **M2** | Untested | Works | MLX | MLX |
| **M3** | Untested | Works | MLX | MLX |
| **M4** | Full support | Works | ANE + CPU | ANE (1.78 TFLOPS sustained) |
| **M5** | Works (probed) | Works | ANE + CPU | ANE |

The `ane_universal.h` runtime auto-detects the chip and selects the best backend:
- **M4+**: `_ANEInMemoryModel` — fast in-memory MIL compilation (~15ms), direct ANE access
- **M1-M3**: CoreML fallback — compile via `MLModel`, ANE scheduled by CoreML runtime

## Benchmark Results (M1 Max 64GB, macOS 26.4)

### Matmul TFLOPS (fp16)

| Config | MLX | MPS | MLX speedup |
|--------|-----|-----|-------------|
| 768x2048 (Stories110M FFN) | 0.25 | 0.10 | 2.4x |
| 4096x4096 (Zen-4B dim) | 2.36 | 1.45 | 1.6x |
| 4096x11008 (Zen-4B FFN) | **6.72** | 3.20 | **2.1x** |
| 4096x4096 seq=2048 (long ctx) | 6.25 | 4.12 | 1.5x |

### Training (1-layer transformer, fwd+bwd+Adam)

| Backend | ms/step | Notes |
|---------|---------|-------|
| MLX | 62 ms | Metal GPU, best for M1-M3 |
| MPS (PyTorch) | 84 ms | Good PyTorch compat |
| ANE (M4 private) | 107 ms | 12-layer Stories110M |

### CoreML ANE Dispatch (M1 Max)

| Config | ANE (ms) | CPU (ms) | Speedup |
|--------|----------|----------|---------|
| 64x64 seq=16 | 0.62 | 0.22 | 0.36x (overhead) |
| **256x256 seq=64** | **0.41** | **0.99** | **2.44x** |
| 768x768 seq=64 | 2.56 | 1.77 | 0.69x |

ANE wins at medium tensor sizes via CoreML. Too-small or too-large tensors are CPU-bound.

## Architecture

### ANE Training (M4+, `training/`)

From-scratch transformer training using reverse-engineered `_ANEClient` / `_ANECompiler` private APIs. 6 ANE kernel types per layer:

| Kernel | Function | On ANE |
|--------|----------|--------|
| `fwdAttn` | RMSNorm + QKV + SDPA + Wo projection | Forward |
| `fwdFFN` | RMSNorm + SwiGLU FFN | Forward |
| `ffnBwd` | W2^T + SiLU_bwd + W1^T + W3^T | Backward dx |
| `sdpaBwd1` | Wo^T + attention backward (dV, dp) | Backward dx |
| `sdpaBwd2` | Softmax grad + dQ + dK | Backward dx |
| `qkvBwd` | Wq^T + Wk^T + Wv^T → dx | Backward dx |

Weight gradients (dW) computed on CPU via async `cblas_sgemm` overlapped with ANE execution.

### Universal Runtime (`ane_universal.h`)

Auto-detecting runtime that works M1→M4+:

```c
ane_universal_init();                          // Detect backend
UANEKernel *k = ane_compile_inmem(mil, ...);   // M4+: in-memory
// OR: falls back to CoreML compile             // M1-M3: CoreML
ane_universal_write_input(k, 0, data, bytes);
ane_universal_eval(k);                         // Unified evaluation
ane_universal_read_output(k, 0, data, bytes);
ane_universal_free(k);
```

## File Structure

```
ANE/
├── Makefile                     # Top-level build (make test/bench/demo)
├── ane_universal.h              # Universal ANE runtime (M1→M4+)
├── ane_universal_test.m         # Universal test binary
├── ane_probe_m1.m               # M1 ANE diagnostic (API surface, MIL format)
├── ane_coreml_bridge.m          # CoreML → ANE bridge with benchmarks
├── ane_m1_fix.m                 # MIL format fuzzer (tested 40+ combinations)
├── benchmark_apple_silicon.py   # MLX vs MPS vs ANE benchmark suite
├── demo_mlx_training.py         # Working MLX transformer training demo
├── gen_mil_m1.py                # Generate CoreML models for M1 testing
├── benchmark_results.json       # Raw benchmark data
├── BENCHMARKS.md                # Detailed benchmark results
│
├── api_exploration.m            # Initial ANE API discovery
├── inmem_basic.m                # In-memory MIL compilation POC
├── inmem_bench.m                # ANE dispatch latency benchmarks
├── inmem_peak.m                 # Peak TFLOPS measurement
├── sram_bench.m                 # ANE SRAM bandwidth probing
├── sram_probe.m                 # SRAM size/layout exploration
│
└── training/                    # M4+ ANE training (Stories110M)
    ├── Makefile
    ├── train_large.m            # Main: 12-layer Stories110M on ANE
    ├── stories_config.h         # Model config, structs
    ├── stories_io.h             # IOSurface I/O, NEON fp16 conversion
    ├── stories_mil.h            # MIL generators (6 kernel types)
    ├── stories_cpu_ops.h        # vDSP RMSNorm, cross-entropy, Adam
    ├── ane_runtime.h            # ANE private API wrapper
    ├── ane_mil_gen.h            # MIL text generation helpers
    ├── model.h                  # Model weights, initialization
    ├── forward.h                # Forward pass MIL generators
    ├── backward.h               # Backward pass MIL generators
    ├── dashboard.py             # TUI monitoring (loss, power, text gen)
    ├── tokenize.py              # TinyStories data prep
    └── test_*.m                 # Hardware probe/validation tests (10 files)
```

## Why M1-M3 Private API Fails

The `_ANECompiler` binary on M1 (H14 ANE) rejects MIL programs that work on M4 (H16 ANE). Systematic testing (40+ combinations of program version, opset tag, weight format, op types) confirmed:

1. **Descriptor creation**: Works on all chips (v1.0→v1.3, all opsets)
2. **Model creation**: Works on all chips
3. **Compilation**: **Fails on M1** with `CompilationFailure` or `InvalidMILProgram`
4. **Weight-free ops** (identity, relu, add, mul): **Descriptor returns nil** on all chips — the in-memory API requires weights

The ANE compiler is a hardware-specific binary. Apple ships different compiler code for each ANE generation (H14/H15/H16). The MIL→hardware microcode translation is not forward/backward compatible through the private API.

**CoreML works because** it compiles models at a higher level and produces hardware-specific code internally, using the correct compiler for each chip.

## Integration Points

| Project | Path | Integration |
|---------|------|-------------|
| **hanzo/engine** | `~/work/hanzo/engine/` | Rust ML inference engine (Candle/mistral-rs). Metal + MLX quantization already integrated. ANE via CoreML pipeline. |
| **luxcpp/metal** | `~/work/luxcpp/metal/` | 80+ Metal shaders, `LUX_METAL_USE_MLX=ON`. ANE could be a new backend plugin. |
| **lux/accel** | `~/work/lux/accel/` | Go GPU acceleration (CGO→luxcpp). Metal/CUDA backends. |
| **python-sdk** | `~/work/hanzo/python-sdk/` | MLX inference engine + LoRA training docs for Zen models. |
| **studio** | `~/work/hanzo/studio/` | ComfyUI fork with full PyTorch MPS integration. |

## Disclaimer

Independent research into Apple Neural Engine architecture using undocumented APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA §1201(f)). No Apple proprietary code or binaries are included. Not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)

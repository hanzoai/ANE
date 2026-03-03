# Apple Silicon ML Benchmarks — M1 Max 64GB

Benchmarked 2026-03-03 on macOS 26.4, M1 Max (10-core CPU, 32-core GPU, 16-core ANE).

## TL;DR

| Backend | Best Use | Training? | Peak TFLOPS (fp16) |
|---------|----------|-----------|-------------------|
| **MLX** | Training + Inference | Yes | 6.72 |
| **MPS** | PyTorch compat | Yes | 4.12 |
| **ANE (private)** | Training on M4+ | M4+ only | 1.78 (M4) |
| **ANE (CoreML)** | Inference only | No | ~0.002 (overhead) |

**Verdict**: Use **MLX** for everything on M1 Max. ANE private API training only works on M4+.

## Matmul Performance (fp16)

| Config | MLX (ms) | MLX TFLOPS | MPS (ms) | MPS TFLOPS | MLX speedup |
|--------|----------|------------|----------|------------|-------------|
| 768x256 @ 256x768 (Stories110M dim) | 6.1 | 0.05 | 7.7 | 0.04 | 1.3x |
| 768x256 @ 256x2048 (Stories110M FFN) | 3.2 | 0.25 | 7.7 | 0.10 | 2.4x |
| 4096x512 @ 512x4096 (Zen-4B dim) | 7.3 | 2.36 | 11.8 | 1.45 | 1.6x |
| 4096x512 @ 512x11008 (Zen-4B FFN) | 6.9 | 6.72 | 14.5 | 3.20 | 2.1x |
| 4096x2048 @ 2048x4096 (Zen-4B ctx) | 11.0 | 6.25 | 16.7 | 4.12 | 1.5x |

## Transformer Block (forward pass)

| Config | MLX (ms) | MPS (ms) |
|--------|----------|----------|
| Stories110M (dim=768, 12 heads, seq=256) | 7.7 | - |
| Zen-4B (dim=4096, 32 heads, seq=512) | 63.1 | - |

## Training Step (fwd + bwd + Adam)

| Config | MLX (ms) | MPS (ms) | MLX speedup |
|--------|----------|----------|-------------|
| 1-layer dim=768 seq=256 (Stories110M) | 62 | 84 | 1.35x |

## ANE Status by Chip Generation

| Chip | Private API | CoreML ANE | Training | Notes |
|------|-------------|------------|----------|-------|
| M1 | Compile fails (InvalidMILProgram) | Works (slow dispatch) | No | H14 ANE, MIL v1.0 format |
| M2 | Unknown | Works | Unknown | H15 ANE |
| M3 | Unknown | Works | Unknown | H15 ANE |
| M4 | Works (tested) | Works | Yes (9.3ms/step) | H16 ANE, MIL v1.3+ format |
| M5 | Works (probed) | Works | Yes | H16 ANE family |

## ANE Private API Diagnostic (M1 Max)

All 74 ANE classes present including `_ANEInMemoryModel`, `_ANEChainingRequest`.
Descriptor creation succeeds. **Compilation fails** — the M1 H14 ANE compiler rejects
both hand-crafted MIL v1.0 and CoreML-compiled MIL v1.3 text.

Error: `ANECCompile() FAILED: err=(InvalidMILProgram)` / `err=(CompilationFailure)`

CoreML ANE dispatch overhead: ~3-4ms per inference, making it slower than CPU for small ops.

## Recommendations

### For Zen Model Training (Qwen3-4B)
- **M1 Max 64GB**: MLX LoRA fine-tuning. 4B model fits comfortably in unified memory.
- **M4 Pro/Max**: MLX + ANE hybrid (ANE for forward conv, MLX for everything else)
- **Cloud GPU**: For full fine-tuning or 8B+ models

### For Zen Model Inference
- **MLX**: 4-bit quantization, 460+ tok/s target, flexible
- **CoreML**: Production deployment, Apple-optimized scheduling
- **hanzo/engine**: Metal backend via Candle, AFQ quantization, mistral-rs compatible

### For Hanzo Infrastructure
- **hanzo/engine**: Already has Metal + MLX quantization. Add CoreML pipeline for production.
- **luxcpp/metal**: 80+ Metal shaders, MLX integration enabled. ANE could be a new backend plugin.
- **lux/accel**: GPU acceleration via CGO → luxcpp. ANE would need a new backend.

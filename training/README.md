# ANE Training — Stories110M on Apple Neural Engine

Training a 109M-parameter Llama2-architecture transformer (Stories110M) directly on Apple's Neural Engine using private ANE APIs.

![Dashboard](dashboard.gif)

## Architecture

- **Model**: Stories110M — dim=768, hidden=2048, heads=12, layers=12, vocab=32000, seq=256
- **109.53M params** (84.95M transformer + 24.58M embedding)
- **72 ANE kernels** per compile (60 weight-bearing, 12 weight-free sdpaBwd2)
- **6 kernel types per layer**: fwdAttn, fwdFFN, ffnBwd, sdpaBwd1, sdpaBwd2, qkvBwd

## Performance

| Component | Time (ms/step) |
|-----------|---------------|
| ANE eval | 9.6 |
| IO (fp16 conversion) | 4.1 |
| Classifier (cblas) | 9.1 |
| Cross-entropy + residuals | 14.4 |
| RMSNorm | 0.1 |
| **Total** | **107 ms/step** |

## Files

| File | Description |
|------|-------------|
| `train_large.m` | Main training loop — 12-layer forward/backward, checkpoint, exec() restart |
| `stories_config.h` | Model config, structs, alloc helpers |
| `stories_io.h` | IOSurface I/O, NEON fp16 conversion, kernel compile/eval |
| `stories_mil.h` | MIL program generators for all 6 ANE kernel types |
| `stories_cpu_ops.h` | vDSP-vectorized RMSNorm, cross-entropy, Adam, embedding ops |
| `dashboard.py` | TUI dashboard — loss curve, power/CPU/memory graphs, text generation |
| `tokenize.py` | Extract pretokenized TinyStories data |
| `Makefile` | Build targets |

## How it works

1. **Forward pass**: Each layer runs fwdAttn (QKV + SDPA + Wo) and fwdFFN (W1 + SiLU(W3) + W2) on ANE via MIL-compiled kernels. Final RMSNorm + classifier matmul on CPU (cblas).

2. **Backward pass**: Reverse layer order. ffnBwd, sdpaBwd1, sdpaBwd2, qkvBwd on ANE. Weight gradients (dW) via async cblas_sgemm on CPU. RMSNorm backward via vDSP.

3. **Compile budget**: ANE has a ~119 compile limit per process. With 72 kernels per batch, we run 10 accumulation steps then `exec()` restart with checkpoint resume.

4. **Data**: Real TinyStories text (20M tokens), mmap'd uint16 token IDs, random position sampling per step.

## Usage

### 1. Download Training Data

```bash
bash download_data.sh
```

Downloads pretokenized TinyStories (Llama 2 BPE, 32K vocab) from [enio/TinyStories](https://huggingface.co/datasets/enio/TinyStories) on HuggingFace. Produces `tinystories_data00.bin` (~41 MB, ~20M tokens).

### 2. Build & Train

```bash
# Baseline: classifier + softmax on CPU
make train_large
./train_large --steps 100        # quick test
./train_large                    # full 10k steps
./train_large --resume           # resume from checkpoint

# ANE-offloaded: classifier + softmax on ANE (faster)
make train_large_ane
./train_large_ane --steps 100
```

**CLI flags:** `--steps N` (default 10000), `--lr F` (default 3e-4), `--resume`.

### 3. Monitor with Dashboard

```bash
pip install blessed psutil numpy
sudo python3 dashboard.py          # live mode (needs powermetrics)
sudo python3 dashboard.py --resume  # attach to resumed training
```

### 4. Benchmarking

Both programs print an **Efficiency Report** at completion:

```
=== Efficiency Report ===
Total steps:     100
Avg train:       107.0 ms/step
ANE TFLOPS:      2.45 sustained
ANE utilization: 15.5% of 15.8 TFLOPS
```

Per-batch timing breakdown during training:

```
ane=9.6 io=4.1 cls=9.1 elem=14.4 rms=0.1 cblas_wait=2.3 ms/step
```

| Metric | What it measures |
|--------|-----------------|
| `ane` | ANE kernel evaluation |
| `io` | fp16↔fp32 IOSurface transfer |
| `cls` | Classifier matmul (CPU cblas) |
| `elem` | Embedding, residual adds, cross-entropy |
| `rms` | RMSNorm forward/backward |
| `cblas_wait` | Waiting for async dW gradient sgemms |

Compare baseline vs ANE-offloaded:

```bash
make train_large && ./train_large --steps 100
make train_large_ane && ./train_large_ane --steps 100
```

## Key techniques

- **NEON vectorized fp16<->fp32**: ARM NEON intrinsics for fast IOSurface data transfer
- **vDSP cross-entropy**: `vDSP_mtrans` + `vvexpf` + `vDSP_sve` — 8x faster than scalar
- **Async weight gradients**: cblas_sgemm dispatched to background queue, overlapped with ANE
- **SDPA causal mask workaround**: ANE hardware ignores attn_mask, so we decompose attention into Q@K^T (ANE conv) + mask+softmax (CPU) + scores@V (ANE conv)

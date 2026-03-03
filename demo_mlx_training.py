#!/usr/bin/env python3
"""
MLX Training Demo — Train a small transformer on Apple Silicon

Shows how to:
1. Build a transformer model in MLX
2. Train with forward + backward + Adam optimizer
3. Generate text from the trained model
4. Benchmark training throughput

This is the practical path for Zen model fine-tuning on M1/M2/M3/M4 Macs.
For full Zen model LoRA fine-tuning, see: ~/work/hanzo/python-sdk/docs/TRAINING.md
"""
import time
import math
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


# ============================================================
# Model: Tiny Transformer (Llama2 architecture)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * rms * self.weight


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, mask=None):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.attention = Attention(dim, n_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def __call__(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class TinyLlama(nn.Module):
    """Small Llama2-architecture transformer for demo/benchmarking."""

    def __init__(self, vocab_size=256, dim=256, n_layers=4, n_heads=8, hidden_dim=512, max_seq=128):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = [TransformerBlock(dim, n_heads, hidden_dim) for _ in range(n_layers)]
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Causal mask
        self.max_seq = max_seq

    def __call__(self, tokens):
        B, T = tokens.shape
        x = self.embed(tokens)
        # Causal attention mask
        mask = mx.full((T, T), float("-inf"))
        mask = mx.triu(mask, k=1)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.head(x)

    def generate(self, prompt_tokens, max_new=50, temperature=0.8):
        """Simple autoregressive generation."""
        tokens = list(prompt_tokens)
        for _ in range(max_new):
            x = mx.array([tokens[-self.max_seq:]])
            logits = self(x)
            next_logits = logits[0, -1] / temperature
            probs = mx.softmax(next_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10))
            mx.eval(next_token)
            tokens.append(next_token.item())
        return tokens


# ============================================================
# Training
# ============================================================

def create_dataset(n_samples=1000, seq_len=64, vocab_size=256):
    """Create synthetic training data — repeating byte patterns."""
    data = []
    for _ in range(n_samples):
        # Random pattern of 4-8 bytes, repeated to fill sequence
        pattern_len = np.random.randint(4, 9)
        pattern = np.random.randint(0, vocab_size, pattern_len)
        repeated = np.tile(pattern, seq_len // pattern_len + 1)[:seq_len + 1]
        data.append(repeated)
    return np.array(data)


def train():
    print("╔═══════════════════════════════════════════╗")
    print("║  MLX Training Demo — Tiny Transformer     ║")
    print("╚═══════════════════════════════════════════╝\n")

    # Config
    vocab_size = 256
    dim = 256
    n_layers = 4
    n_heads = 8
    hidden_dim = 512
    seq_len = 64
    batch_size = 8
    n_epochs = 3
    lr = 3e-4

    # Count params
    model = TinyLlama(vocab_size, dim, n_layers, n_heads, hidden_dim, seq_len)
    mx.eval(model.parameters())
    n_params = sum(p.size for p in model.parameters().values() if isinstance(p, mx.array))
    # Flatten nested
    def count_params(tree):
        total = 0
        if isinstance(tree, mx.array):
            return tree.size
        elif isinstance(tree, dict):
            for v in tree.values():
                total += count_params(v)
        elif isinstance(tree, (list, tuple)):
            for v in tree:
                total += count_params(v)
        return total
    n_params = count_params(model.parameters())
    print(f"Model: dim={dim}, layers={n_layers}, heads={n_heads}, hidden={hidden_dim}")
    print(f"Parameters: {n_params:,} ({n_params * 2 / 1e6:.1f} MB in fp16)")
    print(f"Training: batch={batch_size}, seq={seq_len}, lr={lr}")

    # Data
    print("\nGenerating synthetic dataset...")
    dataset = create_dataset(n_samples=2000, seq_len=seq_len, vocab_size=vocab_size)
    n_batches = len(dataset) // batch_size

    # Optimizer
    optimizer = optim.Adam(learning_rate=lr)

    # Loss function
    def loss_fn(model, x, y):
        logits = model(x)
        # Cross-entropy
        return mx.mean(nn.losses.cross_entropy(logits, y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training loop
    print(f"\nTraining for {n_epochs} epochs ({n_batches} batches/epoch)...\n")
    step = 0
    total_time = 0

    for epoch in range(n_epochs):
        # Shuffle
        perm = np.random.permutation(len(dataset))
        dataset_shuffled = dataset[perm]

        epoch_loss = 0
        epoch_steps = 0
        epoch_start = time.perf_counter()

        for i in range(0, len(dataset_shuffled) - batch_size, batch_size):
            batch = mx.array(dataset_shuffled[i:i + batch_size])
            x = batch[:, :-1]  # Input: all but last token
            y = batch[:, 1:]   # Target: all but first token

            t0 = time.perf_counter()
            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            step_ms = (time.perf_counter() - t0) * 1000

            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_steps += 1
            step += 1
            total_time += step_ms

            if step % 50 == 0 or step == 1:
                tokens_per_sec = batch_size * seq_len / (step_ms / 1000)
                print(f"  step {step:4d} | loss {loss_val:.4f} | {step_ms:.1f} ms/step | {tokens_per_sec:.0f} tok/s")

        epoch_time = time.perf_counter() - epoch_start
        avg_loss = epoch_loss / epoch_steps
        print(f"\n  Epoch {epoch + 1}/{n_epochs}: avg_loss={avg_loss:.4f}, time={epoch_time:.1f}s")

        # Generate sample
        prompt = [72, 101, 108, 108, 111]  # "Hello" in ASCII
        generated = model.generate(prompt, max_new=30, temperature=0.5)
        text = bytes(generated).decode("ascii", errors="replace")
        print(f"  Generated: {text[:80]}")
        print()

    # Final stats
    avg_step_ms = total_time / step
    tokens_per_sec = batch_size * seq_len / (avg_step_ms / 1000)
    print(f"{'='*50}")
    print(f"Training complete!")
    print(f"  Total steps: {step}")
    print(f"  Avg step time: {avg_step_ms:.1f} ms")
    print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
    print(f"  Total time: {total_time / 1000:.1f}s")
    print(f"\nThis demonstrates MLX training on Apple Silicon (Metal GPU).")
    print(f"For Zen model LoRA fine-tuning, use mlx-lm:")
    print(f"  mlx_lm.lora --model mlx-community/Qwen3-4B-Instruct-4bit \\")
    print(f"    --data ./data --train --iters 1000")


if __name__ == "__main__":
    train()

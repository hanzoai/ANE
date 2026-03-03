CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface -framework Accelerate
LDFLAGS = $(FRAMEWORKS) -ldl

# Universal binaries (work on M1→M4+)
BINS = ane_universal_test ane_probe_m1 ane_coreml_bridge ane_m1_fix

# Python venv for coremltools
VENV = .venv/bin/python

.PHONY: all clean bench train demo setup test help

help:
	@echo "ANE — Apple Neural Engine Benchmarks & Training"
	@echo ""
	@echo "  make setup       Install Python deps (coremltools, mlx, torch)"
	@echo "  make all         Build all binaries"
	@echo "  make test        Run universal ANE test (auto-detects M1/M4)"
	@echo "  make bench       Run comprehensive MLX vs MPS vs ANE benchmarks"
	@echo "  make demo        Run MLX training demo"
	@echo "  make train       Build training binaries (M4+ only)"
	@echo "  make clean       Remove built binaries"

all: $(BINS)

setup:
	uv venv .venv
	source .venv/bin/activate && uv pip install coremltools numpy mlx mlx-lm torch
	@echo "Setup complete. Run: make test"

# Universal test (M1→M4+)
ane_universal_test: ane_universal_test.m ane_universal.h
	$(CC) $(CFLAGS) -o $@ ane_universal_test.m $(LDFLAGS)

# Diagnostic probes
ane_probe_m1: ane_probe_m1.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

ane_coreml_bridge: ane_coreml_bridge.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

ane_m1_fix: ane_m1_fix.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Run tests
test: ane_universal_test
	./ane_universal_test

bench: setup
	$(VENV) benchmark_apple_silicon.py

demo: setup
	$(VENV) demo_mlx_training.py

# M4+ training (will fail on M1-M3)
train:
	$(MAKE) -C training train_large

clean:
	rm -f $(BINS)
	$(MAKE) -C training clean

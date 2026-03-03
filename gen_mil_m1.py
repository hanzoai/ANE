#!/usr/bin/env python3
"""Generate a minimal CoreML model and extract the MIL text that M1 ANE accepts."""
import os
import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb

print(f"coremltools: {ct.__version__}")

configs = [
    (64, 64, 16, "small"),
    (256, 256, 64, "medium"),
]

for CH_IN, CH_OUT, SEQ, label in configs:
    print(f"\n{'='*60}")
    print(f"Config: {label} — conv {CH_IN}→{CH_OUT}, seq={SEQ}")

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, CH_IN, 1, SEQ))])
    def prog(x):
        W = mb.const(val=np.random.randn(CH_OUT, CH_IN, 1, 1).astype(np.float16), name="W")
        x_fp16 = mb.cast(x=x, dtype="fp16", name="x_fp16")
        y = mb.conv(x=x_fp16, weight=W, strides=[1, 1], pad_type="valid", name="out")
        return y

    model = ct.convert(
        prog,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )

    pkg_path = f"/tmp/ane_m1_{label}.mlpackage"
    model.save(pkg_path)
    print(f"Saved: {pkg_path}")

    # Walk the package and dump text files
    for root, dirs, files in os.walk(pkg_path):
        for f in sorted(files):
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            rel = os.path.relpath(fp, pkg_path)
            if f.endswith(('.mil', '.json', '.txt')):
                print(f"\n--- {rel} ({sz} bytes) ---")
                with open(fp) as fh:
                    print(fh.read()[:3000])
            elif f.endswith('.bin'):
                print(f"Binary: {rel} ({sz} bytes)")
            elif f.endswith('.mlmodel') or f.endswith('.plist'):
                print(f"\n--- {rel} ({sz} bytes) ---")
                try:
                    with open(fp) as fh:
                        print(fh.read()[:1000])
                except:
                    print(f"  (binary plist)")
            else:
                print(f"File: {rel} ({sz} bytes)")

print("\nDone. Use the extracted MIL text to fix the Objective-C ANE runtime.")

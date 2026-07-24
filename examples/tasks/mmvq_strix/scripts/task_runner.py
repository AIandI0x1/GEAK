#!/usr/bin/env python3
"""Standalone benchmark harness for mmvq Q4_0 kernel on Strix Halo.
No torch dependency — uses hipcc directly."""
import subprocess
import sys
import os
import json
import argparse

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(TASK_DIR, "src")
BUILD_DIR = os.path.join(TASK_DIR, "build")
os.makedirs(BUILD_DIR, exist_ok=True)

HIPCC = os.environ.get("HIPCC", "hipcc")
HIP_ARCH = os.environ.get("HIP_ARCH", "gfx1151")

def compile_kernel():
    """Compile the HIP kernel as a shared library."""
    out = os.path.join(BUILD_DIR, "mmvq.so")
    cmd = [
        HIPCC, "--offload-arch=" + HIP_ARCH, "-O2", "-shared", "-fPIC",
        "-std=c++17", "-o", out,
        os.path.join(SRC_DIR, "mmvq.hip"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr
    return True, out

def compile_and_run_bench():
    """Compile driver + kernel together, run benchmark."""
    ok, msg = compile_kernel()
    if not ok:
        print(f"Kernel compilation: FAIL\n{msg}")
        return None
    print(f"Kernel compilation: OK ({msg})")

    driver_src = os.path.join(BUILD_DIR, "bench_driver.cpp")
    kernel_src = os.path.join(SRC_DIR, "mmvq.hip")
    driver_exe = os.path.join(BUILD_DIR, "bench_driver")
    cmd = [
        HIPCC, "--offload-arch=" + HIP_ARCH, "-O2", "-std=c++17",
        "-o", driver_exe, driver_src, kernel_src,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Driver compile failed: {result.stderr}")
        return None

    env = dict(os.environ, HSA_OVERRIDE_GFX_VERSION="11.5.1")
    result = subprocess.run([driver_exe], capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.stdout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["compile", "correctness", "performance", "bench"])
    args = parser.parse_args()

    if args.command == "compile":
        ok, msg = compile_kernel()
        if not ok:
            print(f"Compilation: FAIL\n{msg}")
            sys.exit(1)
        print(f"Compilation: OK ({msg})")
    elif args.command in ("correctness", "performance", "bench"):
        compile_and_run_bench()

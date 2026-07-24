#!/usr/bin/env python3
"""GEAK Hermes Agent Runner — local-first kernel optimization.

The LLM (hermes-agent) only does what it's good at:
  - Analysis (profiling, bottleneck identification)
  - Code generation (writing optimized kernel variants)
  - Strategy (deciding what to try next)

Everything mechanical is handled by Python directly:
  - Compilation (hipcc)
  - Benchmarking (running the compiled binary)
  - Correctness checking (comparing outputs)
  - File I/O (reading/writing kernel source)

This keeps each hermes-agent call short (single code-gen or analysis turn),
avoiding the timeout issues that arise when asking a slow local LLM to do
multi-step tool use (read -> write -> compile -> benchmark) in one turn.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _find_hermes_binary() -> str:
    import shutil as sh
    hermes = sh.which("hermes")
    if hermes:
        return hermes
    candidates = [
        os.path.expanduser("~/Desktop/0x1-Main/third_party/hermes-agent/.venv/bin/hermes"),
        os.path.expanduser("~/.local/bin/hermes"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError("hermes CLI not found")


def _find_json_in_text(text: str) -> dict | None:
    json_blocks = re.findall(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
    if json_blocks:
        try:
            return json.loads(json_blocks[-1])
        except json.JSONDecodeError:
            pass
    candidates = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start:i + 1])
                start = -1
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


class LocalBenchmarkHarness:
    """Handles compilation and benchmarking without an LLM.

    This is the mechanical side of kernel optimization — given a .hip file,
    compile it with hipcc and run the benchmark driver to measure performance
    and correctness.
    """

    def __init__(self, task_dir: str, hip_arch: str = "gfx1151"):
        self.task_dir = Path(task_dir)
        self.src_dir = self.task_dir / "src"
        self.build_dir = self.task_dir / "build"
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.hipcc = os.environ.get("HIPCC", "hipcc")
        self.hip_arch = hip_arch
        # Check for task-specific compile flags
        self.extra_flags = []
        flags_file = self.task_dir / "compile_flags.txt"
        if flags_file.exists():
            self.extra_flags = flags_file.read_text().strip().split()

    def compile_kernel(self, kernel_file: str = "mmvq.hip") -> tuple[bool, str]:
        """Compile a HIP kernel file to a shared library."""
        src = self.src_dir / kernel_file
        out = self.build_dir / src.with_suffix(".so").name
        cmd = [
            self.hipcc, f"--offload-arch={self.hip_arch}", "-O2",
            "-shared", "-fPIC", "-std=c++17", "-o", str(out), str(src),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, result.stderr
        return True, str(out)

    def compile_and_run(self, kernel_file: str, driver_file: str) -> dict:
        """Compile kernel + driver together, run benchmark, return parsed results."""
        src = self.src_dir / kernel_file
        driver = self.build_dir / driver_file
        exe = self.build_dir / "bench_driver"

        cmd = [
            self.hipcc, f"--offload-arch={self.hip_arch}", "-O3", "-std=gnu++17",
            "-ffast-math", "-fPIC",
        ] + self.extra_flags + [
            "-o", str(exe), str(driver), str(src),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"ok": False, "error": "compile", "stderr": result.stderr[-2000:]}

        env = dict(os.environ, HSA_OVERRIDE_GFX_VERSION="11.5.1")
        result = subprocess.run([str(exe)], capture_output=True, text=True, env=env)
        return {"ok": True, "stdout": result.stdout, "stderr": result.stderr}

    def run_benchmark(self, kernel_file: str = "mmvq.hip",
                      driver_file: str = "bench_driver.cpp") -> dict:
        """Full compile + benchmark cycle. Returns structured results."""
        self.log("Compiling kernel + driver...")
        res = self.compile_and_run(kernel_file, driver_file)
        if not res.get("ok"):
            self.log(f"Compile failed: {res.get('stderr', '')[:200]}")
            return res

        self.log("Running benchmark...")
        stdout = res.get("stdout", "")
        self.log(stdout.strip())

        # Parse results
        results = {"ok": True, "raw_output": stdout}
        for line in stdout.splitlines():
            if "Correctness:" in line:
                results["correctness"] = "PASS" if "PASS" in line else "FAIL"
            elif line.startswith("Baseline:"):
                results["baseline_ms"] = float(line.split(":")[1].strip().replace("ms", "").strip())
            elif line.startswith("Optimized:"):
                results["optimized_ms"] = float(line.split(":")[1].strip().replace("ms", "").strip())
            elif line.startswith("v2:"):
                results["v2_ms"] = float(line.split(":")[1].strip().replace("ms", "").strip())
            elif "Speedup" in line and "v2" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    results["v2_speedup"] = float(parts[-1].strip().replace("x", ""))
        return results

    def log(self, msg: str) -> None:
        print(f"  [harness] {msg}", flush=True)


class HermesAgentRunner:
    """Workflow runner using hermes-agent for LLM steps only."""

    def __init__(
        self,
        workflow_dir: str,
        model: str | None = None,
        provider: str | None = None,
        timeout_s: int = 300,
        retries: int = 2,
    ):
        self.workflow_dir = Path(workflow_dir)
        self.hermes_bin = _find_hermes_binary()
        self.model = model or os.environ.get("GEAK_HERMES_MODEL", "")
        self.provider = provider or os.environ.get("GEAK_HERMES_PROVIDER", "")
        self.timeout_s = timeout_s
        self.retries = max(1, retries)
        self._phase = "init"

    def log(self, msg: str) -> None:
        print(f"  [workflow] {msg}", flush=True)

    def phase(self, name: str) -> None:
        self._phase = name
        self.log(f"=== PHASE: {name} ===")

    def _build_cmd(self, prompt: str, no_tools: bool = True) -> list[str]:
        cmd = [self.hermes_bin, "-z", prompt, "--yolo", "--accept-hooks"]
        if no_tools:
            cmd += ["-t", ""]
        if self.model:
            cmd += ["-m", self.model]
        if self.provider:
            cmd += ["--provider", self.provider]
        return cmd

    def agent(self, prompt: str, opts: dict | None = None) -> dict | None:
        """Spawn hermes-agent for a single LLM turn (analysis or codegen)."""
        opts = opts or {}
        label = opts.get("label", "agent")
        timeout = opts.get("timeout", self.timeout_s)

        for attempt in range(1, self.retries + 1):
            try:
                self.log(f"  [{label}] attempt {attempt}/{self.retries}")
                cmd = self._build_cmd(prompt)
                proc = subprocess.run(
                    cmd, cwd=str(self.workflow_dir),
                    env=dict(os.environ),
                    capture_output=True, text=True, timeout=timeout,
                )
                if proc.returncode != 0:
                    err = proc.stderr[-500:] if proc.stderr else "no stderr"
                    self.log(f"  [{label}] rc={proc.returncode}: {err}")
                    if attempt < self.retries:
                        continue
                    return None

                output = proc.stdout.strip()
                if not output:
                    self.log(f"  [{label}] empty output")
                    if attempt < self.retries:
                        continue
                    return None

                result = _find_json_in_text(output)
                if result is not None:
                    self.log(f"  [{label}] OK (parsed JSON)")
                    return result
                try:
                    result = json.loads(output)
                    self.log(f"  [{label}] OK (raw JSON)")
                    return result
                except json.JSONDecodeError:
                    pass

                self.log(f"  [{label}] no JSON (len={len(output)})")
                self.log(f"  [{label}] preview: {output[:200]}")
                if attempt < self.retries:
                    continue
                return None

            except subprocess.TimeoutExpired:
                self.log(f"  [{label}] timed out after {timeout}s")
                if attempt < self.retries:
                    continue
                return None
            except Exception as e:
                self.log(f"  [{label}] error: {e}")
                if attempt < self.retries:
                    continue
                return None
        return None

    def agentT(self, prompt: str, opts: dict | None = None) -> dict | None:
        return self.agent(prompt, opts)

    def read_file(self, path: str) -> str:
        """Read a file directly (no LLM needed)."""
        try:
            return Path(path).read_text()
        except OSError as e:
            self.log(f"  [read_file] {path}: {e}")
            return ""

    def write_file(self, path: str, content: str) -> bool:
        """Write a file directly (no LLM needed)."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content)
            return True
        except OSError as e:
            self.log(f"  [write_file] {path}: {e}")
            return False


def run_local_optimization(
    task_dir: str,
    budget: int = 3,
    model: str | None = None,
) -> dict:
    """Run a full kernel optimization cycle with local-first architecture.

    LLM steps (hermes-agent):
      1. Analyze kernel + propose optimization directions
      2. Generate optimized kernel code for each direction
      3. Analyze benchmark results + decide next round

    Local steps (Python):
      - Compile kernel (hipcc)
      - Run benchmark
      - Check correctness
      - Read/write files
    """
    task_dir = str(Path(task_dir).resolve())
    task_name = Path(task_dir).name
    workflow_dir = Path(__file__).resolve().parent.parent / "kernel_workflow"

    runner = HermesAgentRunner(workflow_dir=str(workflow_dir), model=model)
    harness = LocalBenchmarkHarness(task_dir)

    runner.log(f"Task: {task_name}")
    runner.log(f"Budget: {budget} rounds")

    # === Step 1: Baseline benchmark (local, no LLM) ===
    runner.phase("Baseline")
    baseline = harness.run_benchmark()
    if not baseline.get("ok"):
        return {"ok": False, "error": "baseline failed", "detail": baseline}

    baseline_ms = baseline.get("baseline_ms", 0)
    runner.log(f"Baseline: {baseline_ms}ms")

    # Read the kernel source (local) — only the baseline for the prompt
    kernel_src_path = Path(task_dir) / "src" / "mmvq.hip"
    kernel_source = runner.read_file(str(kernel_src_path))
    if not kernel_source:
        return {"ok": False, "error": "cannot read kernel source"}

    # Extract just the baseline kernel for the prompt (keep it short)
    baseline_match = re.search(r'(__global__\s+.*?void\s+\w*baseline\w*.*?^})', kernel_source, re.DOTALL | re.MULTILINE)
    baseline_kernel = baseline_match.group(1) if baseline_match else kernel_source[:3000]

    # Also extract the helper functions (vec_dot, codebook, scale) - they're critical context
    helper_match = re.search(r'(static __device__.*?vec_dot_rocmfp4_q8_1.*?^})', kernel_source, re.DOTALL | re.MULTILINE)
    helper_code = helper_match.group(1) if helper_match else ""

    # Extract block definitions
    block_defs = """
typedef struct { uint8_t qs[16]; uint8_t e[2]; } block_rocmfp4;  // 18 bytes: 16 packed nibbles + 2 UE4M3 scales
typedef struct { uint8_t qs[16]; uint8_t e; } block_rocmfp4_fast;  // 17 bytes: 16 packed nibbles + 1 UE4M3 scale
typedef struct { union { struct { __half d; __half s; }; __half2 ds; }; int8_t qs[32]; } block_q8_1;  // 40 bytes
"""

    best_ms = baseline_ms
    best_speedup = 1.0
    best_variant = "baseline"
    all_variants = [{"name": "baseline", "ms": baseline_ms, "speedup": 1.0}]
    insights = []

    # === Optimization loop ===
    for round_num in range(1, budget + 1):
        runner.phase(f"Optimize round {round_num}")

        # --- LLM step: analyze + propose ---
        analyze_prompt = f"""You are a GPU kernel optimization engineer on Strix Halo (gfx1151, RDNA 3.5).

## Baseline kernel
```cpp
{baseline_kernel}
```

## Current benchmark
- Baseline: {baseline_ms}ms
- Best so far: {best_ms}ms ({best_speedup:.2f}x speedup, variant: {best_variant})

## Previous insights
{json.dumps(insights[-5:], indent=2) if insights else "None yet"}

## Strix Halo characteristics
- 40 CUs, wave32 default (NOT wave64)
- 128GB unified memory (~256 GB/s, no MALL/Infinity Cache)
- WMMA matrix instructions (not MFMA)
- Memory-bound: weight traffic is the floor

Propose ONE optimization direction for this round. Be specific.
Return JSON: {{"direction": {{"id": "r{round_num}", "title": "...", "change": "detailed description", "expected_speedup": 1.2, "risk": "..."}}}}
ONLY the JSON."""

        plan = runner.agent(analyze_prompt, {"label": f"r{round_num}:analyze", "timeout": 120})
        if not plan or not plan.get("direction"):
            runner.log(f"Round {round_num}: no direction proposed, stopping")
            break

        direction = plan["direction"]
        runner.log(f"Round {round_num}: {direction.get('title', 'unknown')}")

        # --- LLM step: generate code ---
        codegen_prompt = f"""You are a GPU kernel engineer on Strix Halo (gfx1151, RDNA 3.5, wave32).

## Optimization direction
{json.dumps(direction, indent=2)}

## Data structures (CRITICAL - use these exact field names)
```cpp
{block_defs}
```

## Baseline kernel (the REAL ROCmFP4 mmvq from llama.cpp)
```cpp
{baseline_kernel}
```

## vec_dot helper (the inner hot loop)
```cpp
{helper_code}
```

## Key intrinsics available on gfx1151
- __builtin_amdgcn_sudot4(true, a, true, b, c, false) — signed int8 dot4 (4 MACs/cycle)
- __builtin_amdgcn_perm(hi, lo, idx) — byte permute
- __shfl_xor, __shfl_down — warp shuffle (wave32)
- uint4 loads — 128-bit vectorized memory

## CRITICAL RULES
1. block_rocmfp4 has fields: qs[16] (packed 4-bit nibbles), e[2] (two UE4M3 scale bytes)
2. block_rocmfp4_fast has fields: qs[16], e (one UE4M3 scale byte)
3. block_q8_1 has fields: ds (half2, .d is delta, .s is sum), qs[32] (int8 values)
4. Do NOT invent field names. Use ONLY the fields listed above.
5. The scale conversion is: rocmfp4_ue4m3_to_fp32_half_finite(e[i]) — call the existing function
6. The codebook lookup is: rocmfp4_get_int_from_codebook_16(aux_q4, nullptr) — returns int2
7. The dot product is: ggml_cuda_dp4a(v.x, q8[l], sumi) — returns int

## Requirements
1. Write a new kernel function called "mmvq_rocmfp4_r{round_num}"
2. Same signature as baseline: (const block_rocmfp4* A, const block_q8_1* x, float* y, int n, int k)
3. Must produce correct results (within 0.1 tolerance vs baseline)
4. Use fixed-size __shared__ memory (max 32KB)
5. Include full warp reduction
6. No #include statements — just the function
7. You CAN call the existing helper functions (vec_dot_rocmfp4_q8_1, etc.) or write your own inner loop

Return JSON: {{"kernel_name": "mmvq_rocmfp4_r{round_num}", "code": "<complete __global__ function>", "notes": "what you changed"}}
The "code" field should contain ONLY the __global__ function.
ONLY the JSON."""

        codegen = runner.agent(codegen_prompt, {"label": f"r{round_num}:codegen", "timeout": 240})
        if not codegen or not codegen.get("code"):
            runner.log(f"Round {round_num}: codegen failed")
            insights.append({"round": round_num, "status": "codegen_failed", "direction": direction.get("title")})
            continue

        new_kernel_code = codegen["code"]
        runner.log(f"Round {round_num}: generated {len(new_kernel_code)} chars of kernel code")

        # --- Local step: insert kernel into source file ---
        # Append the new kernel to mmvq.hip
        with open(kernel_src_path, "a") as f:
            f.write(f"\n// === Round {round_num}: {direction.get('title', '')} ===\n")
            f.write(new_kernel_code)
            f.write("\n")

        # --- Local step: update bench_driver using insertion points ---
        driver_path = Path(task_dir) / "build" / "bench_driver.cpp"
        driver_src = runner.read_file(str(driver_path))
        kernel_name = codegen.get("kernel_name", f"mmvq_q4_0_f16_r{round_num}")

        if kernel_name not in driver_src:
            # Extract the kernel signature from the generated code
            # Handle __launch_bounds__, __global__, etc.
            sig_match = re.search(r'(?:__launch_bounds__\([^)]*\)\s*)?__global__\s+.*?void\s+' + re.escape(kernel_name) + r'\s*\(([^)]+)\)', new_kernel_code)
            if sig_match:
                params = sig_match.group(1)
                # Strip parameter names, keep only types for the extern
                # Simple approach: just use the full params string
                extern_line = f'extern __global__ void {kernel_name}({params});\n'
            else:
                # Fallback: copy types from the baseline extern in the driver
                baseline_extern = re.search(r'extern __global__ void \w*baseline\w*\(([^)]+)\)', driver_src)
                if baseline_extern:
                    extern_line = f'extern __global__ void {kernel_name}({baseline_extern.group(1)});\n'
                else:
                    extern_line = f'extern __global__ void {kernel_name}(const void*, const void*, float*, int, int);\n'

            driver_src = driver_src.replace(
                "// EXTERN_INSERTION_POINT",
                f"// EXTERN_INSERTION_POINT\n{extern_line}"
            )

            # Add benchmark + correctness check at BENCH_INSERTION_POINT
            # Use the same d_A, d_x, d_y that the baseline uses
            bench_code = f'''
    // Round {round_num}: {direction.get('title', '')}
    {{
        hipMemset(d_y, 0, n * sizeof(float));
        dim3 block(32, 2); dim3 grid(n);
        hipLaunchKernelGGL({kernel_name}, grid, block, 0, 0, d_A, d_x, d_y, n, k);
        check(hipMemcpy(h_y, d_y, n * sizeof(float), hipMemcpyDeviceToHost), "cpy r{round_num}");
        int mm = 0;
        for (int i = 0; i < n; i++) {{
            float diff = fabsf(h_y[i] - h_y_ref[i]);
            if (diff > 0.1f * fabsf(h_y_ref[i]) + 0.01f) mm++;
        }}
        printf("r{round_num}_correctness: %s (%d/%d mismatches)\\n", mm < n/100 ? "PASS" : "FAIL", mm, n);
    }}
    double r{round_num}_ms = bench_kernel({kernel_name}, d_A, d_x, d_y, n, k, 5, 100);
    printf("r{round_num}: %.3f ms\\n", r{round_num}_ms);
    printf("r{round_num}_speedup: %.3fx\\n", baseline_ms / r{round_num}_ms);
'''
            driver_src = driver_src.replace(
                "// BENCH_INSERTION_POINT",
                f"// BENCH_INSERTION_POINT\n{bench_code}"
            )

            runner.write_file(str(driver_path), driver_src)

        # --- Local step: compile + benchmark ---
        runner.phase(f"Benchmark round {round_num}")
        result = harness.run_benchmark()
        if not result.get("ok"):
            runner.log(f"Round {round_num}: compile/benchmark failed")
            insights.append({"round": round_num, "status": "compile_failed", "direction": direction.get("title")})
            continue

        # Extract the new variant's timing and correctness
        r_ms = None
        r_correctness = "UNKNOWN"
        for line in result.get("raw_output", "").splitlines():
            if line.startswith(f"r{round_num}:") and "ms" in line:
                try:
                    r_ms = float(line.split(":")[1].strip().replace("ms", "").strip())
                except ValueError:
                    pass
            elif line.startswith(f"r{round_num}_correctness:"):
                r_correctness = "PASS" if "PASS" in line else "FAIL"
            elif line.startswith(f"r{round_num}_speedup:"):
                try:
                    speedup = float(line.split(":")[1].strip().replace("x", "").strip())
                except ValueError:
                    pass

        if r_ms is None:
            runner.log(f"Round {round_num}: could not parse timing")
            insights.append({"round": round_num, "status": "parse_failed", "direction": direction.get("title")})
            continue

        speedup = baseline_ms / r_ms if r_ms > 0 else 0
        runner.log(f"Round {round_num}: {r_ms}ms ({speedup:.2f}x vs baseline, correctness={r_correctness})")

        if r_correctness != "PASS":
            runner.log(f"Round {round_num}: CORRECTNESS FAILED, skipping")
            insights.append({"round": round_num, "status": "correctness_failed", "direction": direction.get("title"), "ms": r_ms})
            continue

        all_variants.append({
            "name": kernel_name,
            "ms": r_ms,
            "speedup": speedup,
            "direction": direction.get("title"),
        })

        if r_ms < best_ms:
            best_ms = r_ms
            best_speedup = speedup
            best_variant = kernel_name
            runner.log(f"Round {round_num}: NEW BEST! {best_ms}ms ({best_speedup:.2f}x)")
            insights.append({
                "round": round_num,
                "status": "improved",
                "direction": direction.get("title"),
                "ms": r_ms,
                "speedup": speedup,
            })
        else:
            runner.log(f"Round {round_num}: no improvement over best ({best_ms}ms)")
            insights.append({
                "round": round_num,
                "status": "no_improve",
                "direction": direction.get("title"),
                "ms": r_ms,
                "speedup": speedup,
            })

    # === Final report ===
    runner.phase("Report")
    runner.log(f"Best variant: {best_variant} at {best_ms}ms ({best_speedup:.2f}x speedup)")
    runner.log(f"All variants: {json.dumps(all_variants, indent=2)}")

    return {
        "ok": True,
        "task": task_name,
        "baseline_ms": baseline_ms,
        "best_ms": best_ms,
        "best_speedup": best_speedup,
        "best_variant": best_variant,
        "all_variants": all_variants,
        "insights": insights,
        "rounds_completed": round_num,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GEAK kernel optimization with hermes-agent (local-first)")
    parser.add_argument("task_dir", help="Path to task directory (e.g. examples/tasks/mmvq_strix)")
    parser.add_argument("--budget", type=int, default=3, help="Number of optimization rounds")
    parser.add_argument("--model", default=None, help="Model override for hermes")
    args = parser.parse_args()

    result = run_local_optimization(
        task_dir=args.task_dir,
        budget=args.budget,
        model=args.model,
    )
    print(json.dumps(result, indent=2))

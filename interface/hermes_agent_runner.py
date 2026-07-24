#!/usr/bin/env python3
"""GEAK Hermes Agent Runner — replaces Claude Code's Workflow tool with hermes-agent.

This module provides the same `agent()` and `phase()` primitives that GEAK's JS
workflows expect, but drives them through `hermes -z` (oneshot mode) instead of
Claude Code's SDK.

Architecture:
  - `agent(prompt, opts)` → calls `hermes -z "<prompt>" --yolo` and parses JSON output
  - `phase(name)` → logs phase transitions (no-op in terms of execution)
  - `pipeline(items, ...fns)` → sequential/pipelined execution (simplified from JS parallel)

Usage:
  from hermes_agent_runner import HermesAgentRunner
  runner = HermesAgentRunner(workflow_dir="/path/to/kernel_workflow")
  result = runner.agent("You are the director...", {"label": "director:setup"})
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _find_hermes_binary() -> str:
    """Find the hermes CLI binary."""
    # Check PATH first
    import shutil
    hermes = shutil.which("hermes")
    if hermes:
        return hermes
    # Check known locations
    candidates = [
        os.path.expanduser("~/Desktop/0x1-Main/third_party/hermes-agent/.venv/bin/hermes"),
        os.path.expanduser("~/.local/bin/hermes"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError("hermes CLI not found. Install hermes-agent or set PATH.")


def _find_json_in_text(text: str) -> dict | None:
    """Extract the last JSON object from text (agent output may have prose around it)."""
    # Try to find JSON blocks marked with ```json
    json_blocks = re.findall(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
    if json_blocks:
        try:
            return json.loads(json_blocks[-1])
        except json.JSONDecodeError:
            pass

    # Try to find bare JSON objects (last one wins, since agents often output
    # reasoning before the final JSON)
    # Match { ... } with balanced braces (simplified)
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


class HermesAgentRunner:
    """Workflow runner that uses hermes-agent instead of Claude Code.

    Provides the same primitives as Claude Code's Workflow tool:
    - agent(prompt, opts): spawn a sub-agent for a role
    - phase(name): mark phase transitions
    - log(msg): print workflow progress
    """

    def __init__(
        self,
        workflow_dir: str,
        model: str | None = None,
        provider: str | None = None,
        timeout_s: int = 3600,
        retries: int = 4,
    ):
        self.workflow_dir = Path(workflow_dir)
        self.hermes_bin = _find_hermes_binary()
        self.model = model or os.environ.get("GEAK_HERMES_MODEL", "")
        self.provider = provider or os.environ.get("GEAK_HERMES_PROVIDER", "")
        self.timeout_s = timeout_s
        self.retries = max(1, retries)
        self._phase = "init"

    def log(self, msg: str) -> None:
        """Print a workflow log message."""
        print(f"  [workflow] {msg}", flush=True)

    def phase(self, name: str) -> None:
        """Mark a phase transition."""
        self._phase = name
        self.log(f"=== PHASE: {name} ===")

    def _build_hermes_cmd(self, prompt: str) -> list[str]:
        """Build the hermes CLI command for a one-shot agent call."""
        cmd = [self.hermes_bin, "-z", prompt, "--yolo", "--accept-hooks"]
        if self.model:
            cmd += ["-m", self.model]
        if self.provider:
            cmd += ["--provider", self.provider]
        # Use GEAK-specific hermes config if set, otherwise fall back to default
        geak_config = os.environ.get("GEAK_HERMES_CONFIG", "")
        if geak_config and os.path.isfile(geak_config):
            # hermes reads config from ~/.hermes/config.yaml by default;
            # we override via env var if the config path is set
            pass  # hermes doesn't support HERMES_CONFIG directly; use HOME override
        return cmd

    def agent(
        self,
        prompt: str,
        opts: dict | None = None,
    ) -> dict | None:
        """Spawn a hermes-agent sub-agent and return its structured JSON output.

        This replaces Claude Code's `agent()` primitive. The agent:
        1. Receives the prompt (which includes role instructions + inputs)
        2. Uses Bash/Read/Write tools to do filesystem/shell work
        3. Returns structured JSON

        Args:
            prompt: The full agent prompt (role + inputs + instructions)
            opts: Options dict with optional 'label', 'phase', 'schema', 'timeout'

        Returns:
            Parsed JSON dict from the agent's output, or None on failure
        """
        opts = opts or {}
        label = opts.get("label", "agent")
        timeout = opts.get("timeout", self.timeout_s)

        for attempt in range(1, self.retries + 1):
            try:
                self.log(f"  [{label}] attempt {attempt}/{self.retries}")
                cmd = self._build_hermes_cmd(prompt)
                env = dict(os.environ)
                # Ensure hermes runs in the workflow directory
                env["HERMES_WORKDIR"] = str(self.workflow_dir)

                proc = subprocess.run(
                    cmd,
                    cwd=str(self.workflow_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                if proc.returncode != 0:
                    err = proc.stderr[-500:] if proc.stderr else "no stderr"
                    self.log(f"  [{label}] hermes exited rc={proc.returncode}: {err}")
                    if attempt < self.retries:
                        self.log(f"  [{label}] retrying...")
                        continue
                    return None

                output = proc.stdout.strip()
                if not output:
                    self.log(f"  [{label}] empty output")
                    if attempt < self.retries:
                        continue
                    return None

                # Try to parse JSON from the output
                result = _find_json_in_text(output)
                if result is not None:
                    self.log(f"  [{label}] OK (parsed JSON)")
                    return result

                # If no JSON found, try parsing the entire output as JSON
                try:
                    result = json.loads(output)
                    self.log(f"  [{label}] OK (parsed raw JSON)")
                    return result
                except json.JSONDecodeError:
                    pass

                self.log(f"  [{label}] no JSON found in output (len={len(output)})")
                self.log(f"  [{label}] output preview: {output[:200]}")
                if attempt < self.retries:
                    continue
                return None

            except subprocess.TimeoutExpired:
                self.log(f"  [{label}] timed out after {timeout}s")
                if attempt < self.retries:
                    self.log(f"  [{label}] retrying...")
                    continue
                return None
            except Exception as e:
                self.log(f"  [{label}] error: {e}")
                if attempt < self.retries:
                    continue
                return None

        return None

    def agentT(self, prompt: str, opts: dict | None = None) -> dict | None:
        """Alias for agent() with timeout/retry handling (matches JS agentT)."""
        return self.agent(prompt, opts)


def role_prompt(
    runner: HermesAgentRunner,
    role: str,
    phase: str,
    intro: str,
    inputs: dict,
) -> str:
    """Build a role agent prompt (matches GEAK's roleAgent() in JS).

    The prompt tells the agent to:
    1. Read the role .md file
    2. Follow its instructions for the current phase
    3. Read knowledge files
    4. Do filesystem/shell work
    5. Return structured JSON
    """
    workflow_dir = runner.workflow_dir
    base = f"""You are the {role}. PHASE={phase}.
First Read {workflow_dir}/roles/{role}.md and follow its instructions for PHASE={phase}.
Read any knowledge files it points you to under {workflow_dir}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). {intro}

## Inputs
{json.dumps(inputs, indent=2)}

Return ONLY the structured JSON the role file specifies."""
    return base


def run_kernel_workflow(
    kernel_path: str,
    task: str = "optimize",
    budget: int = 4,
    target_language: str = "triton",
    gpu_ids: str = "0",
    eval_dir: str | None = None,
    model: str | None = None,
) -> dict:
    """Run the GEAK kernel workflow using hermes-agent.

    This is a simplified Python implementation of kernel_workflow.js that
    uses hermes-agent instead of Claude Code for the agent() calls.

    Args:
        kernel_path: Path to the kernel directory to optimize
        task: Task description
        budget: Number of optimization rounds
        target_language: Target language (triton, hip, flydsl)
        gpu_ids: Comma-separated GPU IDs
        eval_dir: Override eval directory
        model: Model override for hermes

    Returns:
        Dict with workflow results
    """
    workflow_dir = Path(__file__).resolve().parent.parent / "kernel_workflow"
    runner = HermesAgentRunner(
        workflow_dir=str(workflow_dir),
        model=model,
    )

    kernel_path = str(Path(kernel_path).resolve())
    kernel_name = Path(kernel_path).name
    eval_dir = eval_dir or f"/tmp/geak_eval_{kernel_name}_{int(time.time())}"
    os.makedirs(eval_dir, exist_ok=True)

    runner.log(f"Kernel: {kernel_path}")
    runner.log(f"Eval dir: {eval_dir}")
    runner.log(f"Budget: {budget} rounds")
    runner.log(f"Target language: {target_language}")

    # === Phase: Setup ===
    runner.phase("Setup")
    setup = runner.agentT(
        role_prompt(runner, "director", "setup",
                    "Build the isolated evaluation environment.", {
                        "KERNEL_PATH_ORIG": kernel_path,
                        "EXP_ROOT": eval_dir,
                        "EVAL_DIR_OVERRIDE": eval_dir,
                        "KERNEL_NAME_HINT": kernel_name,
                        "TASK": task,
                        "SKILL_DIR": str(workflow_dir),
                        "MODE": "optimize",
                        "TARGET_LANGUAGE": target_language,
                    }),
        {"phase": "Setup", "label": "director:setup"}
    )

    if not setup or not setup.get("eval_dir"):
        runner.log("Setup failed: director did not return eval_dir")
        return {"ok": False, "error": "Setup failed", "setup": setup}

    eval_dir = setup["eval_dir"]
    canonical = setup.get("workspace", kernel_path)
    runner.log(f"Setup done. EVAL_DIR={eval_dir}")

    # === Phase: Benchmark (baseline) ===
    runner.phase("Benchmark")
    bench = runner.agentT(
        role_prompt(runner, "benchmark_engineer", "benchmark",
                    "Measure the baseline kernel performance.", {
                        "EVAL_DIR": eval_dir,
                        "KERNEL_PATH": canonical,
                        "GPU_ID": gpu_ids.split(",")[0],
                        "SKILL_DIR": str(workflow_dir),
                    }),
        {"phase": "Benchmark", "label": "benchmark_engineer:baseline"}
    )

    if not bench:
        runner.log("Baseline benchmark failed")
        return {"ok": False, "error": "Baseline benchmark failed", "setup": setup}

    baseline_ms = bench.get("geomean_ms", 0)
    runner.log(f"Baseline: {baseline_ms}ms")

    # === Phase: Profile ===
    runner.phase("Profile")
    profile = runner.agentT(
        role_prompt(runner, "profile_engineer", "profile",
                    "Profile the kernel to find bottlenecks.", {
                        "EVAL_DIR": eval_dir,
                        "KERNEL_PATH": canonical,
                        "GPU_ID": gpu_ids.split(",")[0],
                        "SKILL_DIR": str(workflow_dir),
                    }),
        {"phase": "Profile", "label": "profile_engineer:profile"}
    )

    profile_summary = profile.get("summary", "") if profile else ""
    runner.log(f"Profile done: {profile_summary[:100]}")

    # === Phase: Optimize (loop) ===
    cumulative = 1.0
    no_improve = 0
    max_no_improve = 2

    for round_num in range(1, budget + 1):
        runner.phase(f"Optimize (round {round_num})")

        # Plan the round
        plan = runner.agentT(
            role_prompt(runner, "tech_lead", "plan_round",
                        "Decide this round's optimization directions.", {
                            "EVAL_DIR": eval_dir,
                            "ROUND": round_num,
                            "BUDGET_REMAINING": budget - round_num + 1,
                            "CUMULATIVE_SPEEDUP": cumulative,
                            "BASELINE_GEOMEAN_MS": baseline_ms,
                            "PROFILE_SUMMARY": profile_summary,
                            "SKILL_DIR": str(workflow_dir),
                        }),
            {"phase": "Optimize", "label": f"tech_lead:plan r{round_num}"}
        )

        if not plan or plan.get("stop") or not plan.get("directions"):
            runner.log(f"Round {round_num}: TechLead chose to stop")
            break

        directions = plan["directions"]
        runner.log(f"Round {round_num}: {len(directions)} direction(s)")

        for d in directions:
            d_id = d.get("id", f"r{round_num}_d0")
            specialty = d.get("specialty", "general")
            out_dir = f"{eval_dir}/round_{round_num}/engineer_{d_id}"
            os.makedirs(out_dir, exist_ok=True)

            # Optimize
            runner.log(f"  Engineer {d_id} ({specialty})...")
            eng = runner.agentT(
                f"""You are Engineer {d_id} (specialty={specialty}) for round {round_num}.
First create YOUR private workspace, then optimize.
```bash
mkdir -p {out_dir}/workspace
( cd {canonical} && tar --exclude=./.git --exclude='*/.git' --exclude=./build --exclude='*/build' \\
    --exclude=./__pycache__ --exclude='*/__pycache__ --exclude=./.torch_ext --exclude='*/.torch_ext' \\
    --exclude='*.so' --exclude='*.o' -cf - . ) | ( cd {out_dir}/workspace && tar -xf - )
```
Then Read {workflow_dir}/roles/engineer.md and {workflow_dir}/knowledge/self_monitoring.md and follow them.
Save best_patch.diff via `cd <KERNEL_PATH> && git diff > {out_dir}/best_patch.diff` when geomean>1.0.

## Inputs
{json.dumps({
    "SPECIALTY": specialty,
    "DIRECTION": d,
    "KERNEL_PATH": f"{out_dir}/workspace",
    "OUTPUT_DIR": out_dir,
    "CANONICAL": canonical,
    "GPU_ID": gpu_ids.split(",")[0],
    "SKILL_DIR": str(workflow_dir),
    "BASELINE_PER_CASE": bench.get("per_case", {}),
}, indent=2)}

Return ONLY the worker_result.json structure.""",
                {"phase": "Optimize", "label": f"eng {d_id}:{specialty}"}
            )

            if not eng or eng.get("status") == "failed":
                runner.log(f"  Engineer {d_id}: failed")
                continue

            speedup = eng.get("geomean_speedup", 1.0)
            runner.log(f"  Engineer {d_id}: speedup={speedup}x")

            if speedup > 1.0:
                # Verify
                runner.log(f"  Verifying {d_id}...")
                patch = f"{out_dir}/best_patch.diff"
                ver = runner.agentT(
                    role_prompt(runner, "verify_engineer", "verify",
                                "Independently re-measure this candidate patch.", {
                                    "CANONICAL": canonical,
                                    "PATCH": patch,
                                    "VERIFY_DIR": f"{out_dir}/verify",
                                    "GPU_ID": gpu_ids.split(",")[0],
                                    "SKILL_DIR": str(workflow_dir),
                                    "BASELINE_PER_CASE": bench.get("per_case", {}),
                                }),
                    {"phase": "Verify", "label": f"verify {d_id}"}
                )

                if ver and ver.get("status") == "verified" and ver.get("correctness") == "pass":
                    verified_speedup = ver.get("geomean_speedup", 1.0)
                    if verified_speedup > 1.0:
                        runner.log(f"  Verified: {verified_speedup}x speedup")
                        cumulative *= verified_speedup
                        no_improve = 0
                    else:
                        runner.log(f"  Verified but no speedup")
                        no_improve += 1
                else:
                    runner.log(f"  Verification failed")
                    no_improve += 1
            else:
                no_improve += 1

        if no_improve >= max_no_improve:
            runner.log(f"Stopping: {no_improve} rounds with no improvement")
            break

    # === Phase: Report ===
    runner.phase("Report")
    report = runner.agentT(
        role_prompt(runner, "director", "report",
                    "Write the final optimization report.", {
                        "EVAL_DIR": eval_dir,
                        "CUMULATIVE_SPEEDUP": cumulative,
                        "BASELINE_GEOMEAN_MS": baseline_ms,
                        "SKILL_DIR": str(workflow_dir),
                    }),
        {"phase": "Report", "label": "director:report"}
    )

    result = {
        "ok": True,
        "eval_dir": eval_dir,
        "baseline_ms": baseline_ms,
        "cumulative_speedup": cumulative,
        "rounds_completed": round_num,
        "report": report,
    }
    runner.log(f"Done. Cumulative speedup: {cumulative:.3f}x")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GEAK kernel workflow with hermes-agent")
    parser.add_argument("kernel_path", help="Path to kernel directory")
    parser.add_argument("--task", default="optimize", help="Task description")
    parser.add_argument("--budget", type=int, default=4, help="Number of optimization rounds")
    parser.add_argument("--target-language", default="triton", help="Target language")
    parser.add_argument("--gpu-ids", default="0", help="GPU IDs")
    parser.add_argument("--eval-dir", default=None, help="Override eval directory")
    parser.add_argument("--model", default=None, help="Model override for hermes")
    args = parser.parse_args()

    result = run_kernel_workflow(
        kernel_path=args.kernel_path,
        task=args.task,
        budget=args.budget,
        target_language=args.target_language,
        gpu_ids=args.gpu_ids,
        eval_dir=args.eval_dir,
        model=args.model,
    )
    print(json.dumps(result, indent=2))

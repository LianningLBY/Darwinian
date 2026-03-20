"""
代码执行沙箱 (code_execute)

优先在 Docker 容器中隔离执行。若 Docker 不可用，自动降级为本地 subprocess 执行。

安全机制（Docker 模式）：
- CPU/内存限制：0.5 CPU、512MB RAM
- 执行超时：300 秒
- 只读挂载数据目录

降级模式（subprocess）：
- Docker 不可用时自动启用
- 在本地 Python 环境中直接运行，受操作系统进程隔离保护
- 同样有超时控制
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import docker
from docker.errors import DockerException

from darwinian.state import ExperimentCode, ExperimentResult


# Docker 镜像（预装常用科研库）
SANDBOX_IMAGE = "darwinian-sandbox:latest"
FALLBACK_IMAGE = "python:3.11-slim"

# 资源限制
CPU_QUOTA = 50000    # 0.5 CPU
MEM_LIMIT = "512m"
TIMEOUT_SECONDS = 300


def code_execute(
    experiment_code: ExperimentCode,
    mode: str = "full",  # "full" | "poison"
    data_dir: str | None = None,
) -> ExperimentResult:
    """
    执行实验代码。优先使用 Docker 沙箱，不可用时降级为本地 subprocess。
    """
    # 先尝试 Docker
    try:
        client = docker.from_env()
        client.ping()   # 确认 daemon 在线
        return _execute_docker(client, experiment_code, mode, data_dir)
    except DockerException:
        # Docker 不可用，降级为本地 subprocess
        return _execute_subprocess(experiment_code, mode, data_dir)


# ---------------------------------------------------------------------------
# Docker 执行路径
# ---------------------------------------------------------------------------

def _execute_docker(
    client,
    experiment_code: ExperimentCode,
    mode: str,
    data_dir: str | None,
) -> ExperimentResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_code_files(tmpdir, experiment_code, mode)

        volumes = {tmpdir: {"bind": "/workspace", "mode": "ro"}}
        if data_dir and Path(data_dir).exists():
            volumes[data_dir] = {"bind": "/data", "mode": "ro"}

        entrypoint = "run_poison.py" if mode == "poison" else "run_full.py"
        pip_install = _build_pip_command(experiment_code.requirements)
        run_command = f"{pip_install} && python /workspace/{entrypoint}"

        stdout_text = ""
        stderr_text = ""

        try:
            container = client.containers.run(
                image=SANDBOX_IMAGE,
                command=f"bash -c '{run_command}'",
                volumes=volumes,
                mem_limit=MEM_LIMIT,
                cpu_quota=CPU_QUOTA,
                remove=True,
                stdout=True,
                stderr=True,
                detach=False,
                timeout=TIMEOUT_SECONDS,
            )
            stdout_text = container.decode("utf-8") if isinstance(container, bytes) else str(container)

        except docker.errors.ContainerError as e:
            stderr_text = e.stderr.decode("utf-8") if isinstance(e.stderr, bytes) else str(e.stderr)
            stdout_text = e.stdout.decode("utf-8") if e.stdout and isinstance(e.stdout, bytes) else ""
        except Exception as e:
            stderr_text = f"Docker 执行异常: {type(e).__name__}: {e}"

        baseline_metrics, proposed_metrics = _parse_metrics(stdout_text)
        return ExperimentResult(
            stdout=stdout_text,
            stderr=stderr_text,
            baseline_metrics=baseline_metrics,
            proposed_metrics=proposed_metrics,
        )


# ---------------------------------------------------------------------------
# 本地 subprocess 降级路径
# ---------------------------------------------------------------------------

def _execute_subprocess(
    experiment_code: ExperimentCode,
    mode: str,
    data_dir: str | None,
) -> ExperimentResult:
    """Docker 不可用时，在本地 Python 进程中执行代码（受超时控制）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_code_files(tmpdir, experiment_code, mode)

        # 安装依赖
        if experiment_code.requirements:
            pkgs = " ".join(experiment_code.requirements)
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q"] + experiment_code.requirements,
                    timeout=120,
                    capture_output=True,
                )
            except Exception:
                pass  # 安装失败继续尝试运行，由执行阶段产生的错误让诊断师处理

        entrypoint = "run_poison.py" if mode == "poison" else "run_full.py"
        entry_path = os.path.join(tmpdir, entrypoint)

        stdout_text = ""
        stderr_text = ""

        try:
            proc = subprocess.run(
                [sys.executable, entry_path],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                cwd=tmpdir,
                env={**os.environ, "DATA_DIR": data_dir or ""},
            )
            stdout_text = proc.stdout
            stderr_text = proc.stderr
        except subprocess.TimeoutExpired:
            stderr_text = f"执行超时（>{TIMEOUT_SECONDS}秒）"
        except Exception as e:
            stderr_text = f"subprocess 执行异常: {type(e).__name__}: {e}"

        baseline_metrics, proposed_metrics = _parse_metrics(stdout_text)
        return ExperimentResult(
            stdout=stdout_text,
            stderr=stderr_text,
            baseline_metrics=baseline_metrics,
            proposed_metrics=proposed_metrics,
        )


# ---------------------------------------------------------------------------
# 公共工具函数
# ---------------------------------------------------------------------------

def _write_code_files(tmpdir: str, code: ExperimentCode, mode: str) -> None:
    """将代码写入临时目录"""
    with open(os.path.join(tmpdir, "dataset_loader.py"), "w") as f:
        f.write(code.dataset_loader_code)

    if mode == "poison":
        with open(os.path.join(tmpdir, "run_poison.py"), "w") as f:
            f.write(_build_poison_runner(code))
    else:
        with open(os.path.join(tmpdir, "baseline.py"), "w") as f:
            f.write(code.baseline_code)
        with open(os.path.join(tmpdir, "proposed.py"), "w") as f:
            f.write(code.proposed_code)
        with open(os.path.join(tmpdir, "run_full.py"), "w") as f:
            f.write(_build_full_runner())


def _build_full_runner() -> str:
    return """\
import subprocess, sys

print("=== Running baseline ===")
result = subprocess.run([sys.executable, "baseline.py"], capture_output=True, text=True, timeout=150)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)

print("=== Running proposed ===")
result = subprocess.run([sys.executable, "proposed.py"], capture_output=True, text=True, timeout=150)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)
"""


def _build_poison_runner(code: ExperimentCode) -> str:
    return f"""\
# Robustness test runner
{code.robustness_code}
"""


def _build_pip_command(requirements: list[str]) -> str:
    if not requirements:
        return "true"
    pkgs = " ".join(requirements)
    return f"pip install -q {pkgs}"


def _parse_metrics(stdout: str) -> tuple[dict, dict]:
    """从 stdout 中提取 {"model": "baseline"|"proposed", "metrics": {...}} 行"""
    baseline: dict = {}
    proposed: dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if obj.get("model") == "baseline":
                baseline = obj.get("metrics", {})
            elif obj.get("model") == "proposed":
                proposed = obj.get("metrics", {})
        except json.JSONDecodeError:
            continue
    return baseline, proposed

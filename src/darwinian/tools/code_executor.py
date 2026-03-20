"""
代码执行沙箱 (code_execute)

负责在隔离的 Docker 容器中执行 LLM 生成的 Python 代码。

安全机制：
- 使用 Docker 容器完全隔离执行环境
- CPU/内存限制：0.5 CPU、512MB RAM
- 执行超时：300 秒
- 禁止网络访问（--network none）
- 只读挂载数据目录
"""

from __future__ import annotations

import json
import os
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
    在 Docker 沙箱中执行实验代码。

    Args:
        experiment_code: 包含 baseline/proposed/dataset_loader 代码的对象
        mode: "full" 运行完整实验；"poison" 运行毒药数据测试
        data_dir: 主机上数据目录的路径（只读挂载）

    Returns:
        ExperimentResult，包含 stdout、stderr 和解析出的指标
    """
    try:
        client = docker.from_env()
    except DockerException as e:
        return ExperimentResult(
            stdout="",
            stderr=f"Docker 连接失败: {e}\n请确保 Docker Desktop 正在运行。",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # 写入代码文件
        _write_code_files(tmpdir, experiment_code, mode)

        # 构建 volumes
        volumes = {}
        if data_dir and Path(data_dir).exists():
            volumes[data_dir] = {"bind": "/data", "mode": "ro"}

        # 确定执行入口脚本
        entrypoint = "run_poison.py" if mode == "poison" else "run_full.py"

        # 构建 tar 包供 Docker 使用
        tar_path = os.path.join(tmpdir, "code.tar")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(tmpdir, arcname=".")

        start_time = time.time()
        stdout_text = ""
        stderr_text = ""

        try:
            # 安装依赖 + 执行
            pip_install = _build_pip_command(experiment_code.requirements)
            run_command = f"{pip_install} && python /workspace/{entrypoint}"

            container = client.containers.run(
                image=SANDBOX_IMAGE,
                command=f"bash -c '{run_command}'",
                volumes={tmpdir: {"bind": "/workspace", "mode": "ro"}, **volumes},
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
            stderr_text = f"执行异常: {type(e).__name__}: {e}"

        elapsed = time.time() - start_time

        # 尝试从 stdout 解析指标 JSON
        baseline_metrics, proposed_metrics = _parse_metrics(stdout_text)

        return ExperimentResult(
            stdout=stdout_text,
            stderr=stderr_text,
            baseline_metrics=baseline_metrics,
            proposed_metrics=proposed_metrics,
        )


def _write_code_files(tmpdir: str, code: ExperimentCode, mode: str) -> None:
    """将代码写入临时目录"""
    # 数据加载代码（普通模式或毒药代码）
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
# Poison / Robustness test runner
{code.poison_code}
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

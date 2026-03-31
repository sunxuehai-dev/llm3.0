"""
在启动高 GPU 占用任务前检查可用显存（通过 nvidia-smi，无需 PyTorch）。

用于 llmfactory 训练、合并、推理启动等接口；未配置阈值时不做任何检查。
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def query_nvidia_gpu_free_mib(gpu_index: int = 0) -> Optional[int]:
    """
    查询指定 GPU 当前可用显存（MiB，与 nvidia-smi 一致）。
    无法查询时返回 None（无 nvidia-smi、非 NVIDIA 环境、命令失败等）。
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        r = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            logger.warning(
                "nvidia-smi 退出码 %s: %s",
                r.returncode,
                (r.stderr or "")[:300],
            )
            return None
        lines = [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]
        if not lines:
            return None
        return int(lines[0])
    except (ValueError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning("查询 GPU 空闲显存失败: %s", e)
        return None


def _truthy_env(val: str | None) -> bool:
    if val is None:
        return False
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def llmfactory_check_min_free_vram(
    *,
    min_free_mib: int,
    gpu_index: int = 0,
    strict_when_unknown: bool = False,
) -> None:
    """
    若 min_free_mib > 0，则要求指定 GPU 当前空闲显存 >= 该值（MiB），否则抛出 HTTPException 503。

    strict_when_unknown: 为 True 且无法查询显存时，视为不满足条件并拒绝（用于必须保证有 GPU 监控的环境）。
    """
    if min_free_mib <= 0:
        return

    from fastapi import HTTPException

    free = query_nvidia_gpu_free_mib(gpu_index)
    if free is None:
        if strict_when_unknown:
            raise HTTPException(
                status_code=503,
                detail=(
                    "已启用显存下限检查且为严格模式，但无法通过 nvidia-smi 获取 GPU 空闲显存。"
                    "请确认已安装 NVIDIA 驱动并可在 PATH 中执行 nvidia-smi，"
                    "或将 LLMFACTORY_GPU_VRAM_GUARD_STRICT 设为 false。"
                ),
            )
        logger.debug("未查询到 GPU 空闲显存，跳过显存下限检查（gpu_index=%s）", gpu_index)
        return

    if free < min_free_mib:
        raise HTTPException(
            status_code=503,
            detail=(
                f"GPU {gpu_index} 当前可用显存约 {free} MiB，低于服务端要求的最低空闲 {min_free_mib} MiB，"
                f"已拒绝本次操作。请结束其他占用显存的进程或调低 LLMFACTORY_MIN_FREE_VRAM_MIB 后再试。"
            ),
        )

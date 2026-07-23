"""视频预处理工具 - 时长检测与分段切割。

检测视频时长，若超过阈值则用 ffmpeg 按指定时长切割为多个片段。
"""

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# 默认阈值：超过 30 分钟的视频需要分段
DEFAULT_MAX_DURATION_SEC = 30 * 60
# 默认分段时长：每段 20 分钟
DEFAULT_SEGMENT_DURATION_SEC = 10 * 60


def get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）。

    Args:
        video_path: 视频文件路径。

    Returns:
        视频时长（秒）。

    Raises:
        RuntimeError: 无法获取视频时长。
    """
    # 先检查文件是否存在
    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"视频文件不存在: {video_path}\n"
            f"请确认文件路径是否正确。"
        )

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"无法获取视频时长: {video_path}, {e}") from e


def _split_video_sync(
    video_path: str,
    output_dir: str,
    segment_duration_sec: float,
) -> list[dict]:
    """同步执行视频分段切割。

    使用 ffmpeg -c copy 无损切割，速度极快。

    Args:
        video_path: 源视频文件路径。
        output_dir: 分段视频输出目录。
        segment_duration_sec: 每段时长（秒）。

    Returns:
        分段信息列表: [{"index": 0, "path": "...", "start": 0, "duration": 1200}, ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    total_duration = get_video_duration(video_path)
    video_ext = Path(video_path).suffix
    video_stem = Path(video_path).stem

    segments = []
    start = 0.0
    index = 0

    while start < total_duration:
        remaining = total_duration - start
        seg_dur = min(segment_duration_sec, remaining)

        seg_filename = f"{video_stem}_part{index:03d}{video_ext}"
        seg_path = os.path.join(output_dir, seg_filename)

        logger.info(
            "切割片段 %d: start=%.1fs, duration=%.1fs -> %s",
            index, start, seg_dur, seg_path,
        )

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(seg_dur),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            seg_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error("ffmpeg 切割失败: %s", result.stderr[-500:])
            raise RuntimeError(
                f"视频切割片段 {index} 失败: {result.stderr[-200:]}"
            )

        segments.append({
            "index": index,
            "path": seg_path,
            "start": round(start, 2),
            "duration": round(seg_dur, 2),
        })

        start += segment_duration_sec
        index += 1

    logger.info("视频切割完成: 共 %d 个片段", len(segments))
    return segments


async def preprocess_video(
    video_path: str,
    output_dir: str = "output",
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    segment_duration_sec: float = DEFAULT_SEGMENT_DURATION_SEC,
) -> list[dict]:
    """视频预处理：检测时长，必要时分段切割。

    Args:
        video_path: 视频文件的绝对路径。
        output_dir: 输出目录。
        max_duration_sec: 超过此时长（秒）则分段，默认 30 分钟。
        segment_duration_sec: 每段时长（秒），默认 20 分钟。

    Returns:
        片段列表。短视频返回单元素列表（原始文件）；
        长视频返回多元素列表（切割后的片段）。
    """
    video_path = str(Path(video_path).expanduser().resolve())
    duration = await asyncio.to_thread(get_video_duration, video_path)

    logger.info(
        "视频时长: %.1fs (%.1f分钟), 分段阈值: %.1fs (%.1f分钟)",
        duration, duration / 60,
        max_duration_sec, max_duration_sec / 60,
    )

    if duration <= max_duration_sec:
        logger.info("视频时长未超过阈值，无需分段")
        return [{
            "index": 0,
            "path": video_path,
            "start": 0.0,
            "duration": round(duration, 2),
        }]

    # 需要分段切割
    logger.info(
        "视频时长超过 %.0f 分钟，将按每 %.0f 分钟切割",
        max_duration_sec / 60,
        segment_duration_sec / 60,
    )

    segments_dir = os.path.join(output_dir, "segments")
    segments = await asyncio.to_thread(
        _split_video_sync,
        video_path, segments_dir, segment_duration_sec,
    )

    return segments

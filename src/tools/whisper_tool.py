"""Whisper 语音转文字工具封装。

将视频文件中的语音内容转录为带时间戳的字幕数据。
所有阻塞操作（ffmpeg、Whisper 推理）通过 asyncio.to_thread 在线程中执行，
确保完整转录完成后才返回结果。
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from agentscope.tool import ToolResponse
from agentscope.message import TextBlock

logger = logging.getLogger(__name__)

# Whisper 转录失败最大重试次数
_MAX_TRANSCRIBE_RETRIES = 3
# 重试间隔（秒）
_TRANSCRIBE_RETRY_INTERVAL = 3 * 60  # 3 分钟

# 全局缓存已加载的 Whisper 模型，避免重复加载
_whisper_model = None
_whisper_model_size = None


def _load_whisper_model(model_size: str = "medium"):
    """加载 Whisper 模型（带缓存）。"""
    global _whisper_model, _whisper_model_size

    if _whisper_model is not None and _whisper_model_size == model_size:
        return _whisper_model

    import whisper

    logger.info("正在加载 Whisper 模型: %s ...", model_size)
    _whisper_model = whisper.load_model(model_size)
    _whisper_model_size = model_size
    logger.info("Whisper 模型加载完成")
    return _whisper_model


def _extract_audio(video_path: str, audio_path: str) -> None:
    """使用 ffmpeg 从视频文件中提取音频。"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                 # 不处理视频
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", "16000",        # 16kHz 采样率（Whisper 要求）
        "-ac", "1",            # 单声道
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 音频提取失败: {result.stderr}"
        )


def _do_transcribe(video_path: str, model_size: str) -> dict:
    """同步执行完整的转录流程（音频提取 + Whisper 推理），确保全部完成后返回。"""
    # 1. 提取音频到 output 目录（保留中间文件供检查）
    video_name = Path(video_path).stem
    audio_dir = os.path.join("output", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, f"{video_name}.wav")

    try:
        logger.info("正在从视频中提取完整音频...")
        _extract_audio(video_path, audio_path)
        # 检查音频文件大小
        audio_size = os.path.getsize(audio_path)
        logger.info("音频提取完成: %s (%.2f MB)", audio_path, audio_size / 1024 / 1024)

    except RuntimeError:
        logger.warning("首次音频提取失败，尝试重新提取...")
        try:
            _extract_audio(video_path, audio_path)
        except RuntimeError as e:
            return {"error": f"音频提取失败: {e}"}

    # 2. 使用 Whisper 进行完整转录（支持失败重试）
    result = None
    last_error = None
    for attempt in range(1, _MAX_TRANSCRIBE_RETRIES + 1):
        try:
            model = _load_whisper_model(model_size)
            logger.info("正在使用 Whisper 转录音频（请等待完整转录）...")
            result = model.transcribe(audio_path)
            break
        except Exception as e:
            last_error = e
            logger.warning(
                "Whisper 转录失败 (第 %d/%d 次尝试): %s",
                attempt, _MAX_TRANSCRIBE_RETRIES, e,
            )
            # 非 base 模型：第一次尝试时先尝试降级到 base
            if attempt == 1 and model_size != "base":
                logger.info("降级使用 base 模型重试...")
                try:
                    model = _load_whisper_model("base")
                    result = model.transcribe(audio_path)
                    break
                except Exception as e2:
                    last_error = e2
                    logger.warning("base 模型也失败: %s", e2)
            # 若还有重试机会，等待后重试
            if attempt < _MAX_TRANSCRIBE_RETRIES:
                logger.info(
                    "等待 %d 秒后重试转录...",
                    _TRANSCRIBE_RETRY_INTERVAL,
                )
                time.sleep(_TRANSCRIBE_RETRY_INTERVAL)
    else:
        # 所有重试均失败
        return {"error": f"Whisper 转录失败（重试 {_MAX_TRANSCRIBE_RETRIES} 次）: {last_error}"}

    # 保留音频文件供检查
    if audio_path and os.path.exists(audio_path):
        logger.info("中间音频文件已保留: %s", audio_path)

    # 3. 格式化输出
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "id": seg["id"],
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    output = {
        "segments": segments,
        "language": result.get("language", "unknown"),
    }

    logger.info(
        "转录完成: 语言=%s, 共 %d 个片段",
        output["language"],
        len(segments),
    )

    return output


async def transcribe_video(
    video_path: str,
    model_size: str = "medium",
) -> ToolResponse:
    """将视频文件中的语音转录为带时间戳的字幕数据。

    在线程池中执行完整的转录流程（音频提取 + Whisper 推理），
    确保全部转录完成后才返回结果，不阻塞事件循环。

    Args:
        video_path: 视频文件的绝对路径或相对路径。
        model_size: Whisper 模型大小，可选 tiny/base/small/medium/large。
            默认为 medium。

    Returns:
        ToolResponse: 包含 JSON 格式字幕数据的响应。
    """
    video_path = str(Path(video_path).expanduser().resolve())
    if not os.path.isfile(video_path):
        return ToolResponse(
            content=[TextBlock(
                type="text",
                text=json.dumps(
                    {"error": f"视频文件不存在: {video_path}"},
                    ensure_ascii=False,
                ),
            )],
        )

    # 在线程池中执行完整的同步转录流程，等待全部完成
    output = await asyncio.to_thread(_do_transcribe, video_path, model_size)

    return ToolResponse(
        content=[TextBlock(
            type="text",
            text=json.dumps(output, ensure_ascii=False, indent=2),
        )],
    )

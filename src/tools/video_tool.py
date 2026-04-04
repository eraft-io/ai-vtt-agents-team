"""视频处理工具封装。

提供视频关键帧提取功能，基于 OpenCV 场景切换检测。
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock

logger = logging.getLogger(__name__)

# 去重用的统一缩放尺寸（宽x高），缩小后比较速度快且抗噪
_DEDUP_SIZE = (128, 128)
# 像素级 MSE 阈值，低于此值视为相同帧（0-255 灰度范围）
_DEDUP_MSE_THRESHOLD = 50.0


def _frame_to_thumb(frame: np.ndarray) -> np.ndarray:
    """将帧缩放为统一尺寸的灰度缩略图，用于去重比较。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, _DEDUP_SIZE, interpolation=cv2.INTER_AREA)


def _is_duplicate_frame(
    thumb: np.ndarray,
    saved_thumbs: list[np.ndarray],
) -> bool:
    """检查缩略图是否与已保存的任一帧重复（基于像素 MSE）。"""
    for saved in saved_thumbs:
        mse = float(np.mean((thumb.astype(np.float32) - saved.astype(np.float32)) ** 2))
        if mse < _DEDUP_MSE_THRESHOLD:
            return True
    return False


def _compute_frame_diff(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """计算两帧之间的差异值（基于直方图差异）。"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return float(diff)


def _extract_keyframes_by_scene(
    video_path: str,
    output_dir: str,
    scene_threshold: float = 0.3,
    min_interval_sec: float = 5.0,
) -> list[dict]:
    """基于场景切换检测提取关键帧。

    Args:
        video_path: 视频文件路径。
        output_dir: 关键帧图片输出目录。
        scene_threshold: 场景切换检测阈值 (0-1)，值越小越敏感。
        min_interval_sec: 相邻关键帧的最小时间间隔（秒）。

    Returns:
        关键帧信息列表。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(f"视频文件信息异常: fps={fps}, frames={total_frames}")

    os.makedirs(output_dir, exist_ok=True)

    keyframes = []
    saved_thumbs = []  # 已保存关键帧的缩略图，用于像素级去重
    prev_frame = None
    last_keyframe_time = -min_interval_sec  # 确保第一帧可以被选中

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        if prev_frame is not None:
            diff = _compute_frame_diff(prev_frame, frame)

            if (
                diff > scene_threshold
                and (current_time - last_keyframe_time) >= min_interval_sec
            ):
                # 检测到场景切换，用像素 MSE 检查是否与已保存帧重复
                thumb = _frame_to_thumb(frame)
                if _is_duplicate_frame(thumb, saved_thumbs):
                    logger.debug("跳过重复帧(MSE): time=%.2fs", current_time)
                else:
                    timestamp_ms = int(current_time * 1000)
                    image_filename = f"frame_{timestamp_ms:08d}.jpg"
                    image_path = os.path.join(output_dir, image_filename)

                    cv2.imwrite(image_path, frame)
                    saved_thumbs.append(thumb)

                    keyframes.append({
                        "timestamp": round(current_time, 2),
                        "image_path": image_path,
                    })

                    last_keyframe_time = current_time
                    logger.debug(
                        "关键帧: time=%.2fs, diff=%.4f, path=%s",
                        current_time, diff, image_path,
                    )
        else:
            # 保存第一帧作为关键帧
            thumb = _frame_to_thumb(frame)
            image_path = os.path.join(output_dir, "frame_00000000.jpg")
            cv2.imwrite(image_path, frame)
            saved_thumbs.append(thumb)
            keyframes.append({
                "timestamp": 0.0,
                "image_path": image_path,
            })
            last_keyframe_time = 0.0

        prev_frame = frame.copy()
        frame_idx += 1

    cap.release()
    return keyframes


def _extract_keyframes_by_interval(
    video_path: str,
    output_dir: str,
    interval_sec: float = 30.0,
) -> list[dict]:
    """按固定时间间隔提取关键帧（降级方案）。

    Args:
        video_path: 视频文件路径。
        output_dir: 关键帧图片输出目录。
        interval_sec: 截取间隔（秒）。

    Returns:
        关键帧信息列表。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"视频 FPS 异常: {fps}")

    os.makedirs(output_dir, exist_ok=True)

    duration = total_frames / fps
    keyframes = []
    saved_thumbs = []  # 已保存关键帧的缩略图，用于像素级去重

    current_time = 0.0
    while current_time < duration:
        frame_idx = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret:
            break

        # 用像素 MSE 与所有已保存帧比较，重复则跳过
        thumb = _frame_to_thumb(frame)
        if _is_duplicate_frame(thumb, saved_thumbs):
            logger.debug("跳过重复帧(MSE): time=%.2fs", current_time)
            current_time += interval_sec
            continue

        timestamp_ms = int(current_time * 1000)
        image_filename = f"frame_{timestamp_ms:08d}.jpg"
        image_path = os.path.join(output_dir, image_filename)

        cv2.imwrite(image_path, frame)
        saved_thumbs.append(thumb)

        keyframes.append({
            "timestamp": round(current_time, 2),
            "image_path": image_path,
        })

        current_time += interval_sec

    cap.release()
    return keyframes


def _do_extract_keyframes(
    video_path: str,
    output_dir: str,
    scene_threshold: float,
    min_interval_sec: float,
) -> dict:
    """同步执行完整的关键帧提取流程，确保全部完成后返回。"""
    try:
        keyframes = _extract_keyframes_by_scene(
            video_path, output_dir, scene_threshold, min_interval_sec,
        )
    except RuntimeError as e:
        logger.warning("场景检测方式提取关键帧失败: %s", e)
        return {"error": f"视频处理失败: {e}"}

    # 如果场景检测提取不足 3 帧，用固定间隔补充
    if len(keyframes) < 3:
        logger.warning(
            "场景检测仅提取到 %d 个关键帧，降级为按 30 秒间隔截取",
            len(keyframes),
        )
        try:
            keyframes = _extract_keyframes_by_interval(
                video_path, output_dir, interval_sec=30.0,
            )
        except RuntimeError as e:
            return {"error": f"关键帧提取失败: {e}"}

    logger.info("关键帧提取完成: 共 %d 个关键帧", len(keyframes))
    return {"keyframes": keyframes}


async def extract_keyframes(
    video_path: str,
    output_dir: str = "output/keyframes",
    scene_threshold: float = 0.08,
    min_interval_sec: float = 5.0,
) -> ToolResponse:
    """从视频中提取关键帧图片。

    基于场景切换检测自动提取关键帧。如果场景检测无法提取到关键帧，
    则降级为按固定时间间隔（每 30 秒）截取。

    Args:
        video_path: 视频文件的绝对路径或相对路径。
        output_dir: 关键帧图片输出目录，默认为 output/keyframes。
        scene_threshold: 场景切换检测阈值 (0-1)，默认 0.3。
        min_interval_sec: 相邻关键帧的最小时间间隔（秒），默认 5。

    Returns:
        ToolResponse: 包含 JSON 格式关键帧数据的响应，结构为:
            {
                "keyframes": [
                    {"timestamp": 5.2, "image_path": "output/keyframes/frame_00005200.jpg"}
                ]
            }
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

    # 在线程池中执行完整的同步关键帧提取流程，等待全部完成
    output = await asyncio.to_thread(
        _do_extract_keyframes,
        video_path, output_dir, scene_threshold, min_interval_sec,
    )

    return ToolResponse(
        content=[TextBlock(
            type="text",
            text=json.dumps(output, ensure_ascii=False, indent=2),
        )],
    )

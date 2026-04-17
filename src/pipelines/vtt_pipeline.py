"""VTT Pipeline - 视频转录翻译工作流编排。

编排多个 Agent 的执行顺序：
  预处理:  检测视频时长，超过 30 分钟则切割为 20 分钟片段
  每个片段:
    阶段一（并行）: 转录 + 关键帧提取
    阶段二（串行）: 总结 -> 翻译 -> 校对
  最终: 合并所有片段的文章
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

from src.agents import (
    create_summarizer_agent,
    create_translator_agent,
    create_proofreader_agent,
)
from src.tools.whisper_tool import transcribe_video
from src.tools.video_tool import extract_keyframes
from src.tools.video_splitter import preprocess_video
from src.pipelines.state_db import StateDB

logger = logging.getLogger(__name__)

# 支持扫描的视频文件扩展名
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".ts"}


def sanitize_filename(name: str) -> str:
    """将视频文件名清洗为合法的输出文件名。

    仅保留英文字母、数字和连字符，去除空格及无意义字符，
    连续连字符合并，首尾连字符去除。
    """
    # 去掉扩展名（如果传入了）
    stem = Path(name).stem if "." in name else name
    # 只保留英文字母、数字、连字符
    cleaned = re.sub(r"[^a-zA-Z0-9-]", "-", stem)
    # 合并连续连字符
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    # 全部转小写
    cleaned = cleaned.lower()
    return cleaned or "untitled"


def scan_video_dir(directory: str) -> list[str]:
    """扫描目录下的所有视频文件（非递归），返回绝对路径列表。"""
    directory = str(Path(directory).expanduser().resolve())
    if not os.path.isdir(directory):
        raise ValueError(f"目录不存在: {directory}")

    videos: list[str] = []
    for fname in sorted(os.listdir(directory)):
        ext = Path(fname).suffix.lower()
        if ext in VIDEO_EXTENSIONS:
            videos.append(os.path.join(directory, fname))

    return videos


# ---------------------------------------------------------------------------
# Checkpoint state helpers
# ---------------------------------------------------------------------------
_STATE_FILE = "pipeline_state.json"


def _load_state(project_dir: str) -> dict | None:
    """加载 pipeline_state.json，不存在则返回 None。"""
    path = os.path.join(project_dir, _STATE_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(project_dir: str, state: dict) -> None:
    """持久化 pipeline state 到 project_dir/pipeline_state.json。"""
    os.makedirs(project_dir, exist_ok=True)
    path = os.path.join(project_dir, _STATE_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")


class VTTPipeline:
    """视频转录翻译 Pipeline。

    编排多个 Agent 完成视频 -> 字幕 -> 总结 -> 翻译 -> 校对的完整流程。
    支持断点续传：每完成一个分段即保存状态，失败后重启可跳过已完成分段。
    """

    def __init__(
        self,
        model_name: str = "qwen3.6-plus",
        api_key: str | None = None,
        whisper_model_size: str = "medium",
        scene_threshold: float = 0.08,
        min_interval_sec: float = 5.0,
        output_dir: str = "output",
        max_duration_sec: float = 30 * 60,
        segment_duration_sec: float = 10 * 60,
    ) -> None:
        """初始化 Pipeline。

        Args:
            model_name: DashScope 模型名称。
            api_key: DashScope API Key。
            whisper_model_size: Whisper 模型大小。
            scene_threshold: 关键帧场景切换阈值。
            min_interval_sec: 关键帧最小间隔。
            output_dir: 输出目录。
            max_duration_sec: 视频超过此时长（秒）则分段，默认 30 分钟。
            segment_duration_sec: 每段时长（秒），默认 20 分钟。
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model_name = model_name
        self.output_dir = output_dir
        self.whisper_model_size = whisper_model_size
        self.scene_threshold = scene_threshold
        self.min_interval_sec = min_interval_sec
        self.max_duration_sec = max_duration_sec
        self.segment_duration_sec = segment_duration_sec

        # 创建 LLM Agent（总结、翻译、校对）
        self.summarizer = create_summarizer_agent(
            model_name=model_name,
            api_key=self.api_key,
        )
        self.translator = create_translator_agent(
            model_name=model_name,
            api_key=self.api_key,
        )
        self.proofreader = create_proofreader_agent(
            model_name=model_name,
            api_key=self.api_key,
        )

        logger.info("VTTPipeline 初始化完成")

    async def run(
        self,
        video_path: str,
        target_language: str = "中文",
    ) -> tuple[str, str]:
        """执行完整的视频转录翻译流程。

        包含视频预处理（时长检测、分段切割）和逐段处理。

        Args:
            video_path: 视频文件路径。
            target_language: 目标翻译语言。

        Returns:
            (最终校对后的 Markdown 文章内容, 输出文件路径)。
        """
        video_path = str(Path(video_path).expanduser().resolve())
        video_name = Path(video_path).stem

        logger.info("=" * 60)
        logger.info("开始处理视频: %s", video_path)
        logger.info("目标语言: %s", target_language)
        logger.info("=" * 60)

        # ============================================================
        # 预处理：检测视频时长，必要时分段切割
        # ============================================================
        logger.info("[预处理] 检测视频时长并判断是否需要分段")
        segments = await preprocess_video(
            video_path=video_path,
            output_dir=self.output_dir,
            max_duration_sec=self.max_duration_sec,
            segment_duration_sec=self.segment_duration_sec,
        )
        logger.info("[预处理] 完成: 共 %d 个片段", len(segments))

        # ============================================================
        # 用视频文件名作为输出目录名
        # ============================================================
        topic_slug = sanitize_filename(video_name)
        logger.info("输出目录: %s", topic_slug)

        # 转录第一个片段（后续处理时复用，避免重复转录）
        first_segment_path = segments[0]["path"]
        logger.info("[转录] 转录第一个片段")
        first_resp = await transcribe_video(
            video_path=first_segment_path,
            model_size=self.whisper_model_size,
        )
        first_transcribe = "".join(
            block["text"] for block in first_resp.content
            if block.get("type") == "text"
        )

        # 输出到视频文件所在目录下的子目录
        video_parent_dir = str(Path(video_path).parent)
        project_dir = os.path.join(video_parent_dir, topic_slug)
        keyframes_dir = os.path.join(project_dir, "keyframes")
        os.makedirs(keyframes_dir, exist_ok=True)

        # ============================================================
        # 断点续传：加载或初始化状态
        # ============================================================
        state = _load_state(project_dir)
        total = len(segments)

        if state and state.get("video_path") == video_path:
            seg_statuses = state.get("segments", {})
            completed_count = sum(
                1 for s in seg_statuses.values() if s == "completed"
            )
            if completed_count > 0:
                logger.info(
                    "[断点续传] 发现已有进度: %d/%d 个片段已完成，从第 %d 个片段继续",
                    completed_count, total, completed_count + 1,
                )
        else:
            # 初始化新状态
            seg_statuses = {}
            state = {
                "video_path": video_path,
                "target_language": target_language,
                "topic_slug": topic_slug,
                "total_segments": total,
                "segments": seg_statuses,
            }
            _save_state(project_dir, state)

        # ============================================================
        # 逐段处理：转录 + 关键帧 + 总结 + 翻译 + 校对
        # ============================================================
        all_segment_articles: list[str | None] = [None] * total

        # 提前确定输出路径，每完成一段即实时写入
        clean_name = sanitize_filename(video_name)
        output_filename = f"{clean_name}.md"
        output_path = os.path.join(project_dir, output_filename)

        for seg in segments:
            seg_idx = seg["index"]
            seg_path = seg["path"]
            seg_key = str(seg_idx)

            # ---- 跳过已完成的片段，从 md 文件加载 ----
            if seg_statuses.get(seg_key) == "completed":
                seg_md_path = os.path.join(
                    project_dir, f"part{seg_idx:03d}_{target_language}.md",
                )
                if os.path.isfile(seg_md_path):
                    with open(seg_md_path, "r", encoding="utf-8") as f:
                        all_segment_articles[seg_idx] = f.read()
                    logger.info(
                        "[断点续传] 片段 %d/%d 已完成，加载缓存: %s",
                        seg_idx + 1, total, seg_md_path,
                    )
                    continue
                else:
                    logger.warning(
                        "[断点续传] 片段 %d/%d 标记为完成但 md 文件不存在，重新处理",
                        seg_idx + 1, total,
                    )

            logger.info("=" * 60)
            logger.info(
                "处理片段 %d/%d: %s (start=%.1fs, dur=%.1fs)",
                seg_idx + 1, total, seg_path,
                seg["start"], seg["duration"],
            )
            logger.info("=" * 60)

            # 发送结构化进度信息（Web 端可展示进度条）
            logger.info(
                "segment_progress",
                extra={"progress": {
                    "current": seg_idx + 1,
                    "total": total,
                    "video_name": video_name,
                    "stage": "processing",
                }},
            )

            try:
                seg_article = await self._process_segment(
                    seg_path=seg_path,
                    seg_index=seg_idx,
                    total_segments=total,
                    project_dir=project_dir,
                    keyframes_dir=keyframes_dir,
                    target_language=target_language,
                    cached_transcribe=first_transcribe if seg_idx == 0 else None,
                )
            except Exception:
                logger.exception("片段 %d/%d 处理失败", seg_idx + 1, total)
                seg_article = None

            if seg_article:
                all_segment_articles[seg_idx] = seg_article
                # 标记完成并持久化状态
                seg_statuses[seg_key] = "completed"
                state["segments"] = seg_statuses
                _save_state(project_dir, state)

                # ---- 实时写入：将已完成的内容立即写入目标文件 ----
                self._flush_to_file(
                    all_segment_articles, output_path, keyframes_dir,
                )
                logger.info(
                    "[实时写入] 片段 %d/%d 已写入: %s",
                    seg_idx + 1, total, output_path,
                )
                logger.info(
                    "segment_done",
                    extra={"progress": {
                        "current": seg_idx + 1,
                        "total": total,
                        "video_name": video_name,
                        "stage": "done",
                    }},
                )
            else:
                seg_statuses[seg_key] = "failed"
                state["segments"] = seg_statuses
                _save_state(project_dir, state)
                logger.error(
                    "片段 %d/%d 处理失败，状态已保存。可重新运行以从此片段继续。",
                    seg_idx + 1, total,
                )
                raise RuntimeError(
                    f"片段 {seg_idx + 1}/{total} 处理失败，已保存进度。"
                    f"重新运行相同视频即可从失败片段继续。"
                )

        # ============================================================
        # 最终：读取已写入的完整文件内容作为返回值
        # ============================================================
        with open(output_path, "r", encoding="utf-8") as f:
            final_article = f.read()

        # 全部完成，更新状态
        state["status"] = "completed"
        state["output_path"] = output_path
        _save_state(project_dir, state)

        logger.info("=" * 60)
        logger.info("处理完成! 共 %d 个片段，文章已保存至: %s", total, output_path)
        logger.info("=" * 60)

        return final_article, output_path

    async def _process_segment(
        self,
        seg_path: str,
        seg_index: int,
        total_segments: int,
        project_dir: str,
        keyframes_dir: str,
        target_language: str,
        cached_transcribe: str | None = None,
    ) -> str | None:
        """处理单个视频片段：转录 + 关键帧 + 总结 + 翻译 + 校对。

        Returns:
            该片段的最终文章内容，或 None 如果失败。
        """
        seg_label = f"[{seg_index + 1}/{total_segments}]"
        video_name = Path(seg_path).stem

        # ============================================================
        # 阶段一: 转录 + 关键帧提取
        # ============================================================
        logger.info("%s 阶段一: 转录 + 关键帧提取", seg_label)

        tmp_kf_dir = os.path.join(
            self.output_dir,
            f"_tmp_keyframes_{hashlib.md5(video_name.encode()).hexdigest()[:6]}_seg{seg_index}",
        )

        if cached_transcribe is not None:
            # 第一段已经转录过，只做关键帧提取
            transcribe_text = cached_transcribe
            keyframe_resp = await extract_keyframes(
                video_path=seg_path,
                output_dir=tmp_kf_dir,
                scene_threshold=self.scene_threshold,
                min_interval_sec=self.min_interval_sec,
            )
        else:
            transcribe_resp, keyframe_resp = await asyncio.gather(
                transcribe_video(
                    video_path=seg_path,
                    model_size=self.whisper_model_size,
                ),
                extract_keyframes(
                    video_path=seg_path,
                    output_dir=tmp_kf_dir,
                    scene_threshold=self.scene_threshold,
                    min_interval_sec=self.min_interval_sec,
                ),
            )
            transcribe_text = "".join(
                block["text"] for block in transcribe_resp.content
                if block.get("type") == "text"
            )

        keyframe_text = "".join(
            block["text"] for block in keyframe_resp.content
            if block.get("type") == "text"
        )

        logger.info("%s 阶段一完成", seg_label)

        # 检查错误
        if self._check_error(transcribe_text):
            logger.error("%s 转录失败: %s", seg_label, transcribe_text)
            return None

        # 移动关键帧到项目目录（带分段前缀防止冲突）
        rename_map = self._move_keyframes(tmp_kf_dir, keyframes_dir, seg_index)
        keyframe_text = self._rewrite_keyframe_paths(
            keyframe_text, keyframes_dir, rename_map,
        )

        # 保存转录原文
        transcribe_path = os.path.join(
            project_dir, f"{video_name}_transcription.json",
        )
        with open(transcribe_path, "w", encoding="utf-8") as f:
            f.write(transcribe_text)
        logger.info("%s 转录原文已保存: %s", seg_label, transcribe_path)

        # ============================================================
        # 阶段二: 总结
        # ============================================================
        logger.info("%s 阶段二: 内容总结", seg_label)

        part_hint = ""
        if total_segments > 1:
            part_hint = f"\n\n注意：这是视频的第 {seg_index + 1}/{total_segments} 部分。\n"

        summarize_msg = Msg(
            name="user",
            role="user",
            content=(
                "请将以下字幕内容总结成分段文章，并在合适位置插入关键帧图片。"
                f"{part_hint}\n\n"
                f"## 字幕数据\n```json\n{transcribe_text}\n```\n\n"
                f"## 关键帧数据\n```json\n{keyframe_text}\n```"
            ),
        )

        summarize_result = await self.summarizer(summarize_msg)
        article = summarize_result.get_text_content()

        if not article or article.strip() == "":
            logger.warning("%s 总结结果为空", seg_label)
            return None

        # ============================================================
        # 阶段三: 翻译
        # ============================================================
        logger.info("%s 阶段三: 翻译为%s", seg_label, target_language)

        translate_msg = Msg(
            name="user",
            role="user",
            content=(
                f"请将以下 Markdown 文章翻译为{target_language}。\n\n"
                f"---\n\n{article}"
            ),
        )

        translate_result = await self.translator(translate_msg)
        translated_article = translate_result.get_text_content()

        # ============================================================
        # 阶段四: 校对
        # ============================================================
        logger.info("%s 阶段四: 校对润色", seg_label)

        proofread_msg = Msg(
            name="user",
            role="user",
            content=(
                "请对以下翻译后的文章进行逐行校对，标注专业术语并润色。\n\n"
                f"---\n\n{translated_article}"
            ),
        )

        proofread_result = await self.proofreader(proofread_msg)
        final_article = proofread_result.get_text_content()

        logger.info("%s 所有阶段完成", seg_label)

        # 保存该片段的独立 md
        seg_article_fixed = self._fix_image_paths(final_article, keyframes_dir)
        seg_md_name = f"part{seg_index:03d}_{target_language}.md"
        seg_md_path = os.path.join(project_dir, seg_md_name)
        with open(seg_md_path, "w", encoding="utf-8") as f:
            f.write(seg_article_fixed)
        logger.info("%s 片段文章已保存: %s", seg_label, seg_md_path)

        return final_article

    async def _extract_topic_slug(self, transcribe_text: str) -> str:
        """从转录内容中用 LLM 提取英文主题名作为目录 slug。"""
        model = OpenAIChatModel(
            model_name=self.model_name,
            api_key=self.api_key,
            stream=False,
            client_kwargs={"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Based on the transcript below, "
                    "generate a short English topic name suitable for a directory name. "
                    "Use lowercase letters, numbers, and hyphens only. "
                    "Keep it concise (2-5 words). "
                    "Return ONLY the slug, nothing else. "
                    "Example: network-address-translation"
                ),
            },
            {
                "role": "user",
                "content": transcribe_text[:2000],  # 取前 2000 字符即可
            },
        ]
        try:
            response = await model(messages)
            slug = "".join(
                block["text"] for block in response.content
                if block.get("type") == "text"
            ).strip().lower()
            # 清理非法字符
            slug = re.sub(r"[^a-z0-9-]", "-", slug)
            slug = re.sub(r"-+", "-", slug).strip("-")
            if slug:
                return slug
        except Exception as e:
            logger.warning("LLM 提取主题失败: %s", e)

        # 降级为视频文件名
        return Path(transcribe_text).stem if False else "untitled"

    @staticmethod
    def _move_keyframes(src_dir: str, dst_dir: str, seg_index: int = 0) -> dict[str, str]:
        """将关键帧从临时目录移动到项目目录，加上分段前缀避免冲突。

        Returns:
            旧文件名 -> 新文件名 的映射表。
        """
        import shutil
        rename_map: dict[str, str] = {}
        if not os.path.isdir(src_dir):
            return rename_map
        prefix = f"part{seg_index:03d}_"
        for fname in os.listdir(src_dir):
            src_path = os.path.join(src_dir, fname)
            if os.path.isfile(src_path):
                new_fname = f"{prefix}{fname}"
                dst_path = os.path.join(dst_dir, new_fname)
                shutil.move(src_path, dst_path)
                rename_map[fname] = new_fname
        # 清理临时目录
        shutil.rmtree(src_dir, ignore_errors=True)
        return rename_map

    @staticmethod
    def _rewrite_keyframe_paths(
        keyframe_text: str,
        keyframes_dir: str,
        rename_map: dict[str, str] | None = None,
    ) -> str:
        """将 keyframe JSON 中的 image_path 更新为新目录下的路径。"""
        try:
            data = json.loads(keyframe_text)
            for kf in data.get("keyframes", []):
                old_path = kf.get("image_path", "")
                fname = os.path.basename(old_path)
                if rename_map and fname in rename_map:
                    fname = rename_map[fname]
                kf["image_path"] = os.path.join(keyframes_dir, fname)
            return json.dumps(data, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, TypeError):
            return keyframe_text

    @staticmethod
    def _fix_image_paths(article: str, keyframes_dir: str) -> str:
        """修正 Markdown 中图片路径为相对路径 keyframes/xxx.jpg。

        匹配所有 ![...](路径) 格式，将路径替换为 keyframes/文件名。
        """
        def _replace(match: re.Match) -> str:
            alt = match.group(1)
            old_path = match.group(2)
            fname = os.path.basename(old_path)
            return f"![{alt}](keyframes/{fname})"

        return re.sub(
            r"!\[([^\]]*)\]\(([^)]*frame_[^)]+\.jpg)\)",
            _replace,
            article,
        )

    @staticmethod
    def _check_error(text: str) -> bool:
        """检查返回内容是否包含错误信息。"""
        try:
            data = json.loads(text)
            return "error" in data
        except (json.JSONDecodeError, TypeError):
            return False

    def _flush_to_file(
        self,
        all_segment_articles: list[str | None],
        output_path: str,
        keyframes_dir: str,
    ) -> None:
        """将当前已完成的所有片段合并写入目标文件（实时刷新）。

        每次调用都会重写整个文件，保证片段顺序正确且内容完整。
        """
        completed = [a for a in all_segment_articles if a]
        if not completed:
            return

        if len(completed) == 1:
            merged = completed[0]
        else:
            merged = "\n\n---\n\n".join(completed)

        merged = self._fix_image_paths(merged, keyframes_dir)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(merged)

    # ------------------------------------------------------------------
    # 批量并发处理多个视频
    # ------------------------------------------------------------------
    @staticmethod
    async def run_batch(
        video_paths: list[str],
        target_language: str = "中文",
        max_concurrency: int = 3,
        *,
        model_name: str = "qwen3.6-plus",
        api_key: str | None = None,
        whisper_model_size: str = "medium",
        scene_threshold: float = 0.08,
        min_interval_sec: float = 5.0,
        output_dir: str = "output",
        max_duration_sec: float = 30 * 60,
        segment_duration_sec: float = 10 * 60,
        video_dir: str = "",
        state_db: StateDB | None = None,
        batch_id: str | None = None,
    ) -> list[dict]:
        """并发处理多个视频文件，通过 SQLite 追踪每个视频的处理状态。

        为每个视频创建独立的 VTTPipeline 实例以隔离 Agent 状态，
        通过信号量控制最大并发数。

        Args:
            video_paths: 视频文件路径列表。
            target_language: 目标翻译语言。
            max_concurrency: 最大并发处理数，默认 3。
            model_name: DashScope 模型名称。
            api_key: DashScope API Key。
            whisper_model_size: Whisper 模型大小。
            scene_threshold: 关键帧场景切换阈值。
            min_interval_sec: 关键帧最小间隔。
            output_dir: 输出目录。
            max_duration_sec: 视频超过此时长则分段。
            segment_duration_sec: 每段时长。
            video_dir: 视频目录（用于断点续传查找）。
            state_db: StateDB 实例（可选，不传则自动创建）。
            batch_id: 已有批次ID（恢复模式）。

        Returns:
            处理结果列表，每项包含:
            {"video_path": str, "status": "success"|"failed",
             "article": str|None, "output_path": str|None, "error": str|None}
        """
        if state_db is None:
            state_db = StateDB()

        # 创建或复用批次
        if batch_id is None:
            batch_id = state_db.create_batch(
                video_paths=video_paths,
                target_language=target_language,
                concurrency=max_concurrency,
                video_dir=video_dir,
            )
            logger.info("创建新批次: %s", batch_id)
        else:
            # 恢复模式：只处理未完成的视频
            pending = state_db.get_pending_paths(batch_id)
            if pending:
                video_paths = pending
                state_db.update_batch_status(batch_id, "running")
                logger.info(
                    "恢复批次 %s: 跳过已完成, 剩余 %d 个视频",
                    batch_id, len(pending),
                )
            else:
                logger.info("批次 %s 所有视频已完成", batch_id)
                return []

        sem = asyncio.Semaphore(max_concurrency)
        results: list[dict] = [None] * len(video_paths)  # type: ignore[list-item]

        logger.info("=" * 60)
        logger.info(
            "批量处理: batch=%s, 共 %d 个视频, 最大并发 %d",
            batch_id, len(video_paths), max_concurrency,
        )
        logger.info("=" * 60)

        async def _process_one(index: int, vpath: str) -> dict:
            async with sem:
                video_name = Path(vpath).stem

                # 查找该视频对应的 task 记录
                task_row = state_db.get_task_by_path(batch_id, vpath)
                task_id = task_row["task_id"] if task_row else None

                # 跳过已完成
                if task_row and task_row["status"] == "completed":
                    logger.info(
                        "[批量 %d/%d] 已完成，跳过: %s",
                        index + 1, len(video_paths), video_name,
                    )
                    return {
                        "video_path": vpath,
                        "status": "success",
                        "article": None,
                        "output_path": task_row.get("output_path"),
                        "error": None,
                    }

                # 标记为处理中
                if task_id:
                    state_db.update_task_status(task_id, "processing")

                logger.info(
                    "[批量 %d/%d] 开始处理: %s",
                    index + 1, len(video_paths), video_name,
                )
                try:
                    pipeline = VTTPipeline(
                        model_name=model_name,
                        api_key=api_key,
                        whisper_model_size=whisper_model_size,
                        scene_threshold=scene_threshold,
                        min_interval_sec=min_interval_sec,
                        output_dir=output_dir,
                        max_duration_sec=max_duration_sec,
                        segment_duration_sec=segment_duration_sec,
                    )
                    article, output_path = await pipeline.run(
                        video_path=vpath,
                        target_language=target_language,
                    )

                    # 标记完成
                    if task_id:
                        state_db.update_task_status(
                            task_id, "completed", output_path=output_path,
                        )

                    logger.info(
                        "[批量 %d/%d] 完成: %s -> %s",
                        index + 1, len(video_paths), video_name, output_path,
                    )
                    return {
                        "video_path": vpath,
                        "status": "success",
                        "article": article,
                        "output_path": output_path,
                        "error": None,
                    }
                except Exception as e:
                    # 标记失败
                    if task_id:
                        state_db.update_task_status(
                            task_id, "failed", error=str(e),
                        )
                    logger.exception(
                        "[批量 %d/%d] 失败: %s",
                        index + 1, len(video_paths), video_name,
                    )
                    return {
                        "video_path": vpath,
                        "status": "failed",
                        "article": None,
                        "output_path": None,
                        "error": str(e),
                    }

        tasks = [
            asyncio.create_task(_process_one(i, vp))
            for i, vp in enumerate(video_paths)
        ]

        done_results = await asyncio.gather(*tasks)
        for i, r in enumerate(done_results):
            results[i] = r

        # 更新批次最终状态
        state_db.finish_batch(batch_id)

        success = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        logger.info("=" * 60)
        logger.info(
            "批量处理完成 (batch=%s): 成功 %d, 失败 %d, 共 %d",
            batch_id, success, failed, len(results),
        )
        logger.info("=" * 60)

        return results

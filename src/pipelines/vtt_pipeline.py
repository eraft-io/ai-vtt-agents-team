"""VTT Pipeline - 视频转录翻译工作流编排。

编排多个 Agent 的执行顺序：
  预处理:  检测视频时长，超过 30 分钟则切割为 20 分钟片段
  每个片段:
    阶段一（并行）: 转录 + 关键帧提取
    阶段二（串行）: 总结 -> 翻译 -> 校对
  最终: 合并所有片段的文章
"""

import asyncio
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

logger = logging.getLogger(__name__)


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
        # 第一个片段的转录结果用于提取主题名
        # ============================================================
        first_segment_path = segments[0]["path"]

        logger.info("[主题提取] 转录第一个片段以提取主题名")
        first_resp = await transcribe_video(
            video_path=first_segment_path,
            model_size=self.whisper_model_size,
        )
        first_transcribe = "".join(
            block["text"] for block in first_resp.content
            if block.get("type") == "text"
        )

        topic_slug = await self._extract_topic_slug(first_transcribe)
        logger.info("视频主题: %s", topic_slug)

        project_dir = os.path.join(self.output_dir, "articles", topic_slug)
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
                logger.info(
                    "[状态保存] 片段 %d/%d 已标记为 completed",
                    seg_idx + 1, total,
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
        # 合并所有片段的文章
        # ============================================================
        completed_articles = [a for a in all_segment_articles if a]

        if len(completed_articles) == 1:
            final_article = completed_articles[0]
        else:
            final_article = "\n\n---\n\n".join(completed_articles)

        # 修正图片路径
        final_article = self._fix_image_paths(final_article, keyframes_dir)

        # 保存最终文章
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{video_name}_{target_language}_{timestamp}.md"
        output_path = os.path.join(project_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_article)

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
            self.output_dir, f"_tmp_keyframes_seg{seg_index}",
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

"""VTT Pipeline - 视频转录翻译工作流编排。

编排 5 个 Agent 的执行顺序：
  阶段一（并行）: TranscriberAgent + KeyframeExtractorAgent
  阶段二（串行）: SummarizerAgent -> TranslatorAgent -> ProofreaderAgent
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from agentscope.message import Msg

from src.agents import (
    create_summarizer_agent,
    create_translator_agent,
    create_proofreader_agent,
)
from src.tools.whisper_tool import transcribe_video
from src.tools.video_tool import extract_keyframes

logger = logging.getLogger(__name__)


class VTTPipeline:
    """视频转录翻译 Pipeline。

    编排多个 Agent 完成视频 -> 字幕 -> 总结 -> 翻译 -> 校对的完整流程。
    """

    def __init__(
        self,
        model_name: str = "qwen-max",
        api_key: str | None = None,
        whisper_model_size: str = "medium",
        scene_threshold: float = 0.08,
        min_interval_sec: float = 5.0,
        output_dir: str = "output",
    ) -> None:
        """初始化 Pipeline。

        Args:
            model_name: DashScope 模型名称。
            api_key: DashScope API Key。
            whisper_model_size: Whisper 模型大小。
            scene_threshold: 关键帧场景切换阈值。
            min_interval_sec: 关键帧最小间隔。
            output_dir: 输出目录。
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model_name = model_name
        self.output_dir = output_dir
        self.whisper_model_size = whisper_model_size
        self.scene_threshold = scene_threshold
        self.min_interval_sec = min_interval_sec

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
    ) -> str:
        """执行完整的视频转录翻译流程。

        Args:
            video_path: 视频文件路径。
            target_language: 目标翻译语言。

        Returns:
            最终校对后的 Markdown 文章内容。
        """
        video_path = str(Path(video_path).expanduser().resolve())
        video_name = Path(video_path).stem

        logger.info("=" * 60)
        logger.info("开始处理视频: %s", video_path)
        logger.info("目标语言: %s", target_language)
        logger.info("=" * 60)

        # ============================================================
        # 阶段一（并行）: 转录 + 关键帧提取（直接调用工具，不经 LLM）
        # ============================================================
        logger.info("[阶段一] 并行执行: 视频转录 + 关键帧提取")

        keyframes_dir = os.path.join(self.output_dir, "keyframes")

        transcribe_resp, keyframe_resp = await asyncio.gather(
            transcribe_video(
                video_path=video_path,
                model_size=self.whisper_model_size,
            ),
            extract_keyframes(
                video_path=video_path,
                output_dir=keyframes_dir,
                scene_threshold=self.scene_threshold,
                min_interval_sec=self.min_interval_sec,
            ),
        )

        # 从 ToolResponse 中提取文本
        transcribe_text = "".join(
            block["text"] for block in transcribe_resp.content
            if block.get("type") == "text"
        )
        keyframe_text = "".join(
            block["text"] for block in keyframe_resp.content
            if block.get("type") == "text"
        )

        logger.info("[阶段一] 完成")

        # 保存转录原文到文件
        transcribe_output_path = os.path.join(
            self.output_dir, "articles", f"{video_name}_transcription.json",
        )
        os.makedirs(os.path.dirname(transcribe_output_path), exist_ok=True)
        with open(transcribe_output_path, "w", encoding="utf-8") as f:
            f.write(transcribe_text)
        logger.info("转录原文已保存至: %s", transcribe_output_path)

        # 检查是否有错误
        if self._check_error(transcribe_text):
            logger.error("转录失败: %s", transcribe_text)
            return f"错误: 视频转录失败\n{transcribe_text}"

        # ============================================================
        # 阶段二（串行）: 总结
        # ============================================================
        logger.info("[阶段二] 执行: 内容总结")

        summarize_msg = Msg(
            name="user",
            role="user",
            content=(
                "请将以下字幕内容总结成分段文章，并在合适位置插入关键帧图片。\n\n"
                f"## 字幕数据\n```json\n{transcribe_text}\n```\n\n"
                f"## 关键帧数据\n```json\n{keyframe_text}\n```"
            ),
        )

        summarize_result = await self.summarizer(summarize_msg)
        article = summarize_result.get_text_content()

        logger.info("[阶段二] 完成")

        # 检查字幕是否为空
        if not article or article.strip() == "":
            logger.warning("总结结果为空，跳过翻译和校对")
            return "提示: 未能从视频中提取到有效字幕内容。"

        # ============================================================
        # 阶段三（串行）: 翻译
        # ============================================================
        logger.info("[阶段三] 执行: 翻译为%s", target_language)

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

        logger.info("[阶段三] 完成")

        # ============================================================
        # 阶段四（串行）: 校对
        # ============================================================
        logger.info("[阶段四] 执行: 校对润色")

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

        logger.info("[阶段四] 完成")

        # ============================================================
        # 保存最终文章
        # ============================================================
        articles_dir = os.path.join(self.output_dir, "articles")
        os.makedirs(articles_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{video_name}_{target_language}_{timestamp}.md"
        output_path = os.path.join(articles_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_article)

        logger.info("=" * 60)
        logger.info("处理完成! 文章已保存至: %s", output_path)
        logger.info("=" * 60)

        return final_article

    @staticmethod
    def _check_error(text: str) -> bool:
        """检查返回内容是否包含错误信息。"""
        try:
            data = json.loads(text)
            return "error" in data
        except (json.JSONDecodeError, TypeError):
            return False

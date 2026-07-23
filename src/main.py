"""AI VTT Agents Team - 多 Agent 视频转录翻译系统入口。

解析用户提示词，提取视频路径和目标语言，启动 Pipeline 执行完整流程。

使用方式:
    python -m src.main "帮我把桌面上的 demo.mp4 转录成中文文章"
    python -m src.main --video /path/to/video.mp4 --language 中文
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path

from agentscope.model import OpenAIChatModel

from src.pipelines.vtt_pipeline import VTTPipeline, scan_video_dir

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def parse_user_prompt(
    prompt: str,
    api_key: str,
    model_name: str = "qwen3.6-plus",
) -> dict:
    """使用 LLM 解析用户提示词，提取视频路径和目标语言。

    Args:
        prompt: 用户输入的提示词。
        api_key: DashScope API Key。
        model_name: 模型名称。

    Returns:
        包含 video_path 和 target_language 的字典。
    """
    model = OpenAIChatModel(
        model_name=model_name,
        api_key=api_key,
        stream=False,
        client_kwargs={"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    )

    parse_prompt = (
        "你是一个提示词解析助手。请从用户输入中提取以下信息，以 JSON 格式返回：\n"
        '- video_path: 视频文件路径（字符串）\n'
        '- target_language: 目标翻译语言（字符串，如 "中文"、"英文"、"日文"）\n\n'
        "如果用户未指定目标语言，默认为 \"中文\"。\n"
        "如果路径包含 ~ 或 桌面 等相对路径，请尝试展开。\n"
        '对于 "桌面"，展开为 "~/Desktop"。\n\n'
        "只返回 JSON，不要有其他内容。示例：\n"
        '{"video_path": "~/Desktop/demo.mp4", "target_language": "中文"}'
    )

    messages = [
        {"role": "system", "content": parse_prompt},
        {"role": "user", "content": prompt},
    ]

    response = await model(messages)

    # 从 LLM 响应中提取 JSON
    text = "".join(
        block["text"] for block in response.content
        if block.get("type") == "text"
    )
    # 尝试提取 JSON 块
    json_match = re.search(r"\{[^}]+\}", text)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return {
                "video_path": result.get("video_path", ""),
                "target_language": result.get("target_language", "中文"),
            }
        except json.JSONDecodeError:
            pass

    # 如果 LLM 解析失败，尝试简单的正则提取
    logger.warning("LLM 解析失败，使用正则提取")
    video_match = re.search(r"[\w~/\\.:]+\.(?:mp4|avi|mkv|mov|wmv|flv|webm)", prompt)
    video_path = video_match.group() if video_match else ""

    # 简单的语言匹配
    lang_map = {
        "中文": "中文", "chinese": "中文",
        "英文": "英文", "english": "英文",
        "日文": "日文", "japanese": "日文",
        "韩文": "韩文", "korean": "韩文",
        "法文": "法文", "french": "法文",
        "德文": "德文", "german": "德文",
    }
    target_language = "中文"
    for key, value in lang_map.items():
        if key in prompt.lower():
            target_language = value
            break

    return {
        "video_path": video_path,
        "target_language": target_language,
    }


def _find_config_path() -> str:
    """按优先级查找配置文件路径。

    优先级:
      1. 当前工作目录下的 config/agent_config.json
      2. 包内默认的 src/config/agent_config.json
    """
    # 1. CWD (开发模式 / 用户在项目目录下运行)
    cwd_config = os.path.join(os.getcwd(), "config", "agent_config.json")
    if os.path.isfile(cwd_config):
        return cwd_config

    # 2. 包内默认配置 (pip install 后)
    pkg_config = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config", "agent_config.json",
    )
    if os.path.isfile(pkg_config):
        return pkg_config

    return ""


def load_config(config_path: str | None = None) -> dict:
    """加载配置文件。

    如果未指定 config_path，按优先级自动查找。
    """
    if config_path is None:
        config_path = _find_config_path()
    if not config_path or not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def _load_pipeline_config() -> tuple[str, str, str, float, float]:
    """加载配置并返回 (api_key, model_name, whisper_model_size, scene_threshold, min_interval_sec)。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        logger.error(
            "请先设置 DASHSCOPE_API_KEY 环境变量:\n"
            '  export DASHSCOPE_API_KEY="sk-xxxxxxxx"'
        )
        sys.exit(1)

    config = load_config()
    model_configs = config.get("model_configs", [])
    agent_configs = config.get("agent_configs", {})

    model_name = "qwen3.6-plus"
    if model_configs:
        model_name = model_configs[0].get("model_name", "qwen3.6-plus")

    whisper_model_size = agent_configs.get(
        "transcriber", {},
    ).get("whisper_model_size", "medium")

    scene_threshold = agent_configs.get(
        "keyframe_extractor", {},
    ).get("scene_threshold", 0.3)

    min_interval_sec = agent_configs.get(
        "keyframe_extractor", {},
    ).get("min_interval_sec", 5)

    return api_key, model_name, whisper_model_size, scene_threshold, min_interval_sec


async def main_async(
    video_path: str | None = None,
    target_language: str | None = None,
    prompt: str | None = None,
    video_paths: list[str] | None = None,
    max_concurrency: int = 3,
) -> None:
    """异步主函数，支持单视频和多视频并发处理。"""
    api_key, model_name, whisper_model_size, scene_threshold, min_interval_sec = (
        _load_pipeline_config()
    )

    # ------------------------------------------------------------------
    # 多视频并发模式
    # ------------------------------------------------------------------
    if video_paths and len(video_paths) > 0:
        if not target_language:
            target_language = "中文"

        logger.info("进入批量并发模式: %d 个视频, 并发度 %d", len(video_paths), max_concurrency)

        results = await VTTPipeline.run_batch(
            video_paths=video_paths,
            target_language=target_language,
            max_concurrency=max_concurrency,
            model_name=model_name,
            api_key=api_key,
            whisper_model_size=whisper_model_size,
            scene_threshold=scene_threshold,
            min_interval_sec=min_interval_sec,
        )

        print("\n" + "=" * 60)
        print("批量处理结果汇总:")
        print("=" * 60)
        for r in results:
            status_icon = "OK" if r["status"] == "success" else "FAIL"
            video_name = Path(r["video_path"]).name
            if r["status"] == "success":
                print(f"  [{status_icon}] {video_name} -> {r['output_path']}")
            else:
                print(f"  [{status_icon}] {video_name} -> {r['error']}")
        return

    # ------------------------------------------------------------------
    # 单视频模式（兼容原有逻辑）
    # ------------------------------------------------------------------
    # 如果用户给了提示词，先解析
    if prompt and (not video_path):
        logger.info("正在解析用户提示词: %s", prompt)
        parsed = await parse_user_prompt(prompt, api_key, model_name)
        video_path = parsed["video_path"]
        if not target_language:
            target_language = parsed["target_language"]
        logger.info("解析结果: 视频=%s, 语言=%s", video_path, target_language)

    if not video_path:
        logger.error("请提供视频文件路径")
        sys.exit(1)

    if not target_language:
        target_language = "中文"

    # 创建并运行 Pipeline
    pipeline = VTTPipeline(
        model_name=model_name,
        api_key=api_key,
        whisper_model_size=whisper_model_size,
        scene_threshold=scene_threshold,
        min_interval_sec=min_interval_sec,
    )

    try:
        result, output_path = await pipeline.run(
            video_path=video_path,
            target_language=target_language,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"最终文章 (已保存至 {output_path}):")
    print("=" * 60)
    print(result)


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="AI VTT Agents Team - 多 Agent 视频转录翻译系统",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help='自然语言提示词，如: "帮我把桌面上的 demo.mp4 转录成中文文章"',
    )
    parser.add_argument(
        "--video",
        type=str,
        nargs="+",
        default=None,
        help="视频文件路径（支持多个，空格分隔；支持 glob 通配符如 *.mp4）",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="目标翻译语言，默认为中文",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="多视频并发处理数，默认为 3",
    )

    args = parser.parse_args()

    # 展开 glob 通配符并收集所有视频路径
    video_paths: list[str] = []
    if args.video:
        for pattern in args.video:
            p = Path(pattern).expanduser()
            if p.is_dir():
                # 目录：自动扫描视频文件
                scanned = scan_video_dir(str(p))
                video_paths.extend(scanned)
            else:
                expanded = glob.glob(str(p))
                if expanded:
                    video_paths.extend(expanded)
                else:
                    # 非通配符路径直接添加
                    video_paths.append(pattern)

    if len(video_paths) > 1:
        # 多视频并发模式
        asyncio.run(
            main_async(
                video_paths=video_paths,
                target_language=args.language,
                max_concurrency=args.concurrency,
            ),
        )
    else:
        # 单视频或提示词模式
        single_video = video_paths[0] if video_paths else None
        asyncio.run(
            main_async(
                video_path=single_video,
                target_language=args.language,
                prompt=args.prompt,
            ),
        )


if __name__ == "__main__":
    main()

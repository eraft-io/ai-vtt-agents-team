"""AI VTT Agents Team - 多 Agent 视频转录翻译系统入口。

解析用户提示词，提取视频路径和目标语言，启动 Pipeline 执行完整流程。

使用方式:
    python -m src.main "帮我把桌面上的 demo.mp4 转录成中文文章"
    python -m src.main --video /path/to/video.mp4 --language 中文
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys

from agentscope.model import DashScopeChatModel

from src.pipelines.vtt_pipeline import VTTPipeline

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
    model_name: str = "qwen-max",
) -> dict:
    """使用 LLM 解析用户提示词，提取视频路径和目标语言。

    Args:
        prompt: 用户输入的提示词。
        api_key: DashScope API Key。
        model_name: 模型名称。

    Returns:
        包含 video_path 和 target_language 的字典。
    """
    model = DashScopeChatModel(
        model_name=model_name,
        api_key=api_key,
        stream=False,
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


def load_config(config_path: str = "config/agent_config.json") -> dict:
    """加载配置文件。"""
    if not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


async def main_async(
    video_path: str | None = None,
    target_language: str | None = None,
    prompt: str | None = None,
) -> None:
    """异步主函数。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        logger.error(
            "请先设置 DASHSCOPE_API_KEY 环境变量:\n"
            '  export DASHSCOPE_API_KEY="sk-xxxxxxxx"'
        )
        sys.exit(1)

    # 加载配置
    config = load_config()
    model_configs = config.get("model_configs", [])
    agent_configs = config.get("agent_configs", {})

    model_name = "qwen-max"
    if model_configs:
        model_name = model_configs[0].get("model_name", "qwen-max")

    whisper_model_size = agent_configs.get(
        "transcriber", {},
    ).get("whisper_model_size", "medium")

    scene_threshold = agent_configs.get(
        "keyframe_extractor", {},
    ).get("scene_threshold", 0.3)

    min_interval_sec = agent_configs.get(
        "keyframe_extractor", {},
    ).get("min_interval_sec", 5)

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

    result = await pipeline.run(
        video_path=video_path,
        target_language=target_language,
    )

    print("\n" + "=" * 60)
    print("最终文章:")
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
        default=None,
        help="视频文件路径",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="目标翻译语言，默认为中文",
    )

    args = parser.parse_args()

    asyncio.run(
        main_async(
            video_path=args.video,
            target_language=args.language,
            prompt=args.prompt,
        ),
    )


if __name__ == "__main__":
    main()

"""翻译字幕 Agent。

将 Markdown 文章翻译为目标语言。
"""

import os

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel

TRANSLATOR_SYS_PROMPT = (
    "你是一个专业翻译。你的任务是将给定的 Markdown 文章翻译为目标语言。\n\n"
    "要求：\n"
    "1. 保持 Markdown 格式不变\n"
    "2. 保留图片引用和链接\n"
    "3. 翻译要准确自然，符合目标语言的表达习惯\n"
    "4. 专业术语优先使用目标语言的通用译法\n"
    "5. 不要添加任何额外的解释或注释\n"
    "6. 直接输出翻译后的完整 Markdown 文章\n\n"
    "你会收到：\n"
    "- 需要翻译的 Markdown 文章\n"
    "- 目标语言\n\n"
    "请直接输出翻译后的 Markdown 文章，不要包含任何其他内容。"
)


def create_translator_agent(
    model_name: str = "qwen-max",
    api_key: str | None = None,
) -> ReActAgent:
    """创建翻译字幕 Agent。

    Args:
        model_name: DashScope 模型名称。
        api_key: DashScope API Key，默认从环境变量读取。

    Returns:
        配置好的 ReActAgent 实例。
    """
    if api_key is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    agent = ReActAgent(
        name="translator",
        sys_prompt=TRANSLATOR_SYS_PROMPT,
        model=DashScopeChatModel(
            model_name=model_name,
            api_key=api_key,
        ),
        memory=InMemoryMemory(),
        formatter=DashScopeChatFormatter(),
    )

    return agent

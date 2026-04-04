"""校对文章 Agent。

对翻译后的文章进行逐行校对，标注专业术语和文字润色。
"""

import os

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

PROOFREADER_SYS_PROMPT = (
    "你是一个严谨的文章校对专家。你的任务是对翻译后的文章进行逐行校对。\n\n"
    "要求：\n"
    "1. 检查翻译的准确性和流畅度\n"
    "2. 标注专业术语（使用括号附上原文，如：机器学习(Machine Learning)）\n"
    "3. 对不通顺的句子进行润色\n"
    "4. 保持 Markdown 格式不变\n"
    "5. 保留图片引用和链接\n"
    "6. 输出校对后的最终版本\n\n"
    "请直接输出校对润色后的完整 Markdown 文章，不要包含修改说明或批注。"
)


def create_proofreader_agent(
    model_name: str = "qwen3.6-plus",
    api_key: str | None = None,
) -> ReActAgent:
    """创建校对文章 Agent。

    Args:
        model_name: DashScope 模型名称。
        api_key: DashScope API Key，默认从环境变量读取。

    Returns:
        配置好的 ReActAgent 实例。
    """
    if api_key is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    agent = ReActAgent(
        name="proofreader",
        sys_prompt=PROOFREADER_SYS_PROMPT,
        model=OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            client_kwargs={"base_url": DASHSCOPE_BASE_URL},
        ),
        memory=InMemoryMemory(),
        formatter=OpenAIChatFormatter(),
    )

    return agent

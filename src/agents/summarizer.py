"""总结字幕 Agent。

将视频字幕内容整理成分段文章，并在合适位置插入视频关键帧。
"""

import os

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel

SUMMARIZER_SYS_PROMPT = (
    "你是一个专业的内容总结编辑。你的任务是将视频字幕内容整理成一篇结构清晰的文章。\n\n"
    "要求：\n"
    "1. 将内容按主题分成多个段落\n"
    "2. 每个段落前给出简明的总结标题（使用 ## 格式）\n"
    "3. 在合适的位置插入视频关键帧图片（使用 Markdown 图片语法：![关键帧](图片路径)）\n"
    "4. 关键帧应插入在与其时间戳最接近的段落中\n"
    "5. 保持原文语言，不进行翻译\n"
    "6. 输出必须是纯 Markdown 格式\n\n"
    "你会收到两部分数据：\n"
    "- 字幕数据（JSON 格式，包含 segments 数组，每个 segment 有 start、end、text 字段）\n"
    "- 关键帧数据（JSON 格式，包含 keyframes 数组，每个 keyframe 有 timestamp 和 image_path）\n\n"
    "请根据字幕内容的语义进行分段总结，并将关键帧图片插入到时间戳对应的段落位置。"
)


def create_summarizer_agent(
    model_name: str = "qwen-max",
    api_key: str | None = None,
) -> ReActAgent:
    """创建总结字幕 Agent。

    Args:
        model_name: DashScope 模型名称。
        api_key: DashScope API Key，默认从环境变量读取。

    Returns:
        配置好的 ReActAgent 实例。
    """
    if api_key is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    agent = ReActAgent(
        name="summarizer",
        sys_prompt=SUMMARIZER_SYS_PROMPT,
        model=DashScopeChatModel(
            model_name=model_name,
            api_key=api_key,
        ),
        memory=InMemoryMemory(),
        formatter=DashScopeChatFormatter(),
    )

    return agent

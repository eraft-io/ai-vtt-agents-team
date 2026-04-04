"""视频转文字 Agent。

使用 Whisper 将视频中的语音转录为带时间戳的字幕数据。
"""

import os

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit

from src.tools.whisper_tool import transcribe_video

TRANSCRIBER_SYS_PROMPT = (
    "你是一个视频转录专家。你的任务是接收视频文件路径，调用 transcribe_video 工具"
    "将视频中的语音转录为带时间戳的字幕文本。输出必须为结构化的 JSON 格式。\n\n"
    "工作流程：\n"
    "1. 接收用户提供的视频文件路径\n"
    "2. 调用 transcribe_video 工具进行转录\n"
    "3. 返回转录结果（JSON 格式，包含 segments 和 language 字段）\n\n"
    "注意：如果工具返回错误信息，请直接将错误信息返回给用户。"
)


def create_transcriber_agent(
    model_name: str = "qwen-max",
    api_key: str | None = None,
    whisper_model_size: str = "medium",
) -> ReActAgent:
    """创建视频转文字 Agent。

    Args:
        model_name: DashScope 模型名称。
        api_key: DashScope API Key，默认从环境变量读取。
        whisper_model_size: Whisper 模型大小。

    Returns:
        配置好的 ReActAgent 实例。
    """
    if api_key is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    toolkit = Toolkit()
    toolkit.register_tool_function(transcribe_video)

    agent = ReActAgent(
        name="transcriber",
        sys_prompt=TRANSCRIBER_SYS_PROMPT,
        model=DashScopeChatModel(
            model_name=model_name,
            api_key=api_key,
        ),
        memory=InMemoryMemory(),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
    )

    return agent

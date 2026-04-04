"""关键帧提取 Agent。

使用 OpenCV 从视频中提取关键帧图片并记录时间戳。
"""

import os

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit

from src.tools.video_tool import extract_keyframes

KEYFRAME_SYS_PROMPT = (
    "你是一个视频关键帧提取专家。你的任务是接收视频文件路径，调用 extract_keyframes 工具"
    "提取关键帧图片，并记录每个关键帧对应的时间戳。\n\n"
    "工作流程：\n"
    "1. 接收用户提供的视频文件路径\n"
    "2. 调用 extract_keyframes 工具进行关键帧提取\n"
    "3. 返回提取结果（JSON 格式，包含 keyframes 数组，每个元素有 timestamp 和 image_path）\n\n"
    "注意：如果工具返回错误信息，请直接将错误信息返回给用户。"
)


def create_keyframe_agent(
    model_name: str = "qwen-max",
    api_key: str | None = None,
    scene_threshold: float = 0.3,
    min_interval_sec: float = 5.0,
) -> ReActAgent:
    """创建关键帧提取 Agent。

    Args:
        model_name: DashScope 模型名称。
        api_key: DashScope API Key，默认从环境变量读取。
        scene_threshold: 场景切换检测阈值。
        min_interval_sec: 相邻关键帧最小间隔。

    Returns:
        配置好的 ReActAgent 实例。
    """
    if api_key is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    toolkit = Toolkit()
    toolkit.register_tool_function(extract_keyframes)

    agent = ReActAgent(
        name="keyframe_extractor",
        sys_prompt=KEYFRAME_SYS_PROMPT,
        model=DashScopeChatModel(
            model_name=model_name,
            api_key=api_key,
        ),
        memory=InMemoryMemory(),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
    )

    return agent

"""Web UI for AI VTT Agents Team.

FastAPI application providing:
- Visual configuration editor for agent_config.json
- Interactive chat window with real-time Pipeline log streaming via WebSocket
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from src.pipelines.vtt_pipeline import VTTPipeline, scan_video_dir
from src.pipelines.state_db import StateDB
from src.main import parse_user_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "agent_config.json"
# 优先使用 CWD 下的配置（开发模式）
_cwd_config = Path.cwd() / "config" / "agent_config.json"
if _cwd_config.is_file():
    CONFIG_PATH = _cwd_config

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="AI VTT Agents Team")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve output/articles so that markdown image paths (e.g. keyframes/xxx.jpg) resolve
# 保留旧兼容，实际图片已通过 /api/file 服务
ARTICLES_DIR = Path.cwd() / "output" / "articles"
ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/articles", StaticFiles(directory=str(ARTICLES_DIR)), name="articles")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global StateDB instance
state_db = StateDB()


# ---------------------------------------------------------------------------
# WebSocket log handler – captures logger output and pushes to client
# ---------------------------------------------------------------------------
class WebSocketLogHandler(logging.Handler):
    """Logging handler that sends records through a WebSocket connection."""

    def __init__(self, ws: WebSocket, loop: asyncio.AbstractEventLoop):
        super().__init__(level=logging.INFO)
        self._ws = ws
        self._loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # 检测结构化进度信息
            progress = getattr(record, "progress", None)
            if progress:
                payload = json.dumps({
                    "type": "progress",
                    **progress,
                })
            else:
                msg = self.format(record)
                payload = json.dumps({
                    "type": "log",
                    "level": record.levelname,
                    "message": msg,
                })
            asyncio.run_coroutine_threadsafe(
                self._ws.send_text(payload),
                self._loop,
            )
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main page."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/config")
async def get_config():
    """Return current agent_config.json contents."""
    if not CONFIG_PATH.is_file():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.put("/api/config")
async def put_config(request: Request):
    """Save updated agent_config.json."""
    data = await request.json()
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return {"status": "ok"}

@app.get("/api/download")
async def download_file(path: str = Query(...)):
    """Download an output file (docx / markdown)."""
    file_path = Path(path).resolve()
    if not file_path.is_file():
        return {"error": "File not found"}
    # 根据扩展名自动选择 media_type
    suffix = file_path.suffix.lower()
    media_map = {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".md": "text/markdown",
    }
    media_type = media_map.get(suffix, "application/octet-stream")
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type=media_type,
    )


@app.get("/api/file")
async def serve_file(path: str = Query(...)):
    """Serve an arbitrary file (images, etc.) by absolute path.

    Used by the frontend to load keyframe images that are now stored
    alongside videos instead of under output/articles/.
    """
    file_path = Path(path).resolve()
    if not file_path.is_file():
        return {"error": "File not found"}
    # Guess media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".svg": "image/svg+xml",
        ".md": "text/markdown", ".json": "application/json",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    return FileResponse(path=str(file_path), media_type=media_type)


# ---------------------------------------------------------------------------
# Batch status API (SQLite-backed)
# ---------------------------------------------------------------------------

@app.get("/api/batches")
async def list_batches(limit: int = 20):
    """列出最近的批次任务。"""
    return state_db.list_batches(limit=limit)


@app.get("/api/batches/{batch_id}")
async def get_batch_detail(batch_id: str):
    """获取批次详情及其所有视频任务状态。"""
    batch = state_db.get_batch(batch_id)
    if not batch:
        return {"error": "Batch not found"}
    tasks = state_db.get_tasks(batch_id)
    return {"batch": batch, "tasks": tasks}


@app.post("/api/batches/{batch_id}/resume")
async def resume_batch(batch_id: str):
    """检查批次是否可恢复，返回待处理视频数。"""
    batch = state_db.get_batch(batch_id)
    if not batch:
        return {"error": "Batch not found"}
    pending = state_db.get_pending_paths(batch_id)
    return {"batch_id": batch_id, "pending_count": len(pending), "pending": pending}


@app.websocket("/ws/pipeline")
async def ws_pipeline(ws: WebSocket):
    """WebSocket endpoint that runs the VTT pipeline with real-time log streaming.

    Supports both single and batch video processing:
    - Single: {"video_path": "...", "target_language": "..."}
    - Batch:  {"video_paths": ["...", "..."], "target_language": "...", "max_concurrency": 3}
    """
    await ws.accept()

    try:
        # Wait for the task payload
        raw = await ws.receive_text()
        payload = json.loads(raw)

        video_path = payload.get("video_path", "")
        video_paths = payload.get("video_paths", [])
        video_dir = payload.get("video_dir", "")
        target_language = payload.get("target_language", "")
        prompt = payload.get("prompt", "")
        max_concurrency = int(payload.get("max_concurrency", 3))
        resume_batch_id = payload.get("resume_batch_id", "")

        # 如果传入了目录，自动扫描视频文件
        if video_dir and not video_paths:
            try:
                video_paths = scan_video_dir(video_dir)
            except ValueError as e:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": str(e),
                }))
                await ws.close()
                return
            if not video_paths:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": f"目录中未找到视频文件: {video_dir}",
                }))
                await ws.close()
                return

        if not video_path and not video_paths and not prompt:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": "请输入指令或视频路径",
            }))
            await ws.close()
            return

        # Load config
        config = {}
        if CONFIG_PATH.is_file():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)

        model_configs = config.get("model_configs", [])
        agent_configs = config.get("agent_configs", {})

        model_name = "qwen3.6-plus"
        if model_configs:
            model_name = model_configs[0].get("model_name", "qwen3.6-plus")

        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        # Config file api_key takes priority over env var (unless it's a placeholder)
        cfg_api_key = ""
        if model_configs:
            cfg_api_key = model_configs[0].get("api_key", "")
        if cfg_api_key and not cfg_api_key.startswith("${"):
            api_key = cfg_api_key
        whisper_model_size = agent_configs.get(
            "transcriber", {},
        ).get("whisper_model_size", "small")
        scene_threshold = agent_configs.get(
            "keyframe_extractor", {},
        ).get("scene_threshold", 0.08)
        min_interval_sec = agent_configs.get(
            "keyframe_extractor", {},
        ).get("min_interval_sec", 5)

        # Attach WebSocket log handler to root logger
        loop = asyncio.get_event_loop()
        ws_handler = WebSocketLogHandler(ws, loop)
        ws_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root_logger = logging.getLogger()
        root_logger.addHandler(ws_handler)

        try:
            # Notify start
            await ws.send_text(json.dumps({
                "type": "stage",
                "stage": "pipeline",
                "status": "started",
                "message": "正在解析指令...",
            }))

            # ----------------------------------------------------------
            # Batch mode: multiple video files processed concurrently
            # ----------------------------------------------------------
            if video_paths and len(video_paths) > 1:
                mode_label = "恢复" if resume_batch_id else "新建"
                await ws.send_text(json.dumps({
                    "type": "stage",
                    "stage": "batch",
                    "status": "started",
                    "message": f"批量模式({mode_label}): {len(video_paths)} 个视频, 并发度 {max_concurrency}",
                }))

                results = await VTTPipeline.run_batch(
                    video_paths=video_paths,
                    target_language=target_language or "中文",
                    max_concurrency=max_concurrency,
                    model_name=model_name,
                    api_key=api_key,
                    whisper_model_size=whisper_model_size,
                    scene_threshold=scene_threshold,
                    min_interval_sec=min_interval_sec,
                    video_dir=video_dir,
                    state_db=state_db,
                    batch_id=resume_batch_id or None,
                )

                await ws.send_text(json.dumps({
                    "type": "batch_result",
                    "results": [
                        {
                            "video_path": r["video_path"],
                            "status": r["status"],
                            "output_path": r.get("output_path"),
                            "error": r.get("error"),
                        }
                        for r in results
                    ],
                }))
                return

            # ----------------------------------------------------------
            # Single video mode (original logic)
            # ----------------------------------------------------------
            # Parse prompt if no explicit video_path
            if not video_path and prompt:
                parsed = await parse_user_prompt(prompt, api_key, model_name)
                video_path = parsed.get("video_path", "")
                if not target_language:
                    target_language = parsed.get("target_language", "中文")
                await ws.send_text(json.dumps({
                    "type": "stage",
                    "stage": "parse",
                    "status": "done",
                    "message": f"解析结果: 视频={video_path}, 语言={target_language}",
                }))

            # If video_paths has exactly 1 entry, use it
            if not video_path and video_paths:
                video_path = video_paths[0]

            if not video_path:
                await ws.send_text(json.dumps({
                    "type": "error",
                    "message": "无法从指令中提取视频路径，请检查输入",
                }))
                return

            if not target_language:
                target_language = "中文"

            pipeline = VTTPipeline(
                model_name=model_name,
                api_key=api_key,
                whisper_model_size=whisper_model_size,
                scene_threshold=scene_threshold,
                min_interval_sec=min_interval_sec,
            )

            article, output_path = await pipeline.run(
                video_path=video_path,
                target_language=target_language,
            )

            # Send final result
            await ws.send_text(json.dumps({
                "type": "result",
                "article": article,
                "output_path": output_path,
            }))

        finally:
            root_logger.removeHandler(ws_handler)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("Pipeline error")
        try:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": str(e),
            }))
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Startup helper
# ---------------------------------------------------------------------------
def start():
    """Entry point for `uv run vtt-web`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    uvicorn.run(
        "src.web.app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
    )

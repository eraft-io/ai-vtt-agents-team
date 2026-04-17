"""SQLite 状态管理 - 记录批量视频处理进度。

提供批次任务（batch）和单个视频任务（task）两层状态跟踪，
支持断点续传：重启后自动跳过已完成的视频。
"""

import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

# 数据库文件默认位于 output/vtt_state.db
DEFAULT_DB_PATH = os.path.join("output", "vtt_state.db")

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS batch_jobs (
    batch_id    TEXT PRIMARY KEY,
    video_dir   TEXT,
    target_language TEXT NOT NULL DEFAULT '中文',
    concurrency INTEGER NOT NULL DEFAULT 3,
    total       INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS video_tasks (
    task_id     TEXT PRIMARY KEY,
    batch_id    TEXT NOT NULL,
    video_path  TEXT NOT NULL,
    video_name  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    output_path TEXT,
    error       TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    completed_at TEXT,
    FOREIGN KEY (batch_id) REFERENCES batch_jobs(batch_id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_batch ON video_tasks(batch_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON video_tasks(status);
"""


def _now() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StateDB:
    """线程安全的 SQLite 状态管理器。"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db_path = db_path
        self._local = threading.local()
        # 初始化表结构
        conn = self._get_conn()
        conn.executescript(_CREATE_TABLES_SQL)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """每个线程使用独立连接。"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self._db_path, check_same_thread=False,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    # ------------------------------------------------------------------
    # Batch 操作
    # ------------------------------------------------------------------
    def create_batch(
        self,
        video_paths: list[str],
        target_language: str = "中文",
        concurrency: int = 3,
        video_dir: str = "",
    ) -> str:
        """创建一个新批次并插入所有视频任务，返回 batch_id。"""
        batch_id = uuid.uuid4().hex[:12]
        now = _now()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO batch_jobs (batch_id, video_dir, target_language, "
            "concurrency, total, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'running', ?, ?)",
            (batch_id, video_dir, target_language, concurrency,
             len(video_paths), now, now),
        )
        for vp in video_paths:
            task_id = uuid.uuid4().hex[:12]
            video_name = Path(vp).stem
            conn.execute(
                "INSERT INTO video_tasks (task_id, batch_id, video_path, "
                "video_name, status, created_at) VALUES (?, ?, ?, ?, 'pending', ?)",
                (task_id, batch_id, vp, video_name, now),
            )
        conn.commit()
        return batch_id

    def get_batch(self, batch_id: str) -> dict | None:
        """获取批次信息。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM batch_jobs WHERE batch_id = ?", (batch_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_batches(self, limit: int = 20) -> list[dict]:
        """列出最近的批次。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM batch_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_batch_status(self, batch_id: str, status: str) -> None:
        """更新批次状态。"""
        conn = self._get_conn()
        conn.execute(
            "UPDATE batch_jobs SET status = ?, updated_at = ? WHERE batch_id = ?",
            (status, _now(), batch_id),
        )
        conn.commit()

    def finish_batch(self, batch_id: str) -> None:
        """根据子任务状态自动计算批次最终状态。"""
        conn = self._get_conn()
        tasks = self.get_tasks(batch_id)
        if not tasks:
            return
        all_done = all(t["status"] in ("completed", "failed") for t in tasks)
        if not all_done:
            return
        has_failed = any(t["status"] == "failed" for t in tasks)
        status = "partial" if has_failed else "completed"
        self.update_batch_status(batch_id, status)

    # ------------------------------------------------------------------
    # Task 操作
    # ------------------------------------------------------------------
    def get_tasks(self, batch_id: str) -> list[dict]:
        """获取某批次下所有视频任务。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM video_tasks WHERE batch_id = ? ORDER BY created_at",
            (batch_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_task_by_path(self, batch_id: str, video_path: str) -> dict | None:
        """通过视频路径查找任务。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM video_tasks WHERE batch_id = ? AND video_path = ?",
            (batch_id, video_path),
        ).fetchone()
        return dict(row) if row else None

    def update_task_status(
        self,
        task_id: str,
        status: str,
        output_path: str | None = None,
        error: str | None = None,
    ) -> None:
        """更新视频任务状态。"""
        conn = self._get_conn()
        now = _now()
        if status == "processing":
            conn.execute(
                "UPDATE video_tasks SET status = ?, started_at = ? WHERE task_id = ?",
                (status, now, task_id),
            )
        elif status in ("completed", "failed"):
            conn.execute(
                "UPDATE video_tasks SET status = ?, output_path = ?, error = ?, "
                "completed_at = ? WHERE task_id = ?",
                (status, output_path, error, now, task_id),
            )
        else:
            conn.execute(
                "UPDATE video_tasks SET status = ? WHERE task_id = ?",
                (status, task_id),
            )
        conn.commit()

    def get_pending_paths(self, batch_id: str) -> list[str]:
        """获取批次中未完成的视频路径列表（pending + failed 状态）。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT video_path FROM video_tasks "
            "WHERE batch_id = ? AND status IN ('pending', 'failed') "
            "ORDER BY created_at",
            (batch_id,),
        ).fetchall()
        return [r["video_path"] for r in rows]

    def find_resumable_batch(self, video_dir: str) -> dict | None:
        """查找指定目录下最近一个可恢复的批次（有未完成任务的 running/partial 批次）。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM batch_jobs "
            "WHERE video_dir = ? AND status IN ('running', 'partial') "
            "ORDER BY created_at DESC LIMIT 1",
            (video_dir,),
        ).fetchone()
        if not row:
            return None
        batch = dict(row)
        pending = self.get_pending_paths(batch["batch_id"])
        if not pending:
            return None
        return batch

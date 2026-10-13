"""Chat sessions persisted in SQLite."""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")
TITLE_MAX_CHARS = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    messages    TEXT NOT NULL,
    transcript  TEXT NOT NULL,
    pending     TEXT
)
"""


def is_session_id(value: str) -> bool:
    """Session ids double as folder names, so only the generated form is accepted."""
    return bool(_SESSION_ID.match(value))


@dataclass
class Session:
    """One chat.

    ``messages`` is the conversation as sent to the Claude API. ``transcript``
    is what the UI shows. ``pending`` holds a question Claude asked through the
    ``ask_user`` tool that the user has not answered yet.
    """

    id: str
    title: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    messages: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, Any] | None = None


class SessionStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create(self) -> Session:
        now = time.time()
        session = Session(id=uuid.uuid4().hex, created_at=now, updated_at=now)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session.id, "", now, now, "[]", "[]", None),
            )
        return session

    def get(self, session_id: str) -> Session | None:
        if not is_session_id(session_id):
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return Session(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            messages=json.loads(row["messages"]),
            transcript=json.loads(row["transcript"]),
            pending=json.loads(row["pending"]) if row["pending"] else None,
        )

    def list(self) -> list[dict[str, Any]]:
        """Session summaries, most recently used first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def save(self, session: Session) -> None:
        session.updated_at = time.time()
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ?, messages = ?, transcript = ?, "
                "pending = ? WHERE id = ?",
                (
                    session.title,
                    session.updated_at,
                    json.dumps(session.messages),
                    json.dumps(session.transcript),
                    json.dumps(session.pending) if session.pending else None,
                    session.id,
                ),
            )

    def delete(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def make_title(text: str, file_count: int) -> str:
    """A short sidebar title from the first user message."""
    first_line = " ".join(text.split())
    if not first_line:
        return f"{file_count} image{'s' if file_count != 1 else ''}" if file_count else "New chat"
    if len(first_line) <= TITLE_MAX_CHARS:
        return first_line
    return first_line[: TITLE_MAX_CHARS - 1].rstrip() + "…"

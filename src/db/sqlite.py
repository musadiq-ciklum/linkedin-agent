# src/db/sqlite.py
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.config import SQLITE_DB_PATH

MIGRATIONS_DIR = Path(__file__).parent.parent.parent / "data" / "db" / "migrations"

_engines: dict[str, object] = {}


def _get_engine(db_path: str | None = None):
    path = db_path or SQLITE_DB_PATH
    if path not in _engines:
        _engines[path] = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    return _engines[path]


@contextmanager
def get_session(db_path: str | None = None) -> Session:
    engine = _get_engine(db_path)
    factory = sessionmaker(bind=engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def run_migrations(db_path: str | None = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                filename   TEXT NOT NULL UNIQUE,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()

        applied = {row["filename"] for row in conn.execute("SELECT filename FROM migrations")}

        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        for path in migration_files:
            if path.name in applied:
                continue
            sql = path.read_text()
            conn.executescript(sql)
            conn.execute("INSERT INTO migrations (filename) VALUES (?)", (path.name,))
            conn.commit()
            print(f"  apply {path.name}")
    finally:
        conn.close()


def is_db_ready(db_path: str | None = None) -> bool:
    try:
        conn = get_connection(db_path)
        conn.execute("SELECT 1 FROM migrations LIMIT 1")
        conn.close()
        return True
    except sqlite3.OperationalError:
        return False

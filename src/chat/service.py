# src/chat/service.py
import json
import uuid

from src.chat.models import ChatSession, Message
from src.chat import repository
from src.db.sqlite import get_session, run_migrations

CHAT_HISTORY_LIMIT = 50


def create_new_session(user_id: int, db_path: str | None = None) -> ChatSession:
    run_migrations(db_path)
    session_id = str(uuid.uuid4())
    with get_session(db_path) as session:
        return repository.create_session(session, session_id, user_id)


def get_or_create_session(user_id: int, db_path: str | None = None) -> ChatSession:
    run_migrations(db_path)
    with get_session(db_path) as session:
        existing = repository.get_latest_session(session, user_id)
        if existing:
            return existing
    return create_new_session(user_id, db_path)


def save_message(
    session_id: str,
    role: str,
    content: str,
    contexts: list[dict] | None = None,
    db_path: str | None = None,
) -> Message:
    run_migrations(db_path)
    contexts_json = json.dumps(contexts) if contexts is not None else None
    with get_session(db_path) as session:
        return repository.add_message(session, session_id, role, content, contexts_json)


def load_session(session_id: str, db_path: str | None = None) -> list[dict]:
    run_migrations(db_path)
    with get_session(db_path) as session:
        messages = repository.get_messages_for_session(session, session_id)
    return _messages_to_dicts(messages)


def load_recent_messages(session_id: str, db_path: str | None = None) -> list[dict]:
    run_migrations(db_path)
    with get_session(db_path) as session:
        messages = repository.get_last_n_messages(session, session_id, CHAT_HISTORY_LIMIT)
    return _messages_to_dicts(messages)


def rename_session(session_id: str, title: str, db_path: str | None = None) -> None:
    run_migrations(db_path)
    with get_session(db_path) as session:
        repository.update_title(session, session_id, title)


def delete_session(session_id: str, db_path: str | None = None) -> None:
    run_migrations(db_path)
    with get_session(db_path) as session:
        repository.delete_session(session, session_id)


def list_sessions(user_id: int, db_path: str | None = None) -> list[ChatSession]:
    run_migrations(db_path)
    with get_session(db_path) as session:
        return repository.list_sessions_for_user(session, user_id)


def _messages_to_dicts(messages: list[Message]) -> list[dict]:
    return [
        {
            "role": m.role,
            "content": m.content,
            "contexts": json.loads(m.contexts) if m.contexts else [],
        }
        for m in messages
    ]

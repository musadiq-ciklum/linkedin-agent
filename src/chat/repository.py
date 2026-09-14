# src/chat/repository.py
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.chat.models import ChatSession, Message


def create_session(session: Session, session_id: str, user_id: int) -> ChatSession:
    chat_session = ChatSession(id=session_id, user_id=user_id)
    session.add(chat_session)
    session.flush()
    session.expunge(chat_session)
    return chat_session


def get_latest_session(session: Session, user_id: int) -> ChatSession | None:
    result = (
        session.query(ChatSession)
        .filter_by(user_id=user_id)
        .order_by(ChatSession.created_at.desc(), text("sessions.rowid DESC"))
        .first()
    )
    if result:
        session.expunge(result)
    return result


def list_sessions_for_user(session: Session, user_id: int) -> list[ChatSession]:
    results = (
        session.query(ChatSession)
        .filter_by(user_id=user_id)
        .order_by(ChatSession.created_at.desc(), text("sessions.rowid DESC"))
        .all()
    )
    for r in results:
        session.expunge(r)
    return results


def add_message(
    session: Session,
    session_id: str,
    role: str,
    content: str,
    contexts: str | None = None,
) -> Message:
    msg = Message(session_id=session_id, role=role, content=content, contexts=contexts)
    session.add(msg)
    session.flush()
    session.expunge(msg)
    return msg


def get_messages_for_session(session: Session, session_id: str) -> list[Message]:
    results = (
        session.query(Message)
        .filter_by(session_id=session_id)
        .order_by(Message.id.asc())
        .all()
    )
    for r in results:
        session.expunge(r)
    return results


def update_title(session: Session, session_id: str, title: str) -> None:
    session.query(ChatSession).filter_by(id=session_id).update({"title": title})


def delete_session(session: Session, session_id: str) -> None:
    session.query(Message).filter_by(session_id=session_id).delete()
    session.query(ChatSession).filter_by(id=session_id).delete()


def get_last_n_messages(session: Session, session_id: str, n: int) -> list[Message]:
    rows = (
        session.query(Message)
        .filter_by(session_id=session_id)
        .order_by(Message.id.desc())
        .limit(n)
        .all()
    )
    for r in rows:
        session.expunge(r)
    return list(reversed(rows))

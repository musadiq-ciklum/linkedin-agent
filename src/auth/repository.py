# src/auth/repository.py
from sqlalchemy.orm import Session

from src.auth.models import User


def get_by_username(session: Session, username: str) -> User | None:
    return session.query(User).filter_by(username=username).first()


def create(session: Session, username: str, hashed_password: str) -> User:
    user = User(username=username, password=hashed_password)
    session.add(user)
    session.flush()
    session.expunge(user)
    return user

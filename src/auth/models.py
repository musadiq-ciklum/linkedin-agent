# src/auth/models.py
from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    password: Mapped[str] = mapped_column(String, nullable=False)  # bcrypt hash, never plaintext
    created_at: Mapped[str] = mapped_column(String, nullable=False, server_default="(datetime('now'))")

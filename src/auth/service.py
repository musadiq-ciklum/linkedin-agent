# src/auth/service.py
import bcrypt
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

from src.config import AUTH_SECRET_KEY
from src.auth.models import User
from src.auth import repository
from src.db.sqlite import get_session, run_migrations


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def register_user(username: str, password: str, db_path: str | None = None) -> User:
    if not username:
        raise ValueError("Username cannot be empty.")
    if not password:
        raise ValueError("Password cannot be empty.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")

    run_migrations(db_path)
    with get_session(db_path) as session:
        if repository.get_by_username(session, username):
            raise ValueError("Username already taken.")
        return repository.create(session, username, hash_password(password))


def login_user(username: str, password: str, db_path: str | None = None) -> str | None:
    run_migrations(db_path)
    with get_session(db_path) as session:
        user = repository.get_by_username(session, username)
        if user is None or not verify_password(password, user.password):
            return None

    serializer = URLSafeTimedSerializer(AUTH_SECRET_KEY)
    return serializer.dumps({"sub": username})


def get_user_id_by_username(username: str, db_path: str | None = None) -> int:
    run_migrations(db_path)
    with get_session(db_path) as session:
        user = repository.get_by_username(session, username)
        if user is None:
            raise ValueError(f"User '{username}' not found.")
        return user.id


def verify_token(token: str, max_age_seconds: int = 86400) -> str | None:
    serializer = URLSafeTimedSerializer(AUTH_SECRET_KEY)
    try:
        data = serializer.loads(token, max_age=max_age_seconds)
        return data["sub"]
    except (SignatureExpired, BadSignature, Exception):
        return None

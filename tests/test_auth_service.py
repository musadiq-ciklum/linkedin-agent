# tests/test_auth_service.py
import pytest
from src.auth.service import (
    hash_password,
    verify_password,
    register_user,
    login_user,
    verify_token,
)


def db(tmp_path):
    return str(tmp_path / "test.db")


def test_hash_password_returns_bcrypt_string(tmp_path):
    result = hash_password("secret")
    assert result.startswith("$2b$")


def test_verify_password_returns_true_for_correct_password(tmp_path):
    hashed = hash_password("secret")
    assert verify_password("secret", hashed) is True


def test_verify_password_returns_false_for_wrong_password(tmp_path):
    hashed = hash_password("secret")
    assert verify_password("wrong", hashed) is False


def test_user_can_register_with_valid_credentials(tmp_path):
    user = register_user("alice", "s3cr3t!!", db_path=db(tmp_path))
    assert user.username == "alice"
    assert user.password != "s3cr3t!!"
    assert user.password.startswith("$2b$")


def test_register_raises_on_empty_username(tmp_path):
    with pytest.raises(ValueError):
        register_user("", "s3cr3t!!", db_path=db(tmp_path))


def test_register_raises_on_empty_password(tmp_path):
    with pytest.raises(ValueError):
        register_user("alice", "", db_path=db(tmp_path))


def test_register_raises_on_password_shorter_than_8_characters(tmp_path):
    with pytest.raises(ValueError, match="8 characters"):
        register_user("alice", "short", db_path=db(tmp_path))


def test_register_raises_on_duplicate_username(tmp_path):
    path = db(tmp_path)
    register_user("alice", "s3cr3t!!", db_path=path)
    with pytest.raises(ValueError, match="Username already taken"):
        register_user("alice", "other_pw!!", db_path=path)


def test_login_returns_token_for_valid_credentials(tmp_path):
    path = db(tmp_path)
    register_user("alice", "s3cr3t!!", db_path=path)
    token = login_user("alice", "s3cr3t!!", db_path=path)
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0


def test_login_returns_none_for_wrong_password(tmp_path):
    path = db(tmp_path)
    register_user("alice", "s3cr3t!!", db_path=path)
    assert login_user("alice", "wrong", db_path=path) is None


def test_login_returns_none_for_unknown_username(tmp_path):
    assert login_user("ghost", "pw", db_path=db(tmp_path)) is None


def test_verify_token_returns_username_for_valid_token(tmp_path):
    path = db(tmp_path)
    register_user("alice", "s3cr3t!!", db_path=path)
    token = login_user("alice", "s3cr3t!!", db_path=path)
    assert verify_token(token) == "alice"


def test_verify_token_returns_none_for_tampered_token(tmp_path):
    assert verify_token("not.a.real.token") is None


def test_verify_token_returns_none_for_expired_token(tmp_path):
    path = db(tmp_path)
    register_user("alice", "s3cr3t!!", db_path=path)
    token = login_user("alice", "s3cr3t!!", db_path=path)
    assert verify_token(token, max_age_seconds=-1) is None

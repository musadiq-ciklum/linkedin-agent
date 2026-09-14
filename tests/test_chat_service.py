# tests/test_chat_service.py
import pytest
from src.auth.service import register_user, get_user_id_by_username
from src.chat.service import (
    create_new_session,
    get_or_create_session,
    save_message,
    load_session,
    load_recent_messages,
    list_sessions,
    rename_session,
    delete_session,
    CHAT_HISTORY_LIMIT,
)


def _db(tmp_path):
    return str(tmp_path / "test.db")


def _make_user(tmp_path, username="alice"):
    path = _db(tmp_path)
    register_user(username, "s3cr3t!!", db_path=path)
    return get_user_id_by_username(username, db_path=path), path


# ── Session creation ──────────────────────────────────────────────────────────

def test_create_new_session_returns_session_with_correct_user_id(tmp_path):
    user_id, path = _make_user(tmp_path)
    s = create_new_session(user_id, db_path=path)
    assert s.user_id == user_id


def test_create_new_session_generates_unique_ids(tmp_path):
    user_id, path = _make_user(tmp_path)
    s1 = create_new_session(user_id, db_path=path)
    s2 = create_new_session(user_id, db_path=path)
    assert s1.id != s2.id


def test_get_or_create_session_creates_session_when_none_exist(tmp_path):
    user_id, path = _make_user(tmp_path)
    s = get_or_create_session(user_id, db_path=path)
    assert s.user_id == user_id
    assert s.id is not None


def test_get_or_create_session_returns_most_recent_existing_session(tmp_path):
    user_id, path = _make_user(tmp_path)
    create_new_session(user_id, db_path=path)
    s2 = create_new_session(user_id, db_path=path)
    result = get_or_create_session(user_id, db_path=path)
    assert result.id == s2.id


# ── Message saving ────────────────────────────────────────────────────────────

def test_save_message_persists_user_message(tmp_path):
    user_id, path = _make_user(tmp_path)
    s = create_new_session(user_id, db_path=path)
    save_message(s.id, "user", "Hello?", db_path=path)
    msgs = load_session(s.id, db_path=path)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello?"


def test_save_message_persists_assistant_message_with_contexts(tmp_path):
    user_id, path = _make_user(tmp_path)
    s = create_new_session(user_id, db_path=path)
    contexts = [{"doc_id": "doc_1", "score": 0.9, "content": "Some text"}]
    save_message(s.id, "assistant", "Here is the answer.", contexts=contexts, db_path=path)
    msgs = load_session(s.id, db_path=path)
    assert msgs[0]["contexts"] == contexts


def test_save_message_stores_empty_contexts_as_empty_list(tmp_path):
    user_id, path = _make_user(tmp_path)
    s = create_new_session(user_id, db_path=path)
    save_message(s.id, "user", "Hi", contexts=[], db_path=path)
    msgs = load_session(s.id, db_path=path)
    assert msgs[0]["contexts"] == []


# ── Session loading ───────────────────────────────────────────────────────────

def test_load_session_returns_messages_in_chronological_order(tmp_path):
    user_id, path = _make_user(tmp_path)
    s = create_new_session(user_id, db_path=path)
    save_message(s.id, "user", "First", db_path=path)
    save_message(s.id, "assistant", "Second", db_path=path)
    msgs = load_session(s.id, db_path=path)
    assert msgs[0]["content"] == "First"
    assert msgs[1]["content"] == "Second"


def test_load_session_returns_empty_list_for_new_session(tmp_path):
    user_id, path = _make_user(tmp_path)
    s = create_new_session(user_id, db_path=path)
    assert load_session(s.id, db_path=path) == []


def test_load_recent_messages_returns_last_n_messages(tmp_path):
    user_id, path = _make_user(tmp_path)
    s = create_new_session(user_id, db_path=path)
    for i in range(CHAT_HISTORY_LIMIT + 5):
        save_message(s.id, "user", f"msg {i}", db_path=path)
    msgs = load_recent_messages(s.id, db_path=path)
    assert len(msgs) == CHAT_HISTORY_LIMIT
    assert msgs[-1]["content"] == f"msg {CHAT_HISTORY_LIMIT + 4}"


def test_load_recent_messages_returns_in_chronological_order(tmp_path):
    user_id, path = _make_user(tmp_path)
    s = create_new_session(user_id, db_path=path)
    for i in range(5):
        save_message(s.id, "user", f"msg {i}", db_path=path)
    msgs = load_recent_messages(s.id, db_path=path)
    for i, msg in enumerate(msgs):
        assert msg["content"] == f"msg {i}"


# ── Session listing ───────────────────────────────────────────────────────────

def test_list_sessions_returns_all_sessions_for_user(tmp_path):
    user_id, path = _make_user(tmp_path)
    create_new_session(user_id, db_path=path)
    create_new_session(user_id, db_path=path)
    assert len(list_sessions(user_id, db_path=path)) == 2


def test_list_sessions_returns_newest_first(tmp_path):
    user_id, path = _make_user(tmp_path)
    s1 = create_new_session(user_id, db_path=path)
    s2 = create_new_session(user_id, db_path=path)
    sessions = list_sessions(user_id, db_path=path)
    assert sessions[0].id == s2.id
    assert sessions[1].id == s1.id


def test_list_sessions_returns_empty_list_when_user_has_no_sessions(tmp_path):
    user_id, path = _make_user(tmp_path)
    assert list_sessions(user_id, db_path=path) == []


# ── Isolation ─────────────────────────────────────────────────────────────────

def test_sessions_are_isolated_per_user(tmp_path):
    alice_id, path = _make_user(tmp_path, "alice")
    register_user("bob", "s3cr3t!!", db_path=path)
    bob_id = get_user_id_by_username("bob", db_path=path)
    create_new_session(alice_id, db_path=path)
    assert list_sessions(bob_id, db_path=path) == []


def test_messages_in_one_session_do_not_appear_in_another(tmp_path):
    user_id, path = _make_user(tmp_path)
    s1 = create_new_session(user_id, db_path=path)
    s2 = create_new_session(user_id, db_path=path)
    save_message(s1.id, "user", "Only in session 1", db_path=path)
    assert load_session(s2.id, db_path=path) == []


# ── Rename ────────────────────────────────────────────────────────────────────

def test_rename_session_updates_title(tmp_path):
    user_id, path = _make_user(tmp_path)
    s = create_new_session(user_id, db_path=path)
    rename_session(s.id, "My Research Session", db_path=path)
    sessions = list_sessions(user_id, db_path=path)
    assert sessions[0].title == "My Research Session"


def test_rename_session_does_not_affect_other_sessions(tmp_path):
    user_id, path = _make_user(tmp_path)
    s1 = create_new_session(user_id, db_path=path)
    s2 = create_new_session(user_id, db_path=path)
    rename_session(s1.id, "Named Session", db_path=path)
    sessions = list_sessions(user_id, db_path=path)
    s2_fetched = next(s for s in sessions if s.id == s2.id)
    assert s2_fetched.title is None


# ── Delete ────────────────────────────────────────────────────────────────────

def test_delete_session_removes_session_from_list(tmp_path):
    user_id, path = _make_user(tmp_path)
    s = create_new_session(user_id, db_path=path)
    delete_session(s.id, db_path=path)
    assert list_sessions(user_id, db_path=path) == []


def test_delete_session_removes_its_messages(tmp_path):
    user_id, path = _make_user(tmp_path)
    s = create_new_session(user_id, db_path=path)
    save_message(s.id, "user", "Hello", db_path=path)
    delete_session(s.id, db_path=path)
    assert load_session(s.id, db_path=path) == []


def test_delete_session_does_not_affect_other_sessions(tmp_path):
    user_id, path = _make_user(tmp_path)
    s1 = create_new_session(user_id, db_path=path)
    s2 = create_new_session(user_id, db_path=path)
    save_message(s2.id, "user", "Keep me", db_path=path)
    delete_session(s1.id, db_path=path)
    msgs = load_session(s2.id, db_path=path)
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Keep me"

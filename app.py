# app.py
import time
import streamlit as st
import streamlit.components.v1 as components
from src.rag.factory import create_rag_pipeline
from src.db.sqlite import is_db_ready
from src.auth.service import login_user, register_user, verify_token, get_user_id_by_username
from src.config import SESSION_MAX_AGE
from src.chat.service import (
    get_or_create_session,
    create_new_session,
    save_message,
    load_session,
    load_recent_messages,
    list_sessions,
    rename_session,
    delete_session,
)

COOKIE_NAME = "auth_token"
COOKIE_MAX_AGE = 86400  # 24 hours


def format_context_label(doc_id: str, score: float) -> str:
    return f"Doc {doc_id} — score: {score:.2f}"


@st.cache_resource
def get_pipeline():
    return create_rag_pipeline()


def generate_session_title(first_message: str) -> str:
    prompt = (
        "Generate a short title (3 to 6 words) for a chat session that starts with this message. "
        "Return ONLY the title, no punctuation, no quotes:\n\n"
        f"{first_message}"
    )
    response = get_pipeline().llm_client.generate(prompt)
    return response.text.strip()


def _set_cookie_and_reload(name: str, value: str) -> None:
    """Set cookie and force a full page reload so st.context.cookies picks it up."""
    components.html(
        f"""<script>
        document.cookie = "{name}={value}; path=/; max-age={COOKIE_MAX_AGE}; SameSite=Lax";
        window.parent.location.reload();
        </script>""",
        height=0,
    )


def _remove_cookie_and_reload(name: str) -> None:
    """Remove cookie and force a full page reload."""
    components.html(
        f"""<script>
        document.cookie = "{name}=; path=/; max-age=0; SameSite=Lax";
        window.parent.location.reload();
        </script>""",
        height=0,
    )


def _render_auth_screen() -> None:
    st.title("LinkedIn RAG Assistant")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            token = login_user(username, password)
            if token:
                _set_cookie_and_reload(COOKIE_NAME, token)
            else:
                st.error("Invalid username or password.")

    with tab_register:
        with st.form("register_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register")
        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match.")
            else:
                try:
                    register_user(username, password)
                    st.success("Account created. Please log in.")
                except ValueError as e:
                    st.error(str(e))


st.set_page_config(page_title="LinkedIn RAG Assistant", layout="wide")

if not is_db_ready():
    st.error("Database is not initialised. Run `python scripts/migrate.py` and restart the app.")
    st.stop()

token = st.context.cookies.get(COOKIE_NAME)
username = verify_token(token) if token else None

if not username:
    _render_auth_screen()
else:
    now = time.time()
    last_activity = st.session_state.get("last_activity")
    if last_activity and (now - last_activity) > SESSION_MAX_AGE:
        st.session_state.clear()
        st.warning("Your session expired due to inactivity. Please log in again.")
        st.stop()
    st.session_state["last_activity"] = now

    st.session_state["auth_token"] = token

    if "user_id" not in st.session_state:
        st.session_state["user_id"] = get_user_id_by_username(username)

    user_id = st.session_state["user_id"]

    if "chat_session_id" not in st.session_state:
        chat_session = get_or_create_session(user_id)
        st.session_state["chat_session_id"] = chat_session.id
        st.session_state["messages"] = load_recent_messages(chat_session.id)

    st.title("LinkedIn RAG Assistant")

    with st.sidebar:
        st.write(f"Signed in as **{username}**")
        if st.button("Logout"):
            st.session_state.clear()
            _remove_cookie_and_reload(COOKIE_NAME)

        if st.button("New Conversation"):
            new_session = create_new_session(user_id)
            st.session_state["chat_session_id"] = new_session.id
            st.session_state["messages"] = []
            st.rerun()

        st.divider()
        st.subheader("Past Sessions")
        past_sessions = list_sessions(user_id)
        current_id = st.session_state["chat_session_id"]
        editing_id = st.session_state.get("editing_session_id")
        confirm_delete_id = st.session_state.get("confirm_delete_id")

        for s in past_sessions:
            label = s.title if s.title else s.created_at[:16].replace("T", " ")
            is_current = s.id == current_id

            if editing_id == s.id:
                with st.form(key=f"rename_form_{s.id}"):
                    new_title = st.text_input("Name", value=s.title or "")
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        save_clicked = st.form_submit_button("Save")
                    with col_cancel:
                        cancel_clicked = st.form_submit_button("Cancel")
                if save_clicked:
                    rename_session(s.id, new_title.strip() or label)
                    st.session_state.pop("editing_session_id", None)
                    st.rerun()
                if cancel_clicked:
                    st.session_state.pop("editing_session_id", None)
                    st.rerun()
            elif confirm_delete_id == s.id:
                st.warning(f'Delete "{label}"?')
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Delete", key=f"yes_{s.id}", type="primary"):
                        delete_session(s.id)
                        st.session_state.pop("confirm_delete_id", None)
                        if is_current:
                            remaining = [x for x in past_sessions if x.id != s.id]
                            if remaining:
                                st.session_state["chat_session_id"] = remaining[0].id
                                st.session_state["messages"] = load_session(remaining[0].id)
                            else:
                                new_s = create_new_session(user_id)
                                st.session_state["chat_session_id"] = new_s.id
                                st.session_state["messages"] = []
                        st.rerun()
                with col_no:
                    if st.button("Cancel", key=f"no_{s.id}"):
                        st.session_state.pop("confirm_delete_id", None)
                        st.rerun()
            else:
                col_label, col_menu = st.columns([5, 1])
                with col_label:
                    if is_current:
                        st.markdown(
                            f'<div style="background:#dce8ff;padding:6px 10px;'
                            f'border-radius:6px;border-left:3px solid #2563eb;'
                            f'font-size:14px;margin:2px 0;">{label}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        if st.button(label, key=f"session_{s.id}", use_container_width=True):
                            st.session_state["chat_session_id"] = s.id
                            st.session_state["messages"] = load_session(s.id)
                            st.rerun()
                with col_menu:
                    with st.popover("⋮", use_container_width=True):
                        if st.button("✎ Rename", key=f"edit_{s.id}", use_container_width=True):
                            st.session_state["editing_session_id"] = s.id
                            st.rerun()
                        if st.button("🗑 Delete", key=f"del_{s.id}", use_container_width=True):
                            st.session_state["confirm_delete_id"] = s.id
                            st.rerun()

        st.divider()
        st.header("Retrieved Context")
        last_assistant = next(
            (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
            None,
        )
        if last_assistant and last_assistant.get("contexts"):
            for ctx in last_assistant["contexts"]:
                with st.expander(format_context_label(ctx["doc_id"], ctx["score"])):
                    st.write(ctx["content"])
        else:
            st.write("No context retrieved yet.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question..."):
        current_session_id = st.session_state["chat_session_id"]

        if not st.session_state.messages:
            title = generate_session_title(prompt)
            rename_session(current_session_id, title)

        save_message(current_session_id, "user", prompt, contexts=[])
        st.session_state.messages.append({"role": "user", "content": prompt, "contexts": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        pipeline = get_pipeline()
        result = pipeline.run_with_context(prompt)
        contexts = [
            {"doc_id": c.doc_id, "score": c.score, "content": c.content}
            for c in result.contexts
        ]

        save_message(current_session_id, "assistant", result.answer, contexts=contexts)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "contexts": contexts,
        })

        with st.chat_message("assistant"):
            st.markdown(result.answer)

        st.rerun()

# app.py
import streamlit as st
import streamlit.components.v1 as components
from src.rag.factory import create_rag_pipeline
from src.db.sqlite import is_db_ready
from src.auth.service import login_user, register_user, verify_token

COOKIE_NAME = "auth_token"
COOKIE_MAX_AGE = 86400  # 24 hours


def format_context_label(doc_id: str, score: float) -> str:
    return f"Doc {doc_id} — score: {score:.2f}"


@st.cache_resource
def get_pipeline():
    return create_rag_pipeline()


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
    st.session_state["auth_token"] = token
    st.title("LinkedIn RAG Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.write(f"Signed in as **{username}**")
        if st.button("Logout"):
            st.session_state.clear()
            _remove_cookie_and_reload(COOKIE_NAME)
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
        st.session_state.messages.append({"role": "user", "content": prompt, "contexts": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        pipeline = get_pipeline()
        result = pipeline.run_with_context(prompt)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "contexts": [
                {"doc_id": c.doc_id, "score": c.score, "content": c.content}
                for c in result.contexts
            ],
        })

        with st.chat_message("assistant"):
            st.markdown(result.answer)

        st.rerun()

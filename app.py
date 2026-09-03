import streamlit as st
from src.rag.factory import create_rag_pipeline


def format_context_label(doc_id: str, score: float) -> str:
    return f"Doc {doc_id} — score: {score:.2f}"


@st.cache_resource
def get_pipeline():
    return create_rag_pipeline()


st.set_page_config(page_title="LinkedIn RAG Assistant", layout="wide")
st.title("LinkedIn RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
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

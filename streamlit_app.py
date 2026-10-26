"""Streamlit chat UI. Talks to the FastAPI service (streaming endpoint) so the
UI and the API share one pipeline and one conversation memory.

Run:  python -m uvicorn api:app   (in one terminal)
      streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="SafeSite Assistant", page_icon="🦺")
st.title("SafeSite Assistant")
st.caption("Answers only from the site-safety documents, with sources.")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.messages = []  # [{"role", "content", "sources"}]

with st.sidebar:
    if st.button("New conversation"):
        httpx.delete(f"{API_URL}/sessions/{st.session_state.session_id}")
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.messages = []
    try:
        st.json(httpx.get(f"{API_URL}/health", timeout=5).json())
    except httpx.HTTPError:
        st.error(f"API not reachable at {API_URL}")


def show_sources(sources: list[dict]) -> None:
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"**{s['file']}, p.{s['page']}**  \n{s['excerpt']}...")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources", []))

if question := st.chat_input("Ask a safety question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    final: dict = {}

    def stream_tokens() -> Iterator[str]:
        """Parse the SSE stream: yield `token` events, keep the `done` payload."""
        body = {"question": question, "session_id": st.session_state.session_id}
        with httpx.stream("POST", f"{API_URL}/ask/stream", json=body, timeout=120) as resp:
            if resp.status_code != 200:
                resp.read()
                yield f"Error {resp.status_code}: {resp.text}"
                return
            event = None
            for line in resp.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    if event == "token":
                        yield data
                    elif event == "done":
                        final.update(data)
                    elif event == "error":
                        yield f"\n\nLLM error: {data}"

    with st.chat_message("assistant"):
        answer = st.write_stream(stream_tokens())
        show_sources(final.get("sources", []))
        if "latency_ms" in final:
            st.caption(f"{final['latency_ms']} ms")
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": final.get("sources", [])})

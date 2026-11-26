"""Streamlit front end for PrivatAI-RAG.

Run (with the API already running on port 8000):
    py -m streamlit run ui.py

This file never decides which team you belong to. It only sends your login
token to the API, and the API looks up your team in its database.
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("PRIVATAI_API_URL", "http://localhost:8000")
TEAMS = ["Legal", "Finance", "HR"]  # keep in sync with app/config.py

st.set_page_config(page_title="PrivatAI-RAG", page_icon="🔒")


def api(method: str, path: str, **kwargs):
    """Call the backend, attaching the login token if we have one."""
    headers = {}
    if st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        r = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=300, **kwargs)
    except requests.ConnectionError:
        st.error("Can't reach the API. Is `py -m uvicorn app.main:app --reload` running?")
        st.stop()
    if r.status_code == 401 and st.session_state.get("token"):
        # Token expired (they last 60 minutes) or the account was disabled.
        st.session_state.clear()
        st.session_state.notice = "Your session expired. Please log in again."
        st.rerun()
    return r


def detail(r) -> str:
    try:
        return r.json().get("detail", "Something went wrong")
    except ValueError:
        return f"Server error ({r.status_code})"


def login_view():
    st.title("🔒 PrivatAI-RAG")
    st.caption("Private answers from your team's own documents.")
    if st.session_state.get("notice"):
        st.warning(st.session_state.pop("notice"))
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        r = api("POST", "/login", data={"username": username, "password": password})
        if r.status_code != 200:
            st.error(detail(r))
            return
        st.session_state.token = r.json()["access_token"]
        st.session_state.me = api("GET", "/me").json()
        st.session_state.messages = []
        st.rerun()


def sidebar():
    me = st.session_state.me
    with st.sidebar:
        st.subheader(f"👤 {me['username']}")
        st.write(f"Team: **{me['team']}**")
        if me["role"] == "admin":
            st.caption("Role: admin")
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()

        st.divider()
        st.subheader("Upload a document")
        uploaded = st.file_uploader("Text or Markdown file", type=["txt", "md"])
        if uploaded and st.button("Index document"):
            with st.spinner("Indexing... the first one can take a minute"):
                r = api("POST", "/documents", files={"file": (uploaded.name, uploaded.getvalue(), "text/plain")})
            if r.status_code == 201:
                st.success(f"Indexed {uploaded.name}")
                st.rerun()
            else:
                st.error(detail(r))

        st.divider()
        st.subheader(f"{me['team']} documents")
        docs = api("GET", "/documents").json()
        if not docs:
            st.caption("Nothing uploaded yet.")
        for d in docs:
            st.caption(f"📄 {d['filename']} · {d['uploaded_by']}")

        if me["role"] == "admin":
            st.divider()
            with st.expander("Admin: create a user"):
                with st.form("new_user", clear_on_submit=True):
                    username = st.text_input("Username")
                    password = st.text_input("Password (min 8 chars)", type="password")
                    team = st.selectbox("Team", TEAMS)
                    role = st.selectbox("Role", ["user", "admin"])
                    if st.form_submit_button("Create"):
                        r = api("POST", "/admin/users", json={
                            "username": username, "password": password, "team": team, "role": role})
                        if r.status_code == 201:
                            st.success(f"Created {username} in {team}")
                        else:
                            st.error(detail(r))


def chat_view():
    me = st.session_state.me
    st.title("🔒 PrivatAI-RAG")
    st.caption(f"Answers come only from **{me['team']}** documents.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("sources"):
                st.caption("Sources: " + ", ".join(m["sources"]))

    question = st.chat_input("Ask about your team's documents")
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            r = api("POST", "/ask", json={"question": question})
        if r.status_code == 200:
            data = r.json()
            answer = data["answer"]
            sources = [s["file"] for s in data["sources"]]
        else:
            answer, sources = f"⚠️ {detail(r)}", []
        st.markdown(answer)
        if sources:
            st.caption("Sources: " + ", ".join(sources))
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


if st.session_state.get("token"):
    sidebar()
    chat_view()
else:
    login_view()
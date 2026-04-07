class ChatMemory:

    def __init__(self):
        if "messages" not in self._session():
            self._session()["messages"] = []

    def _session(self):
        import streamlit as st
        return st.session_state

    def add(self, role, content):
        self._session()["messages"].append(
            {"role": role, "content": content}
        )

    def get(self):
        return self._session()["messages"]

    def clear(self):
        self._session()["messages"] = []
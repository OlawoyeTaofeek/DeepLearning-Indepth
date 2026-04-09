import streamlit as st

from client.utils import Utils
from client.openai_client import OpenAIClient
from memory.chat_memory import ChatMemory

# Page setup
st.set_page_config(
    page_title="DeepChat AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 DeepChat AI Assistant")
st.caption("Streaming OpenAI Chat with Memory")

# Load API
api_key = Utils.load_api_key()
client = OpenAIClient(api_key)
memory = ChatMemory()

# Sidebar
with st.sidebar:

    st.header("⚙️ Settings")

    model = st.selectbox(
        "Choose Model",
        ["gpt-5-mini", "gpt-5"]
    )

    max_tokens = st.slider(
        "Max Tokens",
        50,
        10000,
        200
    )

    if st.button("🗑 Clear Chat"):
        memory.clear()
        st.rerun()

# Display Chat Messages
for msg in memory.get():

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
user_input = st.chat_input("Type your message...")

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    memory.add("user", user_input)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ] + memory.get()

    # Stream AI response
    with st.chat_message("assistant"):

        response_placeholder = st.empty()
        full_response = ""

        for token in client.stream_chat(
            messages,
            model=model,
            max_tokens=max_tokens
        ):
            full_response += token
            response_placeholder.markdown(full_response)

    memory.add("assistant", full_response)
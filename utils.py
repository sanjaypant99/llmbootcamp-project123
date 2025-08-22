# utils.py
import streamlit as st

def display_msg(message: str, role: str):
    """Displays a message in the chat interface."""
    if role in ["user", "assistant"]:
        with st.chat_message(role):
            st.markdown(message)
    elif role == "system":
        st.info(f"ℹ️ {message}")
    elif role == "thinking":
        st.info(f"🤔 {message}")
    elif role == "reasoning":
        st.info(f"💭 AI Reasoning: {message}")
    elif role == "tool_execution":
        st.info(f"🔧 Executing tool: {message}")
    elif role == "tool_result":
        st.markdown(f"**🔧 Tool Result:**\n\n```\n{message}\n```")
    elif role == "error":
        st.error(f"❌ {message}")

def enable_chat_history(func):
    """Decorator to enable chat history and clear button."""
    def wrapper(*args, **kwargs):
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        st.sidebar.markdown("---")

        # Display existing messages (filtering out transient 'thinking' for cleaner history)
        for message in st.session_state.get("messages", []):
             if message["role"] not in ["thinking"]: # Exclude transient messages from history display
                 display_msg(message["content"], message["role"])

        func(*args, **kwargs)
    return wrapper

# Optional: Helper to check if clients are configured
def is_openai_configured():
    return bool(st.session_state.get("openai_client"))

def is_mcp_configured():
    return bool(st.session_state.get("connected", False))

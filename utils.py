# utils.py
import streamlit as st
import os

def configure_openai_api_key():
    """Checks for OpenAI API key in environment variables or Streamlit secrets."""
    api_key = None
    try:
        # Try Streamlit secrets first (recommended for Streamlit apps)
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
            os.environ["OPENAI_API_KEY"] = api_key # Set for openai client
    except Exception:
        pass

    # Fallback to environment variable
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("❌ OpenAI API key (`OPENAI_API_KEY`) not found.")
        st.info("Please set it in Streamlit Secrets or as an environment variable.")
        st.stop() # Halts the app
    return api_key

def configure_mcp_server():
    """Gets the MCP server URL from Streamlit secrets or environment variables."""
    mcp_url = None
    try:
        # Try Streamlit secrets first (recommended)
        if hasattr(st, 'secrets') and 'MCP_SERVER_URL' in st.secrets:
            mcp_url = st.secrets["MCP_SERVER_URL"]
    except Exception:
        pass # Secrets not configured or key not found

    # Fallback to environment variable
    if not mcp_url:
        mcp_url = os.getenv("MCP_SERVER_URL")

    # Do NOT use a hardcoded default. Require user configuration.
    # The application logic will handle the case where mcp_url is None/empty.

    return mcp_url # Returns None or the URL string

def display_msg(message: str, role: str):
    """Displays a message in the chat interface."""
    # Using Streamlit's built-in chat_message for cleaner handling
    if role in ["user", "assistant"]:
        with st.chat_message(role):
            st.markdown(message)
    elif role == "system":
        st.info(f"ℹ️ {message}")
    elif role == "thinking":
        # Using an empty chat message or st.info for transient states
        # For simplicity, using st.info, but could be more sophisticated
        st.info(f"🤔 {message}")
    elif role == "reasoning":
        st.info(f"💭 AI Reasoning: {message}")
    elif role == "tool_execution":
        st.info(f"🔧 Executing tool: {message}")
    elif role == "tool_result":
        # Use st.markdown for better formatting of results
        st.markdown(f"**🔧 Tool Result:**\n\n```\n{message}\n```")
        # Or if you want a custom styled box:
        # st.markdown(f"""
        # <div class="tool-result">
        #     <strong>🔧 Tool Result:</strong><br>
        #     <pre>{message}</pre>
        # </div>
        # """, unsafe_allow_html=True)
    elif role == "error":
        st.error(f"❌ {message}")

def enable_chat_history(func):
    """Decorator to enable chat history and clear button."""
    def wrapper(*args, **kwargs):
        # Clear chat button logic
        if st.sidebar.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        st.sidebar.markdown("---")

        # Display chat messages from history at the start of the main area
        # This needs to be inside the main area, not sidebar.
        # The original code displayed messages *before* the input.
        # This decorator approach places it inside the main function call.
        # A simple loop is fine here.
        # Note: The original display logic was intertwined with the message type.
        # We'll iterate and call display_msg for each.
        # However, 'thinking', 'reasoning', 'tool_execution' were transient.
        # For history display, we might want to filter or format them differently.
        # Let's keep it simple and display all stored messages.
        # But thinking/reasoning might clutter history. Let's filter them out for history display.
        # Actually, let's just display them as they are, using display_msg.
        # The display_msg function handles the types.

        # Display existing messages (this replaces the loop in the original main)
        # Ensure this only runs inside the main chat container.
        # The decorator places this before the main function logic.
        for message in st.session_state.get("messages", []):
             # Avoid re-displaying transient 'thinking' messages from previous runs if they lingered
             if message["role"] not in ["thinking"]: # Exclude transient messages from history display
                 display_msg(message["content"], message["role"])


        # Execute the wrapped function (e.g., the main chat logic)
        func(*args, **kwargs)

    return wrapper

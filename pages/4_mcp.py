# main_mcp_chatbot.py

import asyncio
import json
import os
import time
from typing import List, Dict, Any, Optional

import streamlit as st
# Assuming utils.py is in the same directory or properly configured in PYTHONPATH
import utils
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
import openai


# --- Main Chatbot Class ---

class MCPChatbot:
    def __init__(self):
        # Configure OpenAI (will stop if not found)
        utils.configure_openai_api_key()
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE") # Handles custom base URL if needed
        )
        # Attempt to configure MCP Server URL (will be None if not found)
        self.mcp_server_url = utils.configure_mcp_server()
        # Allow OpenAI model override via env var
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4")

        # Initialize session state variables if they don't exist
        # This can also be done in main, but initializing here ensures they are ready early.
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "mcp_client" not in st.session_state:
            st.session_state.mcp_client = None
        if "available_tools" not in st.session_state:
            st.session_state.available_tools = []
        if "connected" not in st.session_state:
            st.session_state.connected = False
        # conversation_history wasn't heavily used, might be optional
        # if "conversation_history" not in st.session_state:
        #     st.session_state.conversation_history = []


    def _get_tools_description(self) -> str:
        """Get a formatted description of available MCP tools for the AI."""
        if not st.session_state.get("available_tools", []):
            return "No tools are currently available."

        tools_desc = "Available MCP tools:\n"
        for tool in st.session_state.available_tools:
            tools_desc += f"- {tool.name}: {tool.description}\n"
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                # Limit schema size for prompt context to avoid exceeding token limits
                schema_str = json.dumps(tool.inputSchema, indent=2)
                if len(schema_str) > 1000: # Arbitrary limit
                     schema_str = schema_str[:1000] + "... (schema truncated)"
                tools_desc += f"  Input schema: {schema_str}\n"
        return tools_desc

    def _parse_tool_result(self, tool_result: Any) -> str:
        """Parse MCP tool result into a user-friendly format."""
        try:
            if hasattr(tool_result, 'content'):
                content = tool_result.content
                if isinstance(content, list):
                    parsed_items = [getattr(item, 'text', str(item)) for item in content]
                    return "\n".join(parsed_items)
                elif hasattr(content, 'text'):
                    return content.text
                elif isinstance(content, str):
                    return content
                elif isinstance(content, dict):
                    return "\n".join([f"**{k}**: {v}" for k, v in content.items()])
                else:
                    return str(content)
            elif hasattr(tool_result, 'isError'):
                prefix = "❌ **Error**" if tool_result.isError else "✅ **Success**"
                msg = getattr(tool_result, 'content', 'Operation completed.' if not tool_result.isError else 'Unknown error occurred')
                return f"{prefix}: {msg}"
            elif isinstance(tool_result, str):
                return tool_result
            elif isinstance(tool_result, dict):
                formatted_lines = []
                for key, value in tool_result.items():
                    if key == 'content':
                        return self._parse_tool_result(value) # Recurse for nested content
                    else:
                        formatted_lines.append(f"**{key}**: {value}")
                return "\n".join(formatted_lines)
            else:
                result_str = str(tool_result)
                if 'object at 0x' in result_str:
                    return f"Tool executed successfully. Result: {type(tool_result).__name__}"
                return result_str
        except Exception as e:
            return f"⚠️ **Result parsing error**: {str(e)}\n\nRaw result: {str(tool_result)}"

    def _generate_result_summary(self, tool_name: str, tool_args: dict, tool_result: str, user_request: str) -> str:
        """Generate a conversational summary of the tool result using OpenAI."""
        summary_prompt = f"""You are an AI assistant helping a user understand the results of an MCP tool execution.

User's original request: "{user_request}"
Tool executed: {tool_name}
Tool arguments: {json.dumps(tool_args, indent=2)}
Tool result: {tool_result}

Please provide a conversational, friendly paragraph explaining:
1. What the tool did
2. What the results mean
3. Whether it was successful or if there were any issues
4. Any next steps or additional information that might be helpful

Write in a natural, conversational tone as if you're explaining to a friend. Be concise but informative.
"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that explains technical results in a friendly, conversational way."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content or f"The {tool_name} tool completed its task. Here's what it found."
        except Exception as e:
             # Graceful fallback if summary generation fails
            st.warning(f"Could not generate summary for tool result: {e}")
            return f"I executed the {tool_name} tool for you. The operation completed. Here is the raw result:\n\n{tool_result}\n\nLet me know if you need help understanding this!"

    async def _call_mcp_tool(self, tool_name: str, arguments: dict):
        """Call an MCP tool with the given arguments."""
        if not st.session_state.get("mcp_client"):
             raise Exception("MCP client is not initialized. Please connect to the server first.")
        try:
            async with st.session_state.mcp_client:
                result = await st.session_state.mcp_client.call_tool(tool_name, arguments)
                return result
        except Exception as e:
            # Provide more context in the error
            raise Exception(f"Error while calling tool '{tool_name}' with args {arguments}: {str(e)}")

    def _analyze_user_intent(self, user_message: str, tools_description: str) -> Dict[str, Any]:
        """Use OpenAI to analyze user intent and determine if MCP tools should be used."""
        system_prompt = f"""You are an AI assistant that helps users interact with MCP (Model Context Protocol) tools.

Available MCP tools:
{tools_description}

Your job is to:
1. Understand the user's request.
2. Determine if any of the available MCP tools should be used to fulfill the request.
3. If a tool should be used, specify which tool (`tool_name`) and provide the necessary arguments (`tool_arguments`) as a JSON object.
4. Provide your reasoning (`reasoning`).
5. Give a conversational response (`response`) to the user, especially if no tool is needed or before/after tool execution.

Respond ONLY in JSON format with this structure:
{{
    "needs_tool": true/false,
    "tool_name": "exact_tool_name_if_needed", // Required if needs_tool is true
    "tool_arguments": {{ "arg1": "value1", "arg2": "value2" }}, // Required if needs_tool is true, empty dict {{}} if no args needed
    "reasoning": "explanation of your decision",
    "response": "conversational response to the user"
}}

If `needs_tool` is false, you must still provide `reasoning` and `response`.
Ensure `tool_arguments` is always a valid JSON object, even if empty {{}}.
Ensure `tool_name` exactly matches the name listed in the tools description.
"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                # Adding a stop sequence can sometimes help with JSON parsing if the model tends to add extra text
                # stop=["\n\n"] # Example, might need tuning
            )

            content = response.choices[0].message.content or "{}"
            try:
                parsed_response = json.loads(content)
                # Basic validation
                if not isinstance(parsed_response, dict):
                    raise ValueError("Response is not a JSON object")
                return parsed_response
            except (json.JSONDecodeError, ValueError, KeyError):
                # If JSON parsing fails, log the issue and provide a fallback
                st.error(f"AI Intent Analysis Error: Could not parse AI response as JSON.\nRaw AI Output:\n```\n{content}\n```")
                return {
                    "needs_tool": False,
                    "reasoning": "AI response was not in the expected JSON format.",
                    "response": "I'm having trouble processing your request right now. Could you please try rephrasing it?"
                }
        except Exception as e:
            st.error(f"AI Intent Analysis Error: {e}")
            return {
                "needs_tool": False,
                "reasoning": f"Error occurred while analyzing intent: {str(e)}",
                "response": "I'm currently unable to understand your request. Please try again later."
            }


    async def _process_user_message(self, user_message: str):
        """Process user message with AI routing."""
        # Add and display thinking indicator immediately
        thinking_msg_id = st.info("🤔 Analyzing your request...")
        st.session_state.messages.append({"role": "thinking", "content": "Analyzing your request...", "timestamp": time.time()})

        # Get tools description
        tools_desc = self._get_tools_description()

        # Analyze user intent with OpenAI
        intent_analysis = self._analyze_user_intent(user_message, tools_desc)

        # Remove the temporary thinking indicator from the UI
        # Note: Streamlit's `st.info` doesn't return an easy handle to remove it.
        # A common workaround is to use `st.empty()` and `.info()` on it, then `.empty()` to remove.
        # However, for simplicity here, we'll just proceed. The history display logic in `enable_chat_history`
        # filters out 'thinking' messages, so they won't clutter the persistent history.
        # If you want to remove the *visual* indicator, you'd need the `st.empty()` pattern in `utils.display_msg` for 'thinking'.

        # Filter out 'thinking' from messages to keep history clean (optional, as display filter handles it)
        # st.session_state.messages = [msg for msg in st.session_state.messages if msg["role"] != "thinking"]

        # Add reasoning to messages and display it
        reasoning_content = intent_analysis.get('reasoning', 'No reasoning provided.')
        st.session_state.messages.append({"role": "reasoning", "content": reasoning_content, "timestamp": time.time()})
        utils.display_msg(reasoning_content, "reasoning")

        response_content = intent_analysis.get("response", "I processed your request.")

        # If tool usage is needed, execute the tool
        if intent_analysis.get("needs_tool", False):
            tool_name = intent_analysis.get("tool_name")
            tool_args = intent_analysis.get("tool_arguments", {})

            if tool_name:
                try:
                    # Execute the MCP tool
                    exec_msg = f"{tool_name}"
                    if tool_args:
                        exec_msg += f" with arguments: {json.dumps(tool_args, indent=2)}"
                    st.session_state.messages.append({"role": "tool_execution", "content": exec_msg, "timestamp": time.time()})
                    utils.display_msg(exec_msg, "tool_execution")

                    tool_result = await self._call_mcp_tool(tool_name, tool_args)

                    # Parse tool result
                    parsed_result = self._parse_tool_result(tool_result)
                    st.session_state.messages.append({"role": "tool_result", "content": parsed_result, "timestamp": time.time()})
                    utils.display_msg(parsed_result, "tool_result")

                    # Generate a conversational summary of the result
                    result_summary = self._generate_result_summary(tool_name, tool_args, parsed_result, user_message)
                    response_content = result_summary # Update final response sent to user

                except Exception as e:
                    error_msg = str(e)
                    st.session_state.messages.append({"role": "error", "content": error_msg, "timestamp": time.time()})
                    utils.display_msg(error_msg, "error")
                    response_content = f"I tried to use the {tool_name} tool to help with your request, but I encountered an issue: {error_msg}. Let me know if you'd like me to try a different approach!"
            else:
                # needs_tool was True but no tool_name provided by AI
                error_msg = "AI indicated a tool was needed but didn't specify which one."
                st.session_state.messages.append({"role": "error", "content": error_msg, "timestamp": time.time()})
                utils.display_msg(error_msg, "error")
                response_content = "I'm a bit confused about which tool to use. Could you rephrase your request?"

        return response_content

    def _setup_sidebar(self):
        """Setup the sidebar for MCP configuration and display."""
        with st.sidebar:
            st.header("🔧 MCP Server Configuration")

            # Get the configured URL (could be None)
            configured_url = self.mcp_server_url

            # Always show the input field, pre-filled if a URL was found
            server_url = st.text_input(
                "Server URL:",
                value=configured_url if configured_url else "",
                placeholder="Enter your MCP server URL here...",
                help="Enter the MCP server URL. This can also be set via `MCP_SERVER_URL` in Streamlit Secrets or environment variables."
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Connect", type="primary"):
                    if server_url and server_url.strip():
                        with st.spinner("Connecting to MCP server..."):
                            try:
                                transport = StreamableHttpTransport(server_url.strip())
                                # Store the *actual connected URL* in session state
                                st.session_state.mcp_client = Client(transport=transport)
                                st.session_state.connected_url = server_url.strip()

                                # Test connection by listing tools
                                async def get_tools():
                                    async with st.session_state.mcp_client:
                                        tools = await st.session_state.mcp_client.list_tools()
                                        return tools

                                tools = asyncio.run(get_tools())
                                st.session_state.available_tools = tools
                                st.session_state.connected = True
                                st.success("Connected successfully!")

                                conn_msg = f"🔗 Connected to MCP server. Found {len(tools)} available tool(s). You can now chat naturally!"
                                st.session_state.messages.append({"role": "system", "content": conn_msg, "timestamp": time.time()})
                                utils.display_msg(conn_msg, "system")
                                # st.rerun() # Often not strictly necessary after button press, but can ensure UI updates

                            except Exception as e:
                                st.error(f"Connection failed: {str(e)}")
                                st.session_state.connected = False
                                st.session_state.mcp_client = None
                                # Clear tools on failed connection
                                st.session_state.available_tools = []
                                # Add error to chat
                                fail_msg = f"Failed to connect to MCP server: {e}"
                                st.session_state.messages.append({"role": "error", "content": fail_msg, "timestamp": time.time()})
                                utils.display_msg(fail_msg, "error")
                    else:
                        st.error("Please enter a server URL")

            with col2:
                if st.button("🔌 Disconnect"):
                    st.session_state.mcp_client = None
                    st.session_state.connected = False
                    st.session_state.available_tools = []
                    st.session_state.connected_url = None # Clear stored URL
                    st.info("Disconnected from server")
                    disconn_msg = "🔴 Disconnected from MCP server."
                    st.session_state.messages.append({"role": "system", "content": disconn_msg, "timestamp": time.time()})
                    utils.display_msg(disconn_msg, "system")
                    # st.rerun() # Ensure UI reflects disconnection

            # Display connection status
            if st.session_state.get("connected", False):
                st.success("🟢 Connected")
                current_url = st.session_state.get("connected_url", "Unknown")
                st.markdown(f"**Connected to:** `{current_url}`")
            else:
                st.error("🔴 Disconnected")
                if not configured_url and not server_url:
                     st.warning("❗ MCP Server URL not configured. Please enter a URL or set `MCP_SERVER_URL` in Secrets/Environment.")

            # Display available tools if connected
            if st.session_state.get("available_tools", []):
                st.subheader("🛠️ Available Tools")
                # Use a container or expander for better organization
                with st.container(border=True):
                    for tool in st.session_state.available_tools:
                        with st.expander(f"**{tool.name}**", expanded=False): # Start collapsed
                            st.write(f"**Description:** {tool.description}")
                            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                                st.write("**Input Schema:**")
                                # Use st.json for better display of JSON
                                st.json(tool.inputSchema)
            elif st.session_state.get("connected", False):
                 st.info("No tools found on the connected server.")
            else:
                 st.info("Connect to an MCP server to see available tools.")

            # AI Settings
            st.subheader("🧠 AI Settings")
            st.info(f"Using OpenAI Model: `{self.openai_model}` for intelligent routing")
            # Note: API key check is done in __init__ via utils.configure_openai_api_key()


    @utils.enable_chat_history
    def main(self):
        """Main application logic."""
        # Configure Streamlit page
        st.set_page_config(page_title="AI-Powered MCP Chatbot", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
        st.title("🤖 AI-Powered MCP Chatbot")
        st.markdown("Chat naturally! I'll understand your requests and use connected MCP tools when needed.")

        # Setup sidebar (handles connection logic)
        self._setup_sidebar()

        # --- Main Chat Logic ---
        # Check connection status
        is_connected = st.session_state.get("connected", False)

        # Disable input if not connected
        user_input_disabled = not is_connected

        # Chat input
        if prompt := st.chat_input("Ask me anything! I can help you use MCP tools naturally...", disabled=user_input_disabled):
            # Add user message to state and display
            st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": time.time()})
            utils.display_msg(prompt, "user")

            # Process the message if connected
            if is_connected and st.session_state.get("mcp_client"):
                try:
                    # Process with AI routing (async)
                    # Ensure event loop compatibility (Streamlit >= 1.28 handles this better, but run always works)
                    response = asyncio.run(self._process_user_message(prompt))

                    # Add and display assistant response
                    st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": time.time()})
                    utils.display_msg(response, "assistant")

                except Exception as e:
                    # Handle unexpected errors during processing
                    error_msg = f"An unexpected error occurred while processing your message: {str(e)}"
                    st.session_state.messages.append({"role": "error", "content": error_msg, "timestamp": time.time()})
                    utils.display_msg(error_msg, "error")
            else:
                # This case should ideally be prevented by disabling the input,
                # but handle gracefully if it occurs.
                conn_error_msg = "Please connect to an MCP server first before sending messages."
                st.session_state.messages.append({"role": "error", "content": conn_error_msg, "timestamp": time.time()})
                utils.display_msg(conn_error_msg, "error")

        # Footer
        st.markdown("---")
        st.markdown("""
        💡 **How it works:**
        - Chat naturally in plain English.
        - The AI analyzes your request and determines if connected MCP tools are needed.
        - Tools are automatically executed when relevant.
        - You get both conversational responses and summarized tool results.

        **Example requests:**
        - "List all the tools I have available."
        - "Perform a specific action using one of my tools." (Be specific about the tool/action if known)
        - "Help me understand what this MCP setup can do."
        """)


# --- Entry Point ---
if __name__ == "__main__":
    chatbot = MCPChatbot()
    chatbot.main()

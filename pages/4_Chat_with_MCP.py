# main_mcp_chatbot.py

import asyncio
import json
import os
import time
from typing import List, Dict, Any, Optional

import streamlit as st
import utils # Assuming utils.py is in the same directory
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
import openai


# --- Main Chatbot Class ---

class MCPChatbot:
    def __init__(self):
        # Initialize session state variables if they don't exist
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Ensures all necessary session state variables are initialized."""
        defaults = {
            "messages": [],
            "mcp_client": None,
            "available_tools": [],
            "connected": False,
            "connected_url": None,
            "openai_client": None,
            "openai_model": "gpt-4", # Default model
            "openai_api_key": "", # Store key in session state
            "mcp_server_url": "", # Store URL in session state
        }
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def _get_tools_description(self) -> str:
        """Get a formatted description of available MCP tools for the AI."""
        if not st.session_state.get("available_tools", []):
            return "No tools are currently available."

        tools_desc = "Available MCP tools:\n"
        for tool in st.session_state.available_tools:
            tools_desc += f"- {tool.name}: {tool.description}\n"
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                schema_str = json.dumps(tool.inputSchema, indent=2)
                if len(schema_str) > 1000:
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
                        return self._parse_tool_result(value)
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
        if not st.session_state.get("openai_client"):
            return f"OpenAI not configured. Raw result from {tool_name}:\n\n{tool_result}"

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
            response = st.session_state.openai_client.chat.completions.create(
                model=st.session_state.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that explains technical results in a friendly, conversational way."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content or f"The {tool_name} tool completed its task."
        except Exception as e:
            return f"I executed the {tool_name} tool. The operation completed. Here is the raw result:\n\n{tool_result}\n\n(Explanation generation failed: {e})"

    async def _call_mcp_tool(self, tool_name: str, arguments: dict):
        """Call an MCP tool with the given arguments."""
        if not st.session_state.get("mcp_client"):
             raise Exception("MCP client is not initialized. Please connect to the server first.")
        try:
            async with st.session_state.mcp_client:
                result = await st.session_state.mcp_client.call_tool(tool_name, arguments)
                return result
        except Exception as e:
            raise Exception(f"Error calling tool '{tool_name}': {str(e)}")

    def _analyze_user_intent(self, user_message: str, tools_description: str) -> Dict[str, Any]:
        """Use OpenAI to analyze user intent and determine if MCP tools should be used."""
        if not st.session_state.get("openai_client"):
             return {
                "needs_tool": False,
                "reasoning": "OpenAI is not configured.",
                "response": "I cannot process your request because the OpenAI API is not configured. Please enter your OpenAI API key in the sidebar."
            }

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
            response = st.session_state.openai_client.chat.completions.create(
                model=st.session_state.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
            )

            content = response.choices[0].message.content or "{}"
            try:
                parsed_response = json.loads(content)
                if not isinstance(parsed_response, dict):
                    raise ValueError("Response is not a JSON object")
                return parsed_response
            except (json.JSONDecodeError, ValueError, KeyError):
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
                "response": "I'm currently unable to understand your request. Please check your OpenAI configuration and try again later."
            }


    async def _process_user_message(self, user_message: str):
        """Process user message with AI routing."""
        thinking_msg_id = st.info("🤔 Analyzing your request...")
        st.session_state.messages.append({"role": "thinking", "content": "Analyzing your request...", "timestamp": time.time()})

        tools_desc = self._get_tools_description()
        intent_analysis = self._analyze_user_intent(user_message, tools_desc)

        # Filter out 'thinking' from messages for cleaner history
        st.session_state.messages = [msg for msg in st.session_state.messages if msg["role"] != "thinking"]

        reasoning_content = intent_analysis.get('reasoning', 'No reasoning provided.')
        st.session_state.messages.append({"role": "reasoning", "content": reasoning_content, "timestamp": time.time()})
        utils.display_msg(reasoning_content, "reasoning")

        response_content = intent_analysis.get("response", "I processed your request.")

        if intent_analysis.get("needs_tool", False):
            tool_name = intent_analysis.get("tool_name")
            tool_args = intent_analysis.get("tool_arguments", {})

            if tool_name:
                try:
                    exec_msg = f"{tool_name}"
                    if tool_args:
                        exec_msg += f" with arguments: {json.dumps(tool_args, indent=2)}"
                    st.session_state.messages.append({"role": "tool_execution", "content": exec_msg, "timestamp": time.time()})
                    utils.display_msg(exec_msg, "tool_execution")

                    tool_result = await self._call_mcp_tool(tool_name, tool_args)

                    parsed_result = self._parse_tool_result(tool_result)
                    st.session_state.messages.append({"role": "tool_result", "content": parsed_result, "timestamp": time.time()})
                    utils.display_msg(parsed_result, "tool_result")

                    result_summary = self._generate_result_summary(tool_name, tool_args, parsed_result, user_message)
                    response_content = result_summary

                except Exception as e:
                    error_msg = str(e)
                    st.session_state.messages.append({"role": "error", "content": error_msg, "timestamp": time.time()})
                    utils.display_msg(error_msg, "error")
                    response_content = f"I tried to use the {tool_name} tool, but encountered an issue: {error_msg}."
            else:
                error_msg = "AI indicated a tool was needed but didn't specify which one."
                st.session_state.messages.append({"role": "error", "content": error_msg, "timestamp": time.time()})
                utils.display_msg(error_msg, "error")
                response_content = "I'm confused about which tool to use. Could you rephrase?"

        return response_content

    def _setup_sidebar(self):
        """Setup the sidebar for OpenAI and MCP configuration."""
        with st.sidebar:
            st.header("⚙️ Configuration")

            # --- OpenAI Configuration ---
            st.subheader("🧠 OpenAI Settings")
            # Pre-fill with session state or env var
            api_key_from_state = st.session_state.get("openai_api_key", "")
            api_key_from_env = os.getenv("OPENAI_API_KEY", "")
            initial_api_key = api_key_from_state if api_key_from_state else api_key_from_env

            openai_api_key = st.text_input("OpenAI API Key:", value=initial_api_key, type="password", help="Enter your OpenAI API key. You can also set OPENAI_API_KEY in your environment.")

            # Model selection
            available_models = ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"] # Add more as needed
            selected_model = st.selectbox(
                "Model:",
                options=available_models,
                index=available_models.index(st.session_state.get("openai_model", "gpt-4")),
                help="Select the OpenAI model to use."
            )

            # --- MCP Configuration ---
            st.subheader("🔧 MCP Server Settings")
            # Pre-fill with session state or env var
            url_from_state = st.session_state.get("mcp_server_url", "")
            url_from_env = os.getenv("MCP_SERVER_URL", "")
            initial_mcp_url = url_from_state if url_from_state else url_from_env

            mcp_server_url = st.text_input("MCP Server URL:", value=initial_mcp_url, type="default", help="Enter your MCP server URL. You can also set MCP_SERVER_URL in your environment.")

            # Action Buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save & Initialize", type="primary"):
                    # Update session state
                    st.session_state.openai_api_key = openai_api_key
                    st.session_state.openai_model = selected_model
                    st.session_state.mcp_server_url = mcp_server_url

                    # Initialize OpenAI Client
                    if openai_api_key:
                        try:
                            st.session_state.openai_client = openai.OpenAI(api_key=openai_api_key)
                            st.success("✅ OpenAI initialized!")
                        except Exception as e:
                             st.error(f"❌ Failed to initialize OpenAI client: {e}")
                             st.session_state.openai_client = None
                    else:
                        st.warning("⚠️ OpenAI API key is empty. AI features will be disabled.")
                        st.session_state.openai_client = None

                    # Initialize MCP Client (only if URL is provided)
                    if mcp_server_url:
                        try:
                            transport = StreamableHttpTransport(mcp_server_url)
                            st.session_state.mcp_client = Client(transport=transport)
                            # Test connection by listing tools
                            async def get_tools():
                                async with st.session_state.mcp_client:
                                    tools = await st.session_state.mcp_client.list_tools()
                                    return tools
                            tools = asyncio.run(get_tools())
                            st.session_state.available_tools = tools
                            st.session_state.connected = True
                            st.session_state.connected_url = mcp_server_url
                            st.success("✅ MCP connected!")
                            conn_msg = f"🔗 Connected to MCP server. Found {len(tools)} available tool(s)."
                            st.session_state.messages.append({"role": "system", "content": conn_msg, "timestamp": time.time()})
                            utils.display_msg(conn_msg, "system")
                        except Exception as e:
                             st.error(f"❌ Failed to connect to MCP server: {e}")
                             st.session_state.mcp_client = None
                             st.session_state.connected = False
                             st.session_state.available_tools = []
                             st.session_state.connected_url = None
                    else:
                         st.info("ℹ️ MCP URL is empty. Connect to a server later if needed.")
                         st.session_state.mcp_client = None
                         st.session_state.connected = False
                         st.session_state.available_tools = []
                         st.session_state.connected_url = None

                    st.rerun() # Refresh UI to reflect changes

            with col2:
                if st.button("🔌 Disconnect MCP"):
                    st.session_state.mcp_client = None
                    st.session_state.connected = False
                    st.session_state.available_tools = []
                    st.session_state.connected_url = None
                    st.info("🔴 Disconnected from MCP server.")
                    disconn_msg = "🔴 Disconnected from MCP server."
                    st.session_state.messages.append({"role": "system", "content": disconn_msg, "timestamp": time.time()})
                    utils.display_msg(disconn_msg, "system")
                    st.rerun()

            # --- Display Status ---
            st.markdown("---")
            # OpenAI Status
            if st.session_state.get("openai_client"):
                st.success("🟢 OpenAI Configured")
                st.markdown(f"**Model:** `{st.session_state.openai_model}`")
            else:
                st.error("🔴 OpenAI Not Configured")

            # MCP Status
            if st.session_state.get("connected", False):
                st.success("🟢 MCP Connected")
                current_url = st.session_state.get("connected_url", "Unknown")
                st.markdown(f"**Connected to:** `{current_url}`")
            else:
                st.error("🔴 MCP Disconnected")

            # Display available tools if connected
            if st.session_state.get("available_tools", []):
                st.subheader("🛠️ Available Tools")
                with st.container(border=True):
                    for tool in st.session_state.available_tools:
                        with st.expander(f"**{tool.name}**", expanded=False):
                            st.write(f"**Description:** {tool.description}")
                            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                                st.write("**Input Schema:**")
                                st.json(tool.inputSchema)
            elif st.session_state.get("connected", False):
                 st.info("No tools found on the connected server.")
            else:
                 st.info("Connect to an MCP server to see available tools.")


    @utils.enable_chat_history
    def main(self):
        """Main application logic."""
        st.set_page_config(page_title="AI-Powered MCP Chatbot", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
        st.title("🤖 AI-Powered MCP Chatbot")
        st.markdown("Configure OpenAI and MCP in the sidebar, then chat naturally!")

        self._setup_sidebar()

        # Check configuration status
        openai_ok = utils.is_openai_configured()
        mcp_ok = utils.is_mcp_configured() # Checks st.session_state.connected

        # Determine input state
        # Disable chat if OpenAI is not configured (needed for AI logic)
        chat_disabled = not openai_ok

        # Chat input
        if prompt := st.chat_input("Ask me anything...", disabled=chat_disabled):
            st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": time.time()})
            utils.display_msg(prompt, "user")

            if openai_ok: # Double-check inside the handler
                try:
                    response = asyncio.run(self._process_user_message(prompt))
                    st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": time.time()})
                    utils.display_msg(response, "assistant")
                except Exception as e:
                    error_msg = f"Error processing message: {str(e)}"
                    st.session_state.messages.append({"role": "error", "content": error_msg, "timestamp": time.time()})
                    utils.display_msg(error_msg, "error")
            else:
                config_error_msg = "Please configure your OpenAI API key in the sidebar first."
                st.session_state.messages.append({"role": "error", "content": config_error_msg, "timestamp": time.time()})
                utils.display_msg(config_error_msg, "error")

        # Footer
        st.markdown("---")
        st.markdown("""
        💡 **Instructions:**
        1.  Enter your **OpenAI API Key** and select a model in the sidebar.
        2.  (Optional) Enter your **MCP Server URL**.
        3.  Click **Save & Initialize**.
        4.  Start chatting! The AI will use tools if needed and if connected.
        """)


# --- Entry Point ---
if __name__ == "__main__":
    chatbot = MCPChatbot()
    chatbot.main()

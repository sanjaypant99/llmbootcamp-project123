import utils
import streamlit as st
import os
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool, initialize_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.utilities import ArxivAPIWrapper


from langchain_community.utilities import WikipediaAPIWrapper

BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"

st.set_page_config(page_title="ChatWeb", page_icon="🌐")
st.header('Chatbot with Web Browser Access')
st.write('Equipped with internet agent, enables users to ask questions about recent events')

class ChatbotTools:
    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-4o-mini"
    
    def setup_agent(self):
        # Check if Bing API key is available
        bing_subscription_key = os.getenv('BING_SUBSCRIPTION_KEY')
        wiki_agent = WikipediaQueryRun(api_wrapper = WikipediaAPIWrapper())
        python_repl = PythonREPLTool()
        arxiv = ArxivAPIWrapper()
        if bing_subscription_key:
            # Use BingSearch if API key is available
            bing_search = BingSearchAPIWrapper()
            tools = [
                Tool(
                    name="BingSearch",
                    func=bing_search.run,
                    description="Useful for when you need to answer questions about current events. You should ask targeted questions",
                ),
                Tool(
                name="Wikipedia",
                func=wiki_agent.run,
                description="Useful for when you need to query about a specific topic, person, or event. You should ask targeted questions",
            ),
            Tool(
                name="Python REPL",
                func=python_repl.run,
                description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you expect output it should be printed out.",
            ),
            Tool(
                name="Arxiv",
                func=arxiv.run,
                description="Useful for when you need to answer questions about research papers, scientific articles, and preprints etc",
           )
            ]
        else:
            # Fallback to DuckDuckGo if Bing API key is not available
            ddg_search = DuckDuckGoSearchRun()
            tools = [
                Tool(
                    name="DuckDuckGoSearch",
                    func=ddg_search.run,
                    description="Useful for when you need to answer questions about current events. You should ask targeted questions",
                )
            ]

        # Setup LLM and Agent
        llm = ChatOpenAI(model_name=self.openai_model, streaming=True)
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True
        )
        return agent
    
    @utils.enable_chat_history
    def main(self):
        bing_subscription_key = st.sidebar.text_input(
            label="Bing API Key",
            type="password",
            value=st.session_state['BING_SUBSCRIPTION_KEY'] if 'BING_SUBSCRIPTION_KEY' in st.session_state else '',
            placeholder="sk-..."
        )
        if bing_subscription_key:
            st.session_state['BING_SUBSCRIPTION_KEY'] = bing_subscription_key
            os.environ['BING_SUBSCRIPTION_KEY'] = bing_subscription_key
        else:
            st.warning("No Bing subscription key provided. Falling back to DuckDuckGo.")

        agent = self.setup_agent()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                try:
                    st_cb = StreamlitCallbackHandler(st.container())
                    response = agent.run(user_query, callbacks=[st_cb])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})
                    st.write(response)
                except Exception as e:
                    print(e)

if __name__ == "__main__":
    obj = ChatbotTools()
    obj.main()

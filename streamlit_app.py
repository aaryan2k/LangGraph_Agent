from agent import Agent
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

tool = TavilySearch(max_results=4, api_key=tavily_api_key)

prompt = """You are a smart research assistant. Use the search engine to look up information. \
    You are allowed to make multiple calls (either together or in sequence). \
    Only look up information when you are sure of what you want. \
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    """
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, api_key=api_key)
abot = Agent(model, [tool], system=prompt)

st.title("AI Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me a question"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    messages = [HumanMessage(content=prompt)]
    result = abot.graph.invoke({"messages": messages})
    response = result['messages'][-1].content
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
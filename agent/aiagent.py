from dotenv import load_dotenv
from langchain_openai import  ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from .retrival import rag_tool

load_dotenv()

class chatstate(TypedDict):
   messages:Annotated[list[BaseMessage],add_messages]


def chat_node(state:chatstate):
   msg = state["messages"]
   sys_prompt = SystemMessage(content='''
You are a helpful AI codebase assistance. 
You have a rag tool, if needed you can use that, to get the code from the codebase.

                              ''')
   if not any(m.type == "system" for m in msg):
        msg = [sys_prompt] + msg
   response= llm.invoke(msg)
   return {"messages":[response]}

# *****************tools*******************

tools =[rag_tool]
tool_node = ToolNode([rag_tool])


llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


graph = StateGraph(chatstate)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)
graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")

# Checkpointer
checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

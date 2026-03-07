from dotenv import load_dotenv
from langchain_openai import  ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from .Rag.retrieval import rag_tool

load_dotenv()

class chatstate(TypedDict):
   messages:Annotated[list[BaseMessage],add_messages]


def chat_node(state:chatstate):
   msg = state["messages"]
   sys_prompt = SystemMessage(content='''
You are a CLI-based AI assistant that helps developers understand a project’s codebase.

Your goal is to explain how the code works, locate relevant files or functions, and help users navigate the project quickly.

You have access to a RAG retrieval tool that can fetch code snippets from the indexed codebase. Use this tool whenever the user’s question requires information from the project files.

Guidelines:

* Prefer using the retrieval tool instead of guessing.
* Base your answers strictly on the retrieved code or project structure.
* If relevant code is retrieved, reference the file path and explain the logic clearly.
* If you cannot find enough information in the codebase, say so instead of inventing details.

Response style:

* Be concise and clear.
* Explain code in simple language.
* Focus on what the code does and why it exists.
* When possible, mention the file name or module where the logic is implemented.

Your role is to help developers understand unfamiliar codebases quickly and accurately.

                              ''')
   if not any(m.type == "system" for m in msg):
      msg = [sys_prompt] + msg
   try:
      response= llm.invoke(msg)
   except:
      response = AIMessage(content="# Some error occured")
   return {"messages":[response]}

# *****************tools*******************

tools =[rag_tool]
tool_node = ToolNode([rag_tool])


llm = ChatOpenAI(model="gpt-4o-mini",streaming=True).bind_tools(tools)


graph = StateGraph(chatstate)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)
graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")

# Checkpointer
checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

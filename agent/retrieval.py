from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings , ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()


@tool
def rag_tool (user_query):
   """
   Given user query it retrive releavent code from the codebase
   """
   embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

   vector_db = QdrantVectorStore.from_existing_collection(
      embedding=embedding_model,
      url = "http://localhost:6333",
      collection_name = "codebase rag1"
      )
   search_results = vector_db.similarity_search(
      query = user_query
   )
   context = "\n\n\n".join([
    f"Page Content: {result.page_content}\n"
    f"Page Number: {result.metadata.get('page_label', 'N/A')}\n"
    f"File Location: {result.metadata.get('source', 'Unknown')}"
    for result in search_results
   ])
   return context



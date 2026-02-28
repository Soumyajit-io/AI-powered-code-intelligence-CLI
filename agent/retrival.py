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



def rag_tool (user_query):
   """
   Given user query it retrive releavent context from the book ocaml chapter monads
   """
   embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

   vector_db = QdrantVectorStore.from_existing_collection(
      embedding=embedding_model,
      url = "http://localhost:6333",
      collection_name = "codebase rag"
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
history = [SystemMessage(content = '''
You are a helpful AI assistant who answers user query based on the available context
retrieved from the pdf file along with page number and page_content.

You should only ans the user based on the context provided and navigate the 
user to open the right page number to know more about the topic.

''')]

tools =[rag_tool]
tool_node = ToolNode(tools)



class chatstate(TypedDict):
   messages:Annotated[list[BaseMessage],add_messages]

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools) 

graph = StateGraph(chatstate)

def chat_node(state:chatstate):
   msg = state["messages"]
   sys_prompt = SystemMessage(content='''
You are a helpful AI assistant who answers user queries strictly
based on the available context retrieved from the PDF file.

You must:
- Answer ONLY from the provided context
- Include page number and page_content in your answer
- Guide the user to open the correct page number
- If answer is not in context, say: "The information is not available in the document.
''')
   if not any(m.type == "system" for m in msg):
        msg = [sys_prompt] + msg
   response= llm_with_tools.invoke(msg)
   return {"messages":[response]}

graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)




graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")


# Checkpointer
checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

while True:
    user_msg = input("Enter your question ")
    config = {'configurable':{'thread_id':12}}
    result = chatbot.invoke({"messages": [HumanMessage(content=(user_msg))]},config=config)
    print('AI: ',result["messages"][-1].content)



from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool

import os

load_dotenv()
@tool
def rag_tool (user_query):
   """
   Given user query it retrive releavent code from the codebase
   """
   embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

   vector_db = QdrantVectorStore.from_existing_collection(
      embedding=embedding_model,
      path=os.path.join(os.getcwd(), ".agent"),
      # url = "http://localhost:6333",
      collection_name = "codebase"
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



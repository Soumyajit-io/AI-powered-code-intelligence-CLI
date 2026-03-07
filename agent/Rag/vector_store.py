from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
import os
from langchain_openai import OpenAIEmbeddings
from rich.console import Console
console = Console()
load_dotenv()
def get_vectors(chunks):
   embedding_model = OpenAIEmbeddings(
      model="text-embedding-3-small"
      )
   vector_store = QdrantVectorStore.from_documents(
         documents=chunks,
         embedding=embedding_model,
         path=os.path.join(os.getcwd(), ".agent"),
         # url = "http://localhost:6333",
         collection_name = "codebase"
      )
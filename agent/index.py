from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
load_dotenv()

def get_vector(files:str):
   all_docs = []
   for file in files :
      loader = TextLoader(file,
                          encoding="utf-8",
                           autodetect_encoding=True)
      docs = loader.load()
      for doc in docs:
        doc.metadata["source"] = file

      all_docs.extend(docs)

   text_split = RecursiveCharacterTextSplitter(
      chunk_size = 400,
      chunk_overlap = 100 
      )
   chunks = text_split.split_documents(documents=all_docs)
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
   print("Vector Store created successfully")

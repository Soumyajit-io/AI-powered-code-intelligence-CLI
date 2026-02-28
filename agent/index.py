from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

loader = PyPDFLoader('Monads.pdf')
docs = loader.load()

text_split = RecursiveCharacterTextSplitter(
   chunk_size = 500,
   chunk_overlap = 100 
   )
chunks = text_split.split_documents(documents=docs)
embedding_model = OpenAIEmbeddings(
   model="text-embedding-3-small"
   )
vector_store = QdrantVectorStore.from_documents(
   documents=chunks,
   embedding=embedding_model,
   url = "http://localhost:6333",
   collection_name = "codebase rag"
)
print("Vector Store created successfully")

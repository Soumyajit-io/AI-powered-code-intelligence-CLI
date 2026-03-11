from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def text_based(file_path):
   with open(file_path,"r") as f:
     file= f.read()

   doc = Document(
    page_content=file,
    metadata={
        "file_path": file_path,
        "chunk_id": 0
    }
)
   text_split = RecursiveCharacterTextSplitter(
      chunk_size = 400,
      chunk_overlap = 100 
      )
   chunks = text_split.split_documents([doc])
   for i, d in enumerate(chunks):
      d.metadata["chunk_id"] = i
      print(f"metadata--> {d.metadata}")
      print(f"content--> {d.page_content +"\n\n"}")
   return chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

def splitter(docs:list):
   text_split = RecursiveCharacterTextSplitter(
      chunk_size = 400,
      chunk_overlap = 100 
      )
   chunks = text_split.split_documents(documents=docs)
   return chunks
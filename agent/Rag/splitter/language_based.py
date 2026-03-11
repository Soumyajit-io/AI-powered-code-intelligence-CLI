from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document

# languages
lang = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".c": Language.C,
    ".cpp": Language.CPP,
    ".cs": Language.CSHARP,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".php": Language.PHP,
    ".rb": Language.RUBY,
    ".scala": Language.SCALA,
    ".kt": Language.KOTLIN,
    ".swift": Language.SWIFT,

    ".html": Language.HTML,
    
    ".md": Language.MARKDOWN,

    ".tex": Language.LATEX,
    ".proto": Language.PROTO
}
# splitter
def lang_based(file_path,ext):
   
   with open(file_path,"r") as f:
     file= f.read()

   doc = Document(
    page_content=file,
    metadata={
        "file_path": file_path,
        "language": lang[ext].name,
        "chunk_id": 0
    }
)
   splitter = RecursiveCharacterTextSplitter.from_language(
    language=lang[ext],
    chunk_size=400,
    chunk_overlap=0
   )
   chunks = splitter.split_documents([doc])
   
   
   for i, d in enumerate(chunks):
      d.metadata["chunk_id"] = i
      print(f"metadata--> {d.metadata}")
      print(f"content--> {d.page_content +"\n\n"}")


if __name__ == "__main__":
  lang_based(r"agent\agent.py",".py")
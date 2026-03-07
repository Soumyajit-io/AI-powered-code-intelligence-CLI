from langchain_community.document_loaders import TextLoader


def loader(files:list[str]): # needs files
   all_docs = []
   for file in files :
      loader = TextLoader(file,
                          encoding="utf-8",
                           autodetect_encoding=True)
      docs = loader.load()
      for doc in docs:
        doc.metadata["source"] = file
        all_docs.append(doc)

   return all_docs
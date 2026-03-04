import typer
from .index import get_vector
from .aiagent import chatbot
from .Tools.scanner import scan_project
from langchain_core.messages import HumanMessage
import os


app = typer.Typer()

@app.command()
def init():
   """
   Convert your codebase into vectors
   """
   if not os.path.exists(".agent"):
      os.mkdir(".agent")
   paths = scan_project(os.getcwd())
   get_vector(paths)


@app.command()
def chat() :
   """
   chat with the agent 
   
   """
   print(os.getcwd())
   config = {'configurable':{'thread_id':12}}
   while True:
    user_msg = input("Ask anything: ")
    result = chatbot.invoke({"messages": [HumanMessage(content=(user_msg))]},config=config)
    print('AI: ',result["messages"][-1].content)


# @app.command()
# def ask(Question:str) :
#    pass

# @app.command()
# def reindex(Question:str) :
#    pass

# @app.command()
# def explain(Question:str) :
#    pass

if __name__ == "__main__":
  app()

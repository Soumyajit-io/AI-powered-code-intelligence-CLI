import typer
from .index import get_vector
from .aiagent import chatbot
from .Tools.scanner import scan_project
from .Tools.streaming import stream_chat
from langchain_core.messages import HumanMessage
import os
from rich.console import Console
from rich.markdown import Markdown

console = Console()

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
   while True:
        question = console.input("[bold green]Ask about codebase > [/bold green]")
        if question.lower() in {"exit", "quit"}:
            break
        stream_chat(question)
   


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

import typer
from ..tools.streaming import stream_chat
from rich.console import Console
from rich.markdown import Markdown
from ..ui.cli_ui import start_agent

console = Console()
app = typer.Typer()

@app.command()
def init():
   """
   Convert your codebase into vectors
   """
   start_agent()


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

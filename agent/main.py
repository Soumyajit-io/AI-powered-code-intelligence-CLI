import typer
app = typer.Typer()


@app.command()
def init():
   pass


@app.command()
def ask(Question:str) :
   pass


@app.command()
def chat(Question:str) :
   pass


@app.command()
def reindex(Question:str) :
   pass


@app.command()
def explain(Question:str) :
   pass


from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
import os
import time
from ..tools.scanner import scan_project
from ..rag.load_files import loader
from ..rag.splitter.text_based import splitter
from ..rag.vector_store import get_vectors
from rich.console import Console
console = Console()
def start_agent ():
   console.print(Panel("🚀 Initialising the Agent...", style="bold cyan"))
   if not os.path.exists(".agent"):
      os.mkdir(".agent")

   with Progress(
      SpinnerColumn(),
      TextColumn("[progress.description]{task.description}"),
      console=console,
   ) as progress:
      # scanner
      task = progress.add_task("Scanning project files...", total=None)
      scanned_files = scan_project(os.getcwd())
      time.sleep(1)
      # loader
      progress.update(task,description="Loading files...")
      loaded_files = loader(scanned_files)
      time.sleep(1)
      # splitter
      progress.update(task,description="Chunking files...")
      chunks = splitter(loaded_files)
      # vector genrator
      progress.update(task,description="Creating embeddings...")
      time.sleep(1)
      progress.update(task,description="Building vector store...")
      get_vectors(chunks)
      
   progress.stop()
      
   

   console.print("\n✅ Agent Initialised Successfully!", style="bold green")

if __name__=="__main__":
   start_agent()
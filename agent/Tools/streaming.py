import uuid
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage,AIMessageChunk
from agent.agent import chatbot

console = Console()

def stream_chat(ques:str):
   config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())
        }
    }
   state = {
      "messages":[HumanMessage(content=ques)]
   }
   answer = ""
   with Live("",console=console,refresh_per_second=10) as live:
         for message_chunk, metadata in chatbot.stream(
            state,
            config=config,
            stream_mode="messages"
        ):

            # stream only AI tokens
            if isinstance(message_chunk, AIMessageChunk):

                token = message_chunk.content

                if token:
                    answer += token
                    live.update(Markdown(answer))
                        
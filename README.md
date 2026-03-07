# AI Codebase CLI Assistant

A **CLI-based AI assistant** that lets developers **chat with their codebase directly from the terminal**.

The tool indexes your project, retrieves relevant code using **RAG (Retrieval Augmented Generation)**, and explains how the code works using an LLM.

It is designed to help developers **understand unfamiliar repositories quickly**, without manually searching through files.

---

## ✨ Features

* 💬 Chat with your **entire codebase**
* 🔎 **RAG-based code retrieval**
* ⚡ **Streaming responses in the terminal**
* 🧠 **LangGraph agent with tool calling**
* 📦 Vector search over project files
* 🖥️ Clean CLI interface

Example questions you can ask:

```
Where is authentication implemented?
Explain the login flow
What does the sidebar component do?
Which file handles database connections?
```

---

## 🖥 Example CLI

```
Ask about codebase > Where is authentication implemented?

File: src/auth/login.py
Function: authenticate_user

Explanation:
This function validates user credentials and creates a session token.
```

Responses are **streamed live in the terminal** for a smooth developer experience.

---

## 🏗 Architecture

The system follows a **Retrieval-Augmented Generation (RAG) pipeline powered by an agent**.

### Codebase Indexing

```
agent init
      │
      ▼
Scan Codebase
      │
      ▼
Load Files
      │
      ▼
Chunk Code
      │
      ▼
Generate Embeddings
      │
      ▼
Vector Store
```

### Query Pipeline

```
agent chat
      │
      ▼
LangGraph Agent
      │
      ▼
Retriever Tool
      │
      ▼
Relevant Code Context
      │
      ▼
LLM Response
      │
      ▼
Streaming CLI Output
```


---

## 🧰 Tech Stack

* **LangGraph** – Agent orchestration
* **LangChain** – LLM tooling framework
* **OpenAI Embeddings** – semantic code search
* **Qdrant** – vector database
* **Rich** – streaming CLI UI
* **Typer** – command-line interface
* **Python** – core implementation

---

## ⚙ Current Capabilities

* Codebase question answering
* Semantic code retrieval
* Tool-enabled AI agent
* Streaming CLI responses

---

## 🛠 Planned Improvements

Future upgrades will include:

* Tree-sitter based code chunking
* Symbol-level indexing (functions / classes)
* Project architecture analysis
* Dependency graph generation
* Git history awareness

---


## 📜 License

MIT License

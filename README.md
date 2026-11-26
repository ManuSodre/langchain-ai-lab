# langchain-ai-lab
Development environment for LLM applications, RAG pipelines and AI agents using LangChain and Python.

## RAG starter

This repository includes a small retrieval-augmented generation example. It reads
`.txt` and `.md` files from `data/raw/`, indexes them with FAISS, and answers a
question using an OpenAI chat model.

### Setup

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
4. Add source documents to `data/raw/`.

Run the example with `python main.py`. Run the tests with `pytest`.

The generated FAISS index is stored in `data/processed/` and is excluded from
version control.

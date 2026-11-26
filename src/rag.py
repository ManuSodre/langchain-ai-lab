from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    RAW_DATA_DIR,
    get_openai_api_key,
)


def load_documents(directory: Path = RAW_DATA_DIR) -> list[Document]:
    """Load plain-text and Markdown documents from a directory tree."""
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": str(path)},
        )
        for path in paths
    ]


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def build_vector_store(directory: Path = RAW_DATA_DIR) -> FAISS:
    """Create an in-memory FAISS index from documents in the given directory."""
    documents = load_documents(directory)
    if not documents:
        raise FileNotFoundError(
            f"No .txt or .md documents found in {directory}. Add source files first."
        )

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=get_openai_api_key(),
    )
    return FAISS.from_documents(split_documents(documents), embeddings)


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents with their source paths."""
    return "\n\n".join(
        f"Source: {document.metadata.get('source', 'unknown')}\n"
        f"{document.page_content}"
        for document in documents
    )


def answer_question(
    question: str,
    vector_store: VectorStore,
    model: BaseChatModel,
    k: int = 4,
) -> str:
    """Answer a question using the most relevant indexed documents."""
    documents = vector_store.similarity_search(question, k=k)
    if not documents:
        return "I couldn't find relevant information in the indexed documents."

    messages = [
        SystemMessage(
            content=(
                "Answer the question using only the supplied context. "
                "If the context does not contain the answer, say so. "
                "Cite the source paths when possible.\n\n"
                f"Context:\n{format_context(documents)}"
            )
        ),
        HumanMessage(content=question),
    ]
    response = model.invoke(messages)
    return str(response.content)

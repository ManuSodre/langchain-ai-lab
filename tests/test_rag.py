from langchain_core.documents import Document

from src.rag import answer_question, format_context


class FakeVectorStore:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.requested_k: int | None = None

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        self.requested_k = k
        return self.documents[:k]


class FakeResponse:
    content = "The answer is in the source."


class FakeChatModel:
    def invoke(self, messages: list[object]) -> FakeResponse:
        return FakeResponse()


def test_format_context_includes_document_source() -> None:
    context = format_context(
        [Document(page_content="RAG basics", metadata={"source": "guide.md"})]
    )

    assert "Source: guide.md" in context
    assert "RAG basics" in context


def test_answer_question_retrieves_documents_and_returns_model_response() -> None:
    store = FakeVectorStore(
        [Document(page_content="Relevant context", metadata={"source": "notes.txt"})]
    )

    answer = answer_question("What is RAG?", store, FakeChatModel(), k=2)

    assert answer == "The answer is in the source."
    assert store.requested_k == 2


def test_answer_question_handles_no_relevant_documents() -> None:
    store = FakeVectorStore([])

    answer = answer_question("Unknown topic", store, FakeChatModel())

    assert answer == "I couldn't find relevant information in the indexed documents."

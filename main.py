from src.config import PROCESSED_DATA_DIR
from src.llm import get_chat_model
from src.rag import answer_question, build_vector_store


def main() -> None:
    question = input("Ask a question about your documents: ").strip()
    if not question:
        raise SystemExit("Please enter a question.")

    vector_store = build_vector_store()
    answer = answer_question(question, vector_store, get_chat_model())
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(PROCESSED_DATA_DIR / "faiss_index"))
    print(f"\n{answer}")


if __name__ == "__main__":
    main()

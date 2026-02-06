from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


RESUME = "resume"


def get_vector_store(collection_name: str) -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )
    return vector_store

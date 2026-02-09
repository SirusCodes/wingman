import os
from langchain_chroma import Chroma
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings


def get_vector_store(collection_name: str) -> Chroma:
    embeddings = CloudflareWorkersAIEmbeddings(
        model_name="@cf/qwen/qwen3-embedding-0.6b",
        api_token=os.getenv("CF_AI_API_TOKEN"),
        account_id=os.getenv("CF_ACCOUNT_ID"),
    )
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
    )
    return vector_store

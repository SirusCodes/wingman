from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ai.store import get_vector_store


def store_pdf_doc(file_path: str, collection_name: str, data_id: str) -> list[str]:
    loader = PyPDFLoader(file_path)
    doc = loader.load()

    return store_docs(
        list(map(lambda d: d.metadata.update({"data_id": data_id}) or d, doc)),
        collection_name,
    )


def store_docs(docs: list[Document], collection_name: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    store = get_vector_store(collection_name=collection_name)

    doc_ids = store.add_documents(all_splits)
    return doc_ids

from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    HTMLSemanticPreservingSplitter,
)
from bs4 import BeautifulSoup, Tag
import requests

from ai.store import get_vector_store


def clear_documents_with_data_id(data_id: str, user_id: str) -> None:
    """Clear documents with the given data_id from the vector store."""
    store = get_vector_store(collection_name=user_id)
    store.delete(where={"metadata.data_id": data_id})


def store_portfolio(url: str, data_id: str, user_id: str) -> list[str]:
    """Store the portfolio data in the vector store."""
    site = crawl_website(url, max_depth=0)

    text_splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[
            ("h1", "Header 1"),
            ("h2", "Header 2"),
        ],
        elements_to_preserve=["table", "ul", "ol", "code"],
        denylist_tags=["script", "style", "head"],
        external_metadata={"data_id": data_id, "source": url},
        preserve_links=True,
        chunk_overlap=200,
        preserve_parent_metadata=True,
        normalize_text=True,
    )

    store = get_vector_store(collection_name=user_id)
    return store.add_documents(text_splitter.split_text(site))


def store_blog(url: str, data_id: str, user_id: str) -> list[str]:
    """Recurse upto 3 levels with the same domain and store the data in the vector store."""
    sites = crawl_website(url, max_depth=2)

    def code_handler(element: Tag) -> str:
        data_lang = element.get("data-lang")
        code_format = f"<code:{data_lang}>{element.get_text()}</code>"

        return code_format

    text_splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[
            ("h1", "Header 1"),
            ("h2", "Header 2"),
        ],
        elements_to_preserve=["table", "ul", "ol", "code"],
        denylist_tags=["script", "style", "head"],
        custom_handlers={"code": code_handler},
        external_metadata={"data_id": data_id, "source": url},
        preserve_links=True,
        chunk_overlap=200,
        preserve_parent_metadata=True,
        normalize_text=True,
    )

    store = get_vector_store(collection_name=user_id)
    doc_ids = []
    for site in sites:
        doc_ids.extend(store.add_documents(text_splitter.split_text(site)))

    return doc_ids


def store_pdf_doc(file_path: str, data_id: str, user_id: str) -> list[str]:
    loader = PyPDFLoader(file_path, extraction_mode="layout")
    docs = loader.load()

    for doc in docs:
        doc.metadata.update({"data_id": data_id, "source": "resume"})

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    store = get_vector_store(collection_name=user_id)

    doc_ids = store.add_documents(all_splits)
    return doc_ids


def crawl_website(
    url: str,
    max_depth: int = 2,
    visited: set = None,
    base_domain: str = None,
    current_depth: int = 0,
) -> list[str]:
    """Custom recursive crawler that stays within the same domain."""
    if visited is None:
        visited = set()

    if base_domain is None:
        base_domain = urlparse(url).netloc

    # Remove hash fragment from URL
    parsed = urlparse(url)
    url_without_fragment = parsed._replace(fragment="").geturl()

    # Stop if max depth reached or URL already visited
    if current_depth > max_depth or url_without_fragment in visited:
        return []

    visited.add(url_without_fragment)
    docs = []

    try:
        print(
            f"{'  ' * current_depth}Crawling: {url_without_fragment} (depth {current_depth})"
        )
        response = requests.get(url_without_fragment, timeout=10)
        response.raise_for_status()

        # Only process HTML documents
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            print(f"{'  ' * current_depth}Skipping non-HTML document: {content_type}")
            return []

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract text content
        text_content = soup.get_text(separator="\n", strip=True)

        # Append text content
        docs.append(text_content)

        # Extract all links if not at max depth
        if current_depth < max_depth:
            links = soup.find_all("a", href=True)
            for link in links:
                href = link["href"]
                # Convert relative URLs to absolute
                absolute_url = urljoin(url_without_fragment, href)

                # Remove fragment from the absolute URL
                parsed_link = urlparse(absolute_url)
                absolute_url_no_fragment = parsed_link._replace(fragment="").geturl()

                # Check if link is within the same domain
                link_domain = parsed_link.netloc
                if (
                    link_domain == base_domain
                    and absolute_url_no_fragment not in visited
                ):
                    # Recursively crawl
                    child_docs = crawl_website(
                        absolute_url_no_fragment,
                        max_depth,
                        visited,
                        base_domain,
                        current_depth + 1,
                    )
                    docs.extend(child_docs)

    except Exception as e:
        print(f"{'  ' * current_depth}Error crawling {url_without_fragment}: {e}")

    return docs

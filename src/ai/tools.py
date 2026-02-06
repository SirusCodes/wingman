from typing import TypedDict
from langchain.tools import tool, ToolRuntime
from pydantic import BaseModel, Field

from ai.store import get_vector_store


class Context(TypedDict):
    admin_name: str
    user_id: str


class GetDataTool(BaseModel):
    """Get data from the admin's data sources (resume, blog, or portfolio)."""

    query: str = Field(description="The query to search for in the data sources.")


@tool(args_schema=GetDataTool, response_format="content_and_artifact")
def get_data(query: str, runtime: ToolRuntime[Context]):
    runtime.stream_writer("Getting data.")

    user_id = runtime.context["user_id"]
    store = get_vector_store(collection_name=user_id)
    results = store.similarity_search(query, k=5)
    content = "\n\n".join([doc.page_content for doc in results])
    return content, results


def get_all():
    return [get_data]

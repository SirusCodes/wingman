from langchain.tools import tool
from pydantic import BaseModel, Field

from ai.store import RESUME, get_vector_store


class ResumeDataTool(BaseModel):
    """Get data from the admin's resume."""

    query: str = Field(description="The query to get data from the resume.")


@tool(args_schema=ResumeDataTool, response_format="content_and_artifact")
def get_data_from_resume(query: str):
    store = get_vector_store(collection_name=RESUME)
    results = store.similarity_search(query, k=5)
    content = "\n\n".join([doc.page_content for doc in results])
    return content, results


def get_all():
    return [get_data_from_resume]

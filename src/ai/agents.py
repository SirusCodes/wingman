import json
from typing import Any
from langchain_cloudflare import ChatCloudflareWorkersAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
    ModelRequest,
    dynamic_prompt,
    AgentMiddleware,
    hook_config,
    SummarizationMiddleware,
)
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.vectorstores import VectorStore


from ai import tools
from ai.store import get_vector_store


@dynamic_prompt
def system_prompt(request: ModelRequest) -> str:
    admin_name = request.runtime.context.get("admin_name", "Admin")

    return f"""You are a wingman of {admin_name}. Other users will ask you questions about {admin_name}.
    You can use the tools to answer the questions. Always use the tools when you are not sure about the answer.

    If you don't find any relevant information from the tools, you can try to find close enough information and use it to answer the question.
    eg. If someone asks about {admin_name}'s work in backend dev but you only find information about frontend dev, you can emphasize the frontend dev experience and try to relate it to backend dev.
    """


class ChatCache(AgentMiddleware):
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return

        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            return

        query = last_message.content
        # Check if the same query has been asked before
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=1)
        if not results or results[0][1] < 0.6:  # Use cache only if similarity is high
            print("Cache miss.", results)
            return

        print(f"Cache hit! Similarity score: {results[0][1]}")
        cached_response_json = results[0][0].metadata.get("response")
        if not cached_response_json:
            return

        # Deserialize the cached response and reconstruct the AIMessage
        try:
            response_dict = json.loads(cached_response_json)
            cached_message = AIMessage(**response_dict)
            cached_message.metadata = {"cached": True}
        except Exception as e:
            print(f"Failed to deserialize cached response: {e}")
            return

        return {
            "messages": [cached_message],
            "jump_to": "end",
        }

    def after_model(self, state, runtime) -> None:
        messages = state.get("messages", [])
        if not messages:
            return

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return

        response = last_message

        if hasattr(response, "metadata") and response.metadata.get("cached"):
            print("Response is from cache, not caching again.")
            return

        if response.tool_calls or response.invalid_tool_calls:
            print("Skipping cache for tool-call-only AI message.")
            return

        human_messages = [msg.text for msg in messages if isinstance(msg, HumanMessage)]

        if not human_messages:
            return

        doc = Document(
            page_content="\n\n".join(human_messages),
            metadata={"response": response.model_dump_json()},
        )
        self.vector_store.add_documents([doc])


def get_agent():

    chat_model = ChatCloudflareWorkersAI(
        model_name="@cf/qwen/qwen3-30b-a3b-fp8", temperature=0.7
    )

    agent = create_agent(
        model=chat_model,
        tools=tools.get_all(),
        context_schema=tools.Context,
        middleware=[
            system_prompt,
            ToolCallLimitMiddleware(thread_limit=20, run_limit=3),
            ChatCache(get_vector_store(collection_name="chat_cache")),
            SummarizationMiddleware(
                model=init_chat_model("google_genai:gemma-3-27b-it", temperature=0.7),
                trigger=("tokens", 4000),
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    return agent

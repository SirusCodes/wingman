from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.graph.state import CompiledStateGraph

from ai import tools


def get_agent(admin_name: str) -> CompiledStateGraph:
    chat_model = init_chat_model("google_genai:gemini-3-flash-preview")
    prompt = f"""You are a wingman of {admin_name}. Other users will ask you questions about {admin_name}.
    You can use the tools to answer the questions. Always use the tools when you are not sure about the answer.

    If you don't find any relevant information from the tools, you can try to find close enough information and use it to answer the question.
    eg. If someone asks about {admin_name}'s work in backend dev but you only find information about frontend dev, you can emphasize the frontend dev experience and try to relate it to backend dev.
    """

    agent = create_agent(chat_model, tools.get_all(), system_prompt=prompt)
    return agent

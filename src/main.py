import json
import os
import uuid
from typing import AsyncGenerator, Literal

import dotenv
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ai.agents import get_agent
from ai.doc_manager import store_portfolio, store_blog, store_pdf_doc

dotenv.load_dotenv()

user_id = "default_user"

# Ensure required environment variables are set
required_env_vars = [
    "GOOGLE_API_KEY",
    "CF_ACCOUNT_ID",
    "CF_AI_API_TOKEN",
    "ADMIN_NAME",
    "ADMIN_PASSWORD",
    "CHROMA_API_KEY",
    "CHROMA_TENANT",
    "CHROMA_DATABASE",
]
for var in required_env_vars:
    if var not in os.environ:
        raise RuntimeError(
            f"Environment variable {var} is not set. Please set it before running the server."
        )


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for the chat endpoint"""

    thread_id: str = Field(
        ..., description="Unique identifier for the conversation thread"
    )
    prompt: str = Field(..., description="The user's message/prompt")


class TokenCount(BaseModel):
    """Token usage information"""

    input_tokens: int | None = Field(None, description="Number of input tokens")
    output_tokens: int | None = Field(None, description="Number of output tokens")
    total_tokens: int | None = Field(None, description="Total number of tokens")


MessageType = Literal[
    "HumanMessage",
    "AIMessage",
    "ToolMessage",
    "SystemMessage",
    "complete",
    "error",
]

ActionType = Literal[
    "received_prompt",
    "calling_tool",
    "generating",
    "processing",
    "complete",
    "error",
]


class ChatEvent(BaseModel):
    """Individual event in the streaming response"""

    type: MessageType = Field(
        ...,
        description="Type of event (e.g., 'HumanMessage', 'AIMessage', 'tool_call')",
    )
    content: str = Field(..., description="Event content")
    tokens: TokenCount | None = Field(None, description="Token usage for this event")
    is_cached: bool = Field(False, description="Whether this response came from cache")
    action: ActionType | None = Field(
        None,
        description="What the LLM is doing (e.g., 'thinking', 'calling_tool', 'generating')",
    )
    tool_name: str | None = Field(
        None, description="Name of the tool being called (if action is 'calling_tool')"
    )


def get_message_type(message: AnyMessage) -> MessageType:
    """Map message instance to a MessageType literal."""
    if isinstance(message, HumanMessage):
        return "HumanMessage"
    if isinstance(message, AIMessage):
        return "AIMessage"
    if isinstance(message, ToolMessage):
        return "ToolMessage"
    if isinstance(message, SystemMessage):
        return "SystemMessage"
    return "error"


class ChatStreamResponse(BaseModel):
    """Streaming response model for the chat endpoint"""

    thread_id: str = Field(..., description="Thread ID for the conversation")
    status: str = Field(default="streaming", description="Status of the response")


# Initialize FastAPI app
app = FastAPI(
    title="ChatAbout API",
    description="Stream-based chat API with agent support",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
_agent = None


def get_chat_responses():
    """Generate OpenAPI responses for chat endpoint"""
    return {
        200: {
            "description": "Streaming response with Server-Sent Events",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "thread_id": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": [
                                    "HumanMessage",
                                    "AIMessage",
                                    "ToolMessage",
                                    "SystemMessage",
                                    "complete",
                                    "error",
                                ],
                            },
                            "content": {"type": "string"},
                            "tokens": {
                                "type": "object",
                                "properties": {
                                    "input_tokens": {"type": ["integer", "null"]},
                                    "output_tokens": {"type": ["integer", "null"]},
                                    "total_tokens": {"type": ["integer", "null"]},
                                },
                            },
                            "is_cached": {"type": "boolean"},
                            "action": {
                                "type": ["string", "null"],
                                "enum": [
                                    "received_prompt",
                                    "calling_tool",
                                    "generating",
                                    "processing",
                                    "complete",
                                    "error",
                                    None,
                                ],
                            },
                            "tool_name": {"type": ["string", "null"]},
                        },
                    }
                }
            },
        },
        422: {"description": "Validation error"},
    }


def get_agent_instance():
    """Get or initialize the agent"""
    global _agent
    if _agent is None:
        _agent = get_agent()
    return _agent


async def stream_chat_events(
    thread_id: str,
    prompt: str,
) -> AsyncGenerator[str, None]:
    """
    Stream events from the agent.

    Yields SSE formatted strings with JSON data containing:
    - Message type and content
    - Token counts (input/output)
    - Cache status
    - LLM action information
    """
    agent = get_agent_instance()

    try:
        # Stream events from the agent
        admin_name = os.environ["ADMIN_NAME"]
        async for event in agent.astream(
            {"messages": [HumanMessage(content=prompt)]},
            stream_mode="values",
            context={
                "admin_name": admin_name,
                "user_id": user_id,
            },
            config={"configurable": {"thread_id": thread_id}},
        ):
            # Extract the last message from the event
            if "messages" in event and event["messages"]:
                message: AnyMessage = event["messages"][-1]

                # Determine action based on message type and content
                message_type = get_message_type(message)
                action: ActionType | None = None
                tool_name: str | None = None
                if isinstance(message, AIMessage):
                    if message.tool_calls:
                        action = "calling_tool"
                        # Extract tool names from tool calls
                        tool_names = [
                            tc.get("name", tc.get("tool", "unknown"))
                            for tc in message.tool_calls
                        ]
                        tool_name = ", ".join(tool_names) if tool_names else None
                    else:
                        action = "generating"
                elif isinstance(message, HumanMessage):
                    action = "received_prompt"
                else:
                    action = "processing"

                # Extract token usage information
                tokens = None
                if hasattr(message, "usage_metadata") and message.usage_metadata:
                    tokens = TokenCount(
                        input_tokens=message.usage_metadata.get("input_tokens"),
                        output_tokens=message.usage_metadata.get("output_tokens"),
                        total_tokens=(
                            (
                                message.usage_metadata.get("input_tokens", 0)
                                + message.usage_metadata.get("output_tokens", 0)
                            )
                            if message.usage_metadata.get("input_tokens")
                            and message.usage_metadata.get("output_tokens")
                            else None
                        ),
                    )

                # Check if response is from cache
                is_cached = False
                if hasattr(message, "metadata") and isinstance(message.metadata, dict):
                    is_cached = message.metadata.get("cached", False)

                # Prepare the event data
                event_data = {
                    "thread_id": thread_id,
                    "type": message_type,
                    "content": message.content.strip(),
                    "tokens": tokens.model_dump(exclude_none=True) if tokens else None,
                    "is_cached": is_cached,
                    "action": action,
                    "tool_name": tool_name,
                }

                # Yield as SSE formatted data
                yield f"data: {json.dumps(event_data)}\n\n"

        # Send completion event
        yield f"data: {json.dumps({'thread_id': thread_id, 'type': 'complete', 'content': '', 'action': 'complete'})}\n\n"

    except Exception as e:
        # Send error event
        error_data = {
            "thread_id": thread_id,
            "type": "error",
            "content": str(e),
            "action": "error",
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


class UploadResponse(BaseModel):
    """Response model for the upload endpoint"""

    success: bool = Field(..., description="Whether the upload was successful")
    message: str = Field(..., description="Status message")
    data_ids: dict[str, list[str]] = Field(
        default_factory=dict, description="Document IDs for each uploaded data type"
    )


@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Upload"],
    summary="Upload resume, portfolio, and blog",
    description="Upload resume, portfolio, and blog links with password authentication",
)
async def upload_data(
    password: str = Form(..., description="Admin password for authentication"),
    resume_url: str | None = Form(None, description="Resume PDF URL"),
    portfolio_url: str | None = Form(None, description="Portfolio website URL"),
    blog_url: str | None = Form(None, description="Blog website URL"),
):
    """
    Upload resume, portfolio, and blog data with password verification.

    Args:
        password: Admin password (verified against ADMIN_PASSWORD env var)
        resume_url: Optional URL to resume PDF
        portfolio_url: Optional portfolio website URL
        blog_url: Optional blog website URL

    Returns:
        UploadResponse with document IDs for each uploaded data type
    """
    # Verify password
    admin_password = os.environ.get("ADMIN_PASSWORD")
    if password != admin_password:
        raise HTTPException(status_code=401, detail="Invalid password")

    # Validate that at least one data source is provided
    if not resume_url and not portfolio_url and not blog_url:
        raise HTTPException(
            status_code=422,
            detail="At least one of resume_url, portfolio_url, or blog_url must be provided",
        )

    doc_ids = {}

    # Process resume
    if resume_url:
        data_id = f"resume_{user_id}_{uuid.uuid4().hex[:8]}"
        resume_doc_ids = store_pdf_doc(resume_url, data_id, user_id)
        doc_ids["resume"] = resume_doc_ids

    # Process portfolio
    if portfolio_url:
        data_id = f"portfolio_{user_id}_{uuid.uuid4().hex[:8]}"
        portfolio_doc_ids = store_portfolio(portfolio_url, data_id, user_id)
        doc_ids["portfolio"] = portfolio_doc_ids

    # Process blog
    if blog_url:
        data_id = f"blog_{user_id}_{uuid.uuid4().hex[:8]}"
        blog_doc_ids = store_blog(blog_url, data_id, user_id)
        doc_ids["blog"] = blog_doc_ids

    return UploadResponse(
        success=True,
        message="Data uploaded successfully",
        data_ids=doc_ids,
    )


@app.post(
    "/chat",
    response_class=StreamingResponse,
    tags=["Chat"],
    summary="Stream chat responses",
    description="Stream chat responses using the AI agent with support for conversation threads",
    responses=get_chat_responses(),
)
async def chat(request: ChatRequest):
    """
    Stream chat responses from the AI agent.

    This endpoint accepts a prompt and thread_id, then streams the agent's response
    as Server-Sent Events (SSE).

    Args:
        request: ChatRequest containing thread_id and prompt

    Returns:
        StreamingResponse with SSE formatted data
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt cannot be empty")

    if not request.thread_id.strip():
        raise HTTPException(status_code=422, detail="thread_id cannot be empty")

    return StreamingResponse(
        stream_chat_events(
            thread_id=request.thread_id,
            prompt=request.prompt,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )

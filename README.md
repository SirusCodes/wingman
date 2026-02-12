# Wingman

An AI-powered chatbot that acts as your wingman, answering questions about using your documents, portfolio, and blog content. Built with LangChain, FastAPI, and powered by advanced language models.

## 🎯 What It Does

Wingman is a conversational AI application that:

- **Answers Questions Intelligently** - Responds to user queries about your background, experience, and portfolio
- **Retrieves Relevant Information** - Uses vector search (RAG) to find and cite relevant documents
- **Maintains Conversation Context** - Keeps track of multi-turn conversations with thread IDs
- **Streams Responses** - Provides real-time streaming responses for better UX
- **Manages Multiple Document Types** - Supports portfolios, blogs, and PDF documents

## ✨ Key Features

- **Vector-Based Retrieval**: Uses Chroma vector database for semantic document search
- **Persistent Conversations**: Maintains conversation threads with full history using Postgres
- **Tool-Integrated Agents**: LangChain agents with tool calling capabilities
- **Streaming Responses**: Real-time chat streaming for responsive UI

## 📋 Prerequisites

- Python 3.13 or higher
- Postgres database (for conversation persistence)
- API keys for:
  - Google Generative AI (`GOOGLE_API_KEY`)
  - Cloudflare Workers AI (`CF_ACCOUNT_ID`, `CF_AI_API_TOKEN`)
  - Chroma Vector Database (`CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE`)
- Admin credentials for authentication (`ADMIN_NAME`, `ADMIN_PASSWORD`)

## 🚀 Setup

### 1. Clone and Install

```bash
git clone <repository-url>
cd wingman
uv sync
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory with from .env.example and fill in your API keys and credentials.

### 3. Run the Server

#### Local Development

```bash
uv run main
```

#### Docker

```bash
docker build -t wingman .
docker run --env-file .env -p 8000:8000 wingman
```

The API will start on `http://localhost:8000`

## 📚 Project Structure

```
wingman/
├── main.py                  # FastAPI application and endpoints
├── src/
│   ├── __init__.py
│   └── ai/
│       ├── __init__.py
│       ├── agents.py        # LangChain agent configuration
│       ├── tools.py         # Tool definitions for agents
│       ├── doc_manager.py   # Document storage and management
│       └── store.py         # Vector store initialization
├── Dockerfile               # Docker container configuration
├── .dockerignore             # Docker build ignore file
├── .env.example             # Example environment variables
├── pyproject.toml           # Project dependencies and metadata
├── requirements.txt         # Pip dependencies (for Docker)
├── uv.lock                  # Locked dependencies (uv)
├── README.md                # This file
└── LICENCE                  # Project license
```

## 🔌 API Endpoints

### Chat Endpoint

**POST** `/chat`

Send a message and get a streaming response.

**Request:**

```json
{
  "thread_id": "unique-thread-id",
  "prompt": "Tell me about the admin's experience"
}
```

**Response:** Streaming text responses

### Document Management

- **POST** `/upload` - Upload resume (PDF URL), portfolio URL, and blog URL

## 🛠️ Technologies Used

- **Framework**: FastAPI
- **AI/ML**: LangChain, LanGraph
- **LLMs**: Google Generative AI, Cloudflare Workers AI
- **Vector Database**: Chroma
- **Document Processing**: BeautifulSoup4, PyPDF
- **Web Framework**: Uvicorn (FastAPI default)

## 📝 Usage Example

```python
import requests

# Start a new conversation
thread_id = "user-123-conversation-1"

# Send a message
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "thread_id": thread_id,
        "prompt": "What are their main skills?"
    },
    stream=True
)

# Stream the response
for chunk in response.iter_content(decode_unicode=True):
    print(chunk, end="", flush=True)
```

## 🔐 Authentication

The application requires admin authentication for sensitive operations. Configure admin credentials via environment variables.

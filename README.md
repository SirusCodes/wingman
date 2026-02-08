# ChatAbout

An AI-powered chatbot that acts as your wingman, answering questions about using your documents, portfolio, and blog content. Built with LangChain, FastAPI, and powered by advanced language models.

## 🎯 What It Does

ChatAbout is a conversational AI application that:

- **Answers Questions Intelligently** - Responds to user queries about your background, experience, and portfolio
- **Retrieves Relevant Information** - Uses vector search (RAG) to find and cite relevant documents
- **Maintains Conversation Context** - Keeps track of multi-turn conversations with thread IDs
- **Streams Responses** - Provides real-time streaming responses for better UX
- **Manages Multiple Document Types** - Supports portfolios, blogs, and PDF documents

## ✨ Key Features

- **Vector-Based Retrieval**: Uses Chroma vector database for semantic document search
- **Persistent Conversations**: Maintains conversation threads with full history
- **Tool-Integrated Agents**: LangChain agents with tool calling capabilities
- **Streaming Responses**: Real-time chat streaming for responsive UI

## 📋 Prerequisites

- Python 3.13 or higher
- API keys for:
  - Google Generative AI (`GOOGLE_API_KEY`)
  - Cloudflare Workers AI (`CF_ACCOUNT_ID`, `CF_AI_API_TOKEN`)
  - Chroma Vector Database (`CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE`)

## 🚀 Setup

### 1. Clone and Install

```bash
git clone <repository-url>
cd chatabout
uv sync
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory with:

```env
# Google AI
GOOGLE_API_KEY=your_google_api_key_here

# Cloudflare Workers AI
CF_ACCOUNT_ID=your_cloudflare_account_id
CF_AI_API_TOKEN=your_cloudflare_token

# Chroma Vector Database
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=your_chroma_tenant
CHROMA_DATABASE=your_chroma_database

# Admin Credentials
ADMIN_NAME=Your Name
ADMIN_PASSWORD=your_secure_password
```

### 3. Run the Server

```bash
uv run src/main.py
```

The API will start on `http://localhost:8000`

## 📚 Project Structure

```
chatabout/
├── src/
│   ├── main.py              # FastAPI application and endpoints
│   ├── ai/
│   │   ├── agents.py        # LangChain agent configuration
│   │   ├── tools.py         # Tool definitions for agents
│   │   ├── doc_manager.py   # Document storage and management
│   │   └── store.py         # Vector store initialization
│   └── __init__.py
├── pyproject.toml           # Project dependencies and metadata
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
  "message": "Tell me about the admin's experience"
}
```

**Response:** Streaming text responses

### Document Management

- **POST** `/store-portfolio` - Upload portfolio information
- **POST** `/store-blog` - Upload blog content
- **POST** `/store-pdf` - Upload PDF documents

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
        "message": "What are their main skills?"
    },
    stream=True
)

# Stream the response
for chunk in response.iter_content(decode_unicode=True):
    print(chunk, end="", flush=True)
```

## 🔐 Authentication

The application requires admin authentication for sensitive operations. Configure admin credentials via environment variables.

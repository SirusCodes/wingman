# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/health")
# def health_check():
#     return {"status": "ok"}

from ai.agents import get_agent
from ai.doc_manager import store_pdf_doc
from ai.store import RESUME

import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


def main():
    collection_name = RESUME
    doc_ids = store_pdf_doc(
        "./resume-darshan-rander.pdf", collection_name, data_id="resume"
    )
    print(doc_ids)

    agent = get_agent(admin_name="Darshan Rander")

    query = "What are Darshan's skills in backend development?"
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()

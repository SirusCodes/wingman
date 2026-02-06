from ai.agents import get_agent
from ai.doc_manager import store_pdf_doc, store_website

import getpass
import os
import dotenv

dotenv.load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


def main():
    user_id = "test_user"

    doc_ids = store_pdf_doc(
        "./resume-darshan-rander.pdf", user_id=user_id, data_id="resume"
    )
    doc_ids = store_website(
        "https://darshanrander.com/", user_id=user_id, data_id="blog"
    )
    print(doc_ids)

    agent = get_agent()

    query = "Why should I hire him?"
    for events in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
        context={
            "admin_name": "Darshan Rander",
            "user_id": user_id,
        },
        config={"configurable": {"thread_id": "test_thread"}},
    ):
        events["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()

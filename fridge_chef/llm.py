"""Creates the Gemini chat model."""

import os

from dotenv import load_dotenv


def get_llm():
    load_dotenv()
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0.7,
    )

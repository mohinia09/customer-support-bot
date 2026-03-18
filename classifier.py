from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


Category = Literal["Customer Support", "Other"]


def _get_llm() -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in your environment or .env file.")

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )


def _build_classification_chain():
    """
    LangChain text classification chain:
    prompt -> LLM -> string output parser -> label normalization
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful text classifier. "
                "Classify the email into exactly one of the allowed categories and output ONLY the category label.",
            ),
            (
                "user",
                """Classify the following email as one of:

Customer Support:
- The sender is asking for help, reporting a bug, requesting a refund, facing an issue, asking how to do something,
  asking about billing, account access, password resets, errors, outages, delivery problems, cancellations, or any
  request that requires assistance from support.

Other:
- Anything that is not a support request (e.g., marketing, newsletters, spam, job inquiries, partnerships, internal
  chatter, FYI messages, social outreach, cold emails, announcements).

Email body:
{email_body}

Return ONLY one of these exact labels:
Customer Support
Other""",
            ),
        ]
    )

    llm = _get_llm()
    chain = prompt | llm | StrOutputParser()
    return chain


_CHAIN = None


def classify_email(email_body: str) -> Category:
    """
    Classify an email body as 'Customer Support' or 'Other'.
    """
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = _build_classification_chain()

    text = (email_body or "").strip()
    if not text:
        return "Other"

    raw = _CHAIN.invoke({"email_body": text}).strip()
    label = raw.replace('"', "").replace("'", "").strip()

    if label.lower() == "customer support":
        return "Customer Support"
    return "Other"


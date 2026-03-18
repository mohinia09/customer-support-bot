from __future__ import annotations

import os
from typing import Any, List, Union

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from vector_store import RetrievedChunk, get_retriever


def _get_llm() -> ChatOpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in your environment or .env file.")

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        api_key=api_key,
    )


def _format_hits(hits: List[RetrievedChunk]) -> str:
    if not hits:
        return "No relevant docs found."

    parts: List[str] = []
    for i, h in enumerate(hits, start=1):
        src = h.metadata.get("source", "unknown")
        idx = h.metadata.get("chunk_index", "n/a")
        dist = h.distance
        dist_s = f"{dist:.4f}" if isinstance(dist, (int, float)) else "n/a"
        parts.append(f"[{i}] source={src} chunk_index={idx} distance={dist_s}\n{h.text}")
    return "\n\n---\n\n".join(parts)


def _build_agent():
    retrieve = get_retriever(k=5)

    @tool("faq_policy_lookup")
    def faq_policy_lookup(query: str) -> str:
        """Look up Tech Haven FAQ/policy excerpts relevant to a customer question."""
        return _format_hits(retrieve(query))

    tools = [faq_policy_lookup]

    system_prompt = (
        "You are a customer support agent for Tech Haven.\n"
        "Write friendly, professional email replies with a few appropriate emojis.\n"
        "Use the faq_policy_lookup tool when you need accurate details from Tech Haven FAQs/policies.\n"
        "If the docs don't contain an answer, say so briefly and ask a clarifying question.\n"
        "Sign off EXACTLY as: Mr. Helpful\n"
        "Return ONLY the email reply body (no subject line, no greetings outside the body, no JSON, no markdown)."
    )

    # LangGraph prebuilt agent (current LangChain v1 agent stack)
    # Uses LCEL-style tool calling with the provided tools list.
    return create_react_agent(
        model=_get_llm(),
        tools=tools,
        prompt=SystemMessage(content=system_prompt),
    )


def _build_retriever():
    return get_retriever(k=5)


_AGENT = None
_RETRIEVE = None


def generate_reply(email_body: str) -> Union[str, Dict[str, Any]]:
    """
    Takes an email body as input.

    If the knowledge base has no relevant hits, returns {"escalate": True}.
    Otherwise returns ONLY the reply body (string).
    """
    global _AGENT, _RETRIEVE
    if _RETRIEVE is None:
        _RETRIEVE = _build_retriever()

    body = (email_body or "").strip()
    if not body:
        return "Hi there,\n\nHow can we help you today? 🙂\n\nMr. Helpful"

    hits = _RETRIEVE(body)
    if not hits:
        return {"escalate": True}

    if _AGENT is None:
        _AGENT = _build_agent()

    result = _AGENT.invoke(
        {
            "messages": [
                HumanMessage(content=f"Customer email:\n{body}"),
            ]
        }
    )

    # Result follows the standard LangGraph schema: {"messages": [...]}
    messages = result.get("messages") or []
    for m in reversed(messages):
        content = getattr(m, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


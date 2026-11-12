from doc_agent.nodes._utils import (
    format_context,
    guard_llm_errors,
    question_text,
    reply,
    source_refs,
)
from doc_agent.prompts import QA
from doc_agent.services.base import LLMService, VectorStore
from doc_agent.state import AgentState


def make_qa(llm: LLMService, store: VectorStore, k: int = 4):
    def run(state: AgentState) -> dict:
        question = question_text(state)
        chunks = store.search(question, k)
        if not chunks:
            return reply(
                "I couldn't find anything relevant in the ingested documents.",
                context=[],
                sources=[],
            )
        answer = llm.invoke(QA.format(context=format_context(chunks), question=question))
        return reply(answer, context=chunks, sources=source_refs(chunks))

    return guard_llm_errors("qa", run)

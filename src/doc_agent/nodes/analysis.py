from doc_agent.nodes._utils import (
    format_context,
    guard_llm_errors,
    question_text,
    reply,
    source_refs,
    warn_invalid_citations,
)
from doc_agent.prompts import ANALYSIS
from doc_agent.services.base import LLMService, VectorStore
from doc_agent.state import AgentState


def make_analysis(llm: LLMService, store: VectorStore, k: int = 8):
    def run(state: AgentState) -> dict:
        question = question_text(state)
        chunks = store.search(question, k) or store.all_chunks()[:k]
        if not chunks:
            return reply("No documents have been ingested yet.", context=[], sources=[])
        answer = llm.invoke(ANALYSIS.format(context=format_context(chunks), question=question))
        warn_invalid_citations(answer, len(chunks), "analysis")
        return reply(answer, context=chunks, sources=source_refs(chunks))

    return guard_llm_errors("analysis", run)

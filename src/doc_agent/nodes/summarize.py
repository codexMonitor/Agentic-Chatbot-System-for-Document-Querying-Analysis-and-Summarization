from doc_agent.nodes._utils import reply
from doc_agent.prompts import SUMMARIZE_CHUNK, SUMMARIZE_REDUCE
from doc_agent.services.base import LLMService, VectorStore
from doc_agent.state import AgentState


def make_summarize(llm: LLMService, store: VectorStore, max_chunks: int = 20):
    def run(state: AgentState) -> dict:
        chunks = store.all_chunks()[:max_chunks]
        if not chunks:
            return reply("No documents have been ingested yet.", context=[], sources=[])
        partials = [llm.invoke(SUMMARIZE_CHUNK.format(chunk=chunk.text)) for chunk in chunks]
        bullets = "\n".join(f"- {partial}" for partial in partials)
        answer = llm.invoke(SUMMARIZE_REDUCE.format(partials=bullets))
        # Summaries are not cited per chunk; empty sources also clear stale ones from earlier turns.
        return reply(answer, context=chunks, sources=[])

    return run

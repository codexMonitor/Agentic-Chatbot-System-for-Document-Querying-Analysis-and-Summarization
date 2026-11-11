from doc_agent.nodes._utils import last_user_text, reply
from doc_agent.prompts import ANALYSIS
from doc_agent.services.base import LLMService, VectorStore
from doc_agent.state import AgentState


def make_analysis(llm: LLMService, store: VectorStore, k: int = 8):
    def run(state: AgentState) -> dict:
        question = last_user_text(state)
        chunks = store.search(question, k) or store.all_chunks()[:k]
        if not chunks:
            return reply("No documents have been ingested yet.", context=[])
        context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(chunks, start=1))
        answer = llm.invoke(ANALYSIS.format(context=context, question=question))
        return reply(answer, context=chunks)

    return run

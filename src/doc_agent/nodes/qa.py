from doc_agent.nodes._utils import last_user_text, reply
from doc_agent.prompts import QA
from doc_agent.services.base import LLMService, VectorStore
from doc_agent.state import AgentState


def make_qa(llm: LLMService, store: VectorStore, k: int = 4):
    def run(state: AgentState) -> dict:
        question = last_user_text(state)
        chunks = store.search(question, k)
        if not chunks:
            return reply("I couldn't find anything relevant in the ingested documents.", context=[])
        context = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(chunks, start=1))
        answer = llm.invoke(QA.format(context=context, question=question))
        return reply(answer, context=chunks)

    return run

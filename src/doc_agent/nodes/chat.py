from doc_agent.nodes._utils import guard_llm_errors, question_text, reply
from doc_agent.prompts import CHAT
from doc_agent.services.base import LLMService
from doc_agent.state import AgentState


def make_chat(llm: LLMService):
    """Answer greetings and assistant meta questions without retrieving documents."""

    def run(state: AgentState) -> dict:
        answer = llm.invoke(CHAT.format(question=question_text(state)))
        return reply(answer, context=[], sources=[])

    return guard_llm_errors("chat", run)

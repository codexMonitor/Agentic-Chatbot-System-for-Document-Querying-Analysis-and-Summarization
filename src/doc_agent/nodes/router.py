import re

from doc_agent.nodes._utils import last_user_text
from doc_agent.prompts import ROUTER
from doc_agent.services.base import LLMService
from doc_agent.state import AgentState

_INTENTS = {"qa", "analysis", "summarize"}


def make_router(llm: LLMService):
    def run(state: AgentState) -> dict:
        raw = llm.invoke(ROUTER.format(question=last_user_text(state)))
        words = re.findall(r"\w+", raw.lower())
        word = words[0] if words else ""
        return {"intent": word if word in _INTENTS else "qa"}

    return run

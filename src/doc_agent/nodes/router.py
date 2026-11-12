import logging
import re

from doc_agent.nodes._utils import last_user_text
from doc_agent.prompts import ROUTER
from doc_agent.services.base import LLMService
from doc_agent.state import AgentState

logger = logging.getLogger(__name__)

_INTENTS = {"qa", "analysis", "summarize"}
_DEFAULT_INTENT = "qa"


def parse_intent(raw: str) -> str:
    """First intent word in the model output (tolerates 'Intent: qa'); `qa` if there is none."""
    for word in re.findall(r"\w+", raw.lower()):
        if word in _INTENTS:
            return word
    return _DEFAULT_INTENT


def make_router(llm: LLMService):
    def run(state: AgentState) -> dict:
        intent = parse_intent(llm.invoke(ROUTER.format(question=last_user_text(state))))
        logger.info("router: intent=%s", intent)
        return {"intent": intent}

    return run

import logging
import re

from doc_agent.nodes._utils import question_text
from doc_agent.prompts import ROUTER
from doc_agent.services.base import LLMService
from doc_agent.services.errors import LLMError
from doc_agent.state import AgentState

logger = logging.getLogger(__name__)

_INTENTS = {"chat", "qa", "analysis", "summarize"}
_DEFAULT_INTENT = "qa"


def parse_intent(raw: str) -> str:
    """First intent word in the model output (tolerates 'Intent: qa'); `qa` if there is none."""
    for word in re.findall(r"\w+", raw.lower()):
        if word in _INTENTS:
            return word
    return _DEFAULT_INTENT


def make_router(llm: LLMService):
    def run(state: AgentState) -> dict:
        try:
            raw = llm.invoke(ROUTER.format(question=question_text(state)))
        except LLMError as error:
            # The chosen node makes its own LLM call and reports the outage to the user.
            logger.warning("router: LLM unavailable (%s), using intent=%s", error, _DEFAULT_INTENT)
            return {"intent": _DEFAULT_INTENT}
        intent = parse_intent(raw)
        logger.info("router: intent=%s", intent)
        return {"intent": intent}

    return run

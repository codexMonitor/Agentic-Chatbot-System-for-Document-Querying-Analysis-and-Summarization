import logging

from langchain_core.messages import AIMessage

from doc_agent.nodes._utils import last_user_text
from doc_agent.prompts import REWRITE
from doc_agent.services.base import LLMService
from doc_agent.services.errors import LLMError
from doc_agent.state import AgentState

logger = logging.getLogger(__name__)

# Long answers (for example summaries) are cut so the rewrite prompt stays small.
_MAX_TURN_CHARS = 600


def _format_turn(message) -> str:
    role = "Assistant" if isinstance(message, AIMessage) else "User"
    content = message.content if isinstance(message.content, str) else str(message.content)
    if len(content) > _MAX_TURN_CHARS:
        content = content[:_MAX_TURN_CHARS] + "..."
    return f"{role}: {content}"


def make_rewrite(llm: LLMService, history_turns: int = 4):
    """Turn a follow-up such as "how long are they?" into a standalone question."""

    def run(state: AgentState) -> dict:
        question = last_user_text(state)
        # The new question is the last message; everything before it is history.
        earlier = (state.get("messages") or [])[:-1]
        if history_turns <= 0 or not earlier:
            return {"standalone_question": question}
        conversation = "\n".join(_format_turn(message) for message in earlier[-history_turns:])
        try:
            raw = llm.invoke(REWRITE.format(conversation=conversation, question=question))
        except LLMError as error:
            logger.warning("rewrite: LLM unavailable (%s), using the original question", error)
            return {"standalone_question": question}
        standalone = raw.strip() or question
        logger.info("rewrite: changed=%s", standalone != question)
        return {"standalone_question": standalone}

    return run

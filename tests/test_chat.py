from langchain_core.messages import HumanMessage

from doc_agent.config import Settings
from doc_agent.graph import build_graph, memory_checkpointer
from doc_agent.nodes.chat import make_chat
from doc_agent.services import Services
from doc_agent.services.llm import FakeLLM
from doc_agent.services.vectorstore import InMemoryVectorStore


class RecordingLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def test_chat_makes_one_llm_call_and_does_not_require_a_store():
    llm = RecordingLLM("Hello! I can help with your documents.")
    result = make_chat(llm)({"messages": [HumanMessage(content="Hello")]})

    assert len(llm.prompts) == 1
    assert "uploaded documents" in llm.prompts[0]
    assert "citations" in llm.prompts[0]
    assert result["answer"] == llm.answer
    assert result["context"] == []
    assert result["sources"] == []


def test_chat_refuses_off_topic_questions():
    llm = RecordingLLM("I can only help with your uploaded documents.")
    result = make_chat(llm)({"messages": [HumanMessage(content="What is the capital of France?")]})

    assert "only help with uploaded documents" in llm.prompts[0]
    assert result["answer"] == "I can only help with your uploaded documents."
    assert result["sources"] == []


def test_greeting_on_an_empty_store_routes_to_chat():
    graph = build_graph(
        Services(llm=FakeLLM(), store=InMemoryVectorStore()), Settings(), memory_checkpointer()
    )
    result = graph.invoke(
        {"messages": [("user", "Hello!")]}, {"configurable": {"thread_id": "chat-empty"}}
    )

    assert result["intent"] == "chat"
    assert result["context"] == []
    assert result["sources"] == []
    assert "uploaded documents" in result["answer"]

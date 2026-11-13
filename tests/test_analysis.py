from langchain_core.messages import HumanMessage

from doc_agent.nodes.analysis import make_analysis
from doc_agent.services.base import Chunk
from doc_agent.services.vectorstore import InMemoryVectorStore

STRUCTURED_ANSWER = """Summary
The two densities produced different energy spreads.

Key points
- Higher density produced a larger energy spread.

Evidence
- The measured spreads were 1.6 MeV and 4.5 MeV [1].

Open questions
- The excerpt does not explain the cause of the difference."""


class FixedLLM:
    def __init__(self, answer: str = STRUCTURED_ANSWER) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def test_analysis_returns_structured_answer_with_context_and_sources():
    store = InMemoryVectorStore()
    store.add("paper.pdf", ["The RMS energy spreads were 1.6 MeV and 4.5 MeV."])
    llm = FixedLLM()

    result = make_analysis(llm, store, k=2)(
        {"messages": [HumanMessage(content="Compare the energy spreads")]}
    )

    assert result["answer"] == STRUCTURED_ANSWER
    assert result["context"] == [Chunk(store.all_chunks()[0].text, "paper.pdf", 0)]
    assert result["sources"] == [{"n": 1, "source": "paper.pdf", "position": 0}]
    for heading in ("Summary", "Key points", "Evidence", "Open questions"):
        assert heading in llm.prompts[0]


def test_analysis_falls_back_to_first_chunks_when_search_has_no_match():
    store = InMemoryVectorStore()
    store.add("paper.pdf", ["The paper reports a measured energy spread."])

    result = make_analysis(FixedLLM(), store, k=1)(
        {"messages": [HumanMessage(content="unmatched query")]}
    )

    assert result["context"] == store.all_chunks()
    assert result["sources"] == [{"n": 1, "source": "paper.pdf", "position": 0}]


def test_analysis_empty_store_returns_no_sources_without_calling_llm():
    class UnusedLLM:
        def invoke(self, prompt: str) -> str:
            raise AssertionError("the LLM should not be called for an empty store")

    result = make_analysis(UnusedLLM(), InMemoryVectorStore())(
        {"messages": [HumanMessage(content="Compare the results")]}
    )

    assert result["answer"] == "No documents have been ingested yet."
    assert result["context"] == []
    assert result["sources"] == []

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from doc_agent.config import Settings
from doc_agent.graph import build_graph, memory_checkpointer, sqlite_checkpointer
from doc_agent.nodes._utils import question_text
from doc_agent.nodes.rewrite import make_rewrite
from doc_agent.nodes.router import make_router
from doc_agent.services import Services
from doc_agent.services.errors import LLMError
from doc_agent.services.llm import FakeLLM
from doc_agent.services.vectorstore import InMemoryVectorStore

CHUNKS = [
    "Scrum is a framework for agile teams.",
    "Scrum sprints last one to four weeks.",
    "Chroma stores embeddings on disk.",
]


class SpyStore(InMemoryVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    def search(self, query, k=4):
        self.queries.append(query)
        return super().search(query, k)


class ScriptedLLM(FakeLLM):
    """FakeLLM that records rewrite prompts and rewrites follow-ups from a script."""

    def __init__(self, rewrites=None, fail_rewrite=False) -> None:
        self.rewrites = rewrites or {}
        self.fail_rewrite = fail_rewrite
        self.rewrite_prompts: list[str] = []

    def invoke(self, prompt):
        if "### TASK: rewrite" in prompt:
            self.rewrite_prompts.append(prompt)
            if self.fail_rewrite:
                raise LLMError("provider down")
            question = prompt.rpartition("New question:")[2].strip()
            return self.rewrites.get(question, question)
        return super().invoke(prompt)


def make_graph(llm, store=None, settings=None, checkpointer=None):
    if store is None:
        store = SpyStore()
        store.add("notes.txt", CHUNKS)
    return build_graph(
        Services(llm=llm, store=store),
        settings or Settings(),
        checkpointer or memory_checkpointer(),
    )


def ask(graph, question, name="t"):
    return graph.invoke({"messages": [("user", question)]}, {"configurable": {"thread_id": name}})


def test_first_turn_makes_no_rewrite_call():
    llm = ScriptedLLM()
    result = ask(make_graph(llm), "What is Scrum?")
    assert llm.rewrite_prompts == []
    assert result["standalone_question"] == "What is Scrum?"


def test_second_turn_makes_exactly_one_rewrite_call():
    llm = ScriptedLLM()
    graph = make_graph(llm)
    ask(graph, "What is Scrum?")
    ask(graph, "How long are sprints?")
    assert len(llm.rewrite_prompts) == 1
    ask(graph, "What is Chroma?")
    assert len(llm.rewrite_prompts) == 2


def test_retrieval_receives_the_rewritten_question():
    store = SpyStore()
    store.add("notes.txt", CHUNKS)
    llm = ScriptedLLM(rewrites={"How long are they?": "How long are Scrum sprints?"})
    graph = make_graph(llm, store)
    ask(graph, "What is Scrum?")
    result = ask(graph, "How long are they?")
    assert store.queries == ["What is Scrum?", "How long are Scrum sprints?"]
    assert result["standalone_question"] == "How long are Scrum sprints?"
    assert result["sources"][0] == {"n": 1, "source": "notes.txt", "position": 1}


def test_rewrite_prompt_has_history_and_the_new_question():
    llm = ScriptedLLM()
    graph = make_graph(llm)
    ask(graph, "What is Scrum?")
    ask(graph, "How long are they?")
    payload = llm.rewrite_prompts[0].partition("\n---\n")[2]
    assert payload.startswith("Conversation:\nUser: What is Scrum?\nAssistant: ")
    assert payload.endswith("\n\nNew question: How long are they?")


def test_llm_error_in_rewrite_falls_back_to_the_original_question(caplog):
    store = SpyStore()
    store.add("notes.txt", CHUNKS)
    graph = make_graph(ScriptedLLM(fail_rewrite=True), store)
    ask(graph, "What is Scrum?")
    with caplog.at_level(logging.WARNING):
        result = ask(graph, "How long are sprints?")
    assert store.queries[-1] == "How long are sprints?"
    assert "sprints last" in result["answer"]
    assert "rewrite: LLM unavailable" in caplog.text


def test_scrum_follow_up_is_rewritten_and_qa_searches_for_it():
    store = SpyStore()
    store.add("notes.txt", CHUNKS)
    llm = ScriptedLLM(rewrites={"How long are they?": "How long are Scrum sprints?"})
    graph = make_graph(llm, store)
    first = ask(graph, "What is Scrum?")
    assert first["intent"] == "qa"
    second = ask(graph, "How long are they?")
    assert second["intent"] == "qa"
    assert store.queries[-1] == "How long are Scrum sprints?"
    assert "Scrum sprints last" in second["answer"]


def test_history_survives_a_rebuilt_graph_on_the_same_sqlite_file(tmp_path):
    db = tmp_path / "memory.sqlite"
    first_saver = sqlite_checkpointer(db)
    ask(make_graph(ScriptedLLM(), checkpointer=first_saver), "What is Scrum?")
    first_saver.conn.close()

    llm = ScriptedLLM(rewrites={"How long are they?": "How long are Scrum sprints?"})
    store = SpyStore()
    store.add("notes.txt", CHUNKS)
    second_saver = sqlite_checkpointer(db)
    result = ask(make_graph(llm, store, checkpointer=second_saver), "How long are they?")
    second_saver.conn.close()
    assert len(llm.rewrite_prompts) == 1
    assert "User: What is Scrum?" in llm.rewrite_prompts[0]
    assert store.queries == ["How long are Scrum sprints?"]
    assert len(result["messages"]) == 4


def test_zero_history_turns_disables_rewriting_and_clears_the_old_value():
    llm = ScriptedLLM(rewrites={"How long are they?": "SHOULD NOT BE USED"})
    graph = make_graph(llm, settings=Settings(rewrite_history_turns=0))
    ask(graph, "What is Scrum?")
    result = ask(graph, "How long are they?")
    assert llm.rewrite_prompts == []
    assert result["standalone_question"] == "How long are they?"


def test_only_the_last_n_messages_are_sent():
    llm = ScriptedLLM()
    graph = make_graph(llm, settings=Settings(rewrite_history_turns=2))
    ask(graph, "What is Scrum?")
    ask(graph, "What is Chroma?")
    ask(graph, "And where is it stored?")
    conversation = llm.rewrite_prompts[-1].partition("Conversation:\n")[2]
    conversation = conversation.partition("\n\nNew question:")[0]
    assert "What is Scrum?" not in conversation
    assert conversation.splitlines()[0] == "User: What is Chroma?"
    assert len(conversation.splitlines()) == 2


def test_long_assistant_turns_are_truncated():
    llm = ScriptedLLM()
    rewrite = make_rewrite(llm, history_turns=4)
    state = {
        "messages": [HumanMessage("q1"), AIMessage("x" * 5000), HumanMessage("and then?")],
    }
    rewrite(state)
    assert len(llm.rewrite_prompts[0]) < 1500


def test_empty_rewrite_output_falls_back_to_the_original():
    class Blank:
        def invoke(self, prompt):
            return "  \n"

    state = {"messages": [HumanMessage("q1"), AIMessage("a1"), HumanMessage("and then?")]}
    assert make_rewrite(Blank())(state) == {"standalone_question": "and then?"}


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"standalone_question": "standalone", "messages": []}, "standalone"),
        ({"standalone_question": "", "messages": []}, ""),
        ({}, ""),
    ],
)
def test_question_text_prefers_the_standalone_question(state, expected):
    assert question_text(state) == expected


def test_question_text_falls_back_to_the_last_message():
    assert question_text({"messages": [HumanMessage("hello")]}) == "hello"
    assert question_text({"standalone_question": "", "messages": [HumanMessage("hi")]}) == "hi"


def test_router_classifies_the_standalone_question():
    state = {
        "messages": [HumanMessage("do that again")],
        "standalone_question": "Summarize the document",
    }
    assert make_router(FakeLLM())(state) == {"intent": "summarize"}


def test_fake_llm_rewrite_returns_the_new_question_unchanged():
    prompt = (
        "### TASK: rewrite\nrules\n---\nConversation:\nUser: hi\n\nNew question: How long are they?"
    )
    assert FakeLLM().invoke(prompt) == "How long are they?"

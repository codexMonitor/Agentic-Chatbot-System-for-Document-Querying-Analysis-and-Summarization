from doc_agent.prompts import ANALYSIS, QA


def test_qa_and_analysis_prompts_require_the_question_language():
    instruction = "Answer in the same language as the question."

    assert instruction in QA
    assert instruction in ANALYSIS

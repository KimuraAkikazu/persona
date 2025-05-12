# tests/test_runner_templates.py

from runner import _TPL_R1, _TPL_RN


def test_round1_template_contains_fields():
    rendered = _TPL_R1.render(question_text="Q?")
    assert '"reasoning"' in rendered
    assert '"answer"' in rendered


def test_roundN_template_includes_turn_and_others():
    other1 = {"reasoning": "first reason", "answer": "B"}
    other2 = {"reasoning": "second reason", "answer": "D"}
    rendered = _TPL_RN.render(turn=2, other1=other1, other2=other2)
    assert "Round 2" in rendered
    assert "first reason" in rendered and "D" in rendered

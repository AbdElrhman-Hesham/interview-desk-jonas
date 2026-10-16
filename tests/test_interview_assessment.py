"""Final interview assessment (app/interview_report.assess) with a scripted model.

No model, microphone, PDF or learner data: a fake LLM streams a fixed reply and
a fake controller holds synthetic answers. These tests pin how model output is
normalised and what the no-model fallback is allowed to claim. They say nothing
about whether the model's feedback is pedagogically valid.

Run:  python -B tests/test_interview_assessment.py
  or: python -m pytest -q tests/test_interview_assessment.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["LOCALAPPDATA"] = tempfile.mkdtemp(prefix="interview-desk-data-")  # see test_local_server_guard

from app import interview_report as report  # noqa: E402

LONG = " ".join(["I listened to the customer, confirmed the account and explained the next step clearly."] * 4)


def controller(n_answers=3, total=3, language="en", answer=LONG):
    answers = [{"id": f"q{i}", "text": f"Question {i}?", "answer": answer} for i in range(1, n_answers + 1)]
    return types.SimpleNamespace(answers=answers, total=total, language=language)


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply

    def stream_chat(self, messages, max_tokens=0):
        yield self.reply, {}


class BrokenLLM:
    def stream_chat(self, messages, max_tokens=0):
        raise ConnectionError("llama-server is gone")
        yield  # pragma: no cover


def model_reply(**overrides) -> str:
    body = {
        "overall_score": 72, "verdict": "Promising", "summary": "Clear, polite answers.",
        "dimensions": [{"key": "communication", "score": 80, "note": "Clear."}],
        "strengths": ["Polite tone"], "improvements": ["Add an outcome"],
        "answer_feedback": [{"question_id": "q1", "score": 75, "feedback": "Good.", "better_answer": "..."}],
    }
    body.update(overrides)
    return json.dumps(body)


def test_model_scores_are_clamped_and_every_answer_gets_feedback():
    ev = report.assess(FakeLLM(model_reply(overall_score="150")), controller())
    assert ev["assessment_mode"] == "local_model"
    assert ev["overall_score"] == 100
    assert [f["question_id"] for f in ev["answer_feedback"]] == ["q1", "q2", "q3"]


def test_string_lists_from_the_model_are_not_split_into_letters():
    ev = report.assess(FakeLLM(model_reply(strengths="Clear and polite answers.",
                                           improvements="Give a concrete example.")), controller())
    assert ev["strengths"] == ["Clear and polite answers."], ev["strengths"]
    assert ev["improvements"] == ["Give a concrete example."], ev["improvements"]


def test_model_reply_without_summary_does_not_say_model_was_unavailable():
    body = json.loads(model_reply())
    del body["summary"], body["verdict"]
    ev = report.assess(FakeLLM(json.dumps(body)), controller())
    assert ev["assessment_mode"] == "local_model"
    assert "not available" not in ev["summary"], ev["summary"]
    # unchanged rule when the model scored but gave no verdict: >= 70 -> "Interview ready"
    assert ev["verdict"] == "Interview ready", ev["verdict"]


def test_unparseable_or_failed_model_falls_back():
    for llm in (FakeLLM("Sorry, I cannot help with that."), BrokenLLM()):
        ev = report.assess(llm, controller())
        assert ev["assessment_mode"] == "fallback"


def test_fallback_never_claims_interview_ready():
    # Long, complete answers maximise the length-based baseline (100/100).
    ev = report.assess(BrokenLLM(), controller(n_answers=6, total=6))
    assert ev["overall_score"] == 100
    assert "ready" not in ev["verdict"].lower(), ev["verdict"]


def test_fallback_does_not_invent_differences_between_dimensions():
    ev = report.assess(BrokenLLM(), controller(n_answers=2, total=6, answer="Yes, I can do that job well."))
    scores = {d["key"]: d["score"] for d in ev["dimensions"]}
    assert len(set(scores.values())) == 1, scores


TESTS = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]

if __name__ == "__main__":
    failed = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)

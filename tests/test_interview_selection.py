"""Focused tests for question selection, count validation, selected-order
execution, detailed evaluation data, PDF download authorization, and
return-home/reset behavior.

These tests import only the interview and interview-report modules (pure
Python, no audio/model dependencies) so they can run on any machine without
the full runtime.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add the project root to the path so `app` can be imported as a package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.interview import (
    GERMAN_QUESTIONS, ENGLISH_QUESTIONS, LANGUAGES,
    InterviewController, InterviewQuestion,
    question_bank, validate_selection, validate_start_params, valid_language,
)


# ─── question_bank API ───────────────────────────────────────────────

class TestQuestionBank(unittest.TestCase):
    def test_returns_all_german_questions(self):
        bank = question_bank("de")
        self.assertEqual(len(bank), len(GERMAN_QUESTIONS))
        self.assertIsInstance(bank[0], dict)
        self.assertIn("id", bank[0])

    def test_returns_all_english_questions(self):
        bank = question_bank("en")
        self.assertEqual(len(bank), len(ENGLISH_QUESTIONS))

    def test_invalid_language_returns_empty(self):
        self.assertEqual(question_bank("fr"), [])
        self.assertEqual(question_bank(""), [])

    def test_ids_are_unique(self):
        for lang in ("de", "en"):
            ids = [q["id"] for q in question_bank(lang)]
            self.assertEqual(len(ids), len(set(ids)), f"Duplicate IDs in {lang}")


# ─── validate_selection ──────────────────────────────────────────────

class TestValidateSelection(unittest.TestCase):
    def test_valid_full_bank(self):
        ids = [q.id for q in GERMAN_QUESTIONS]
        qs, err = validate_selection("de", ids, len(ids))
        self.assertIsNone(err)
        self.assertEqual(len(qs), len(ids))

    def test_valid_subset(self):
        ids = ["de-01", "de-05", "de-12"]
        qs, err = validate_selection("de", ids, 3)
        self.assertIsNone(err)
        self.assertEqual(len(qs), 3)
        self.assertEqual(qs[0].id, "de-01")
        self.assertEqual(qs[1].id, "de-05")
        self.assertEqual(qs[2].id, "de-12")

    def test_invalid_language(self):
        _, err = validate_selection("xx", ["de-01"], 1)
        self.assertIn("Invalid", err)

    def test_count_zero(self):
        _, err = validate_selection("de", [], 0)
        self.assertIn("Count must be", err)

    def test_count_too_high(self):
        ids = [q.id for q in GERMAN_QUESTIONS] + ["extra"]
        _, err = validate_selection("de", ids, len(ids))
        self.assertIn("Count must be", err)

    def test_count_mismatch(self):
        _, err = validate_selection("de", ["de-01", "de-02"], 3)
        self.assertIn("Expected exactly", err)

    def test_duplicate_ids(self):
        _, err = validate_selection("de", ["de-01", "de-01"], 2)
        self.assertIn("Duplicate", err)

    def test_unknown_id(self):
        _, err = validate_selection("de", ["de-01", "en-02"], 2)
        self.assertIn("Unknown", err)

    def test_english_bank(self):
        ids = ["en-03", "en-07"]
        qs, err = validate_selection("en", ids, 2)
        self.assertIsNone(err)
        self.assertEqual(qs[0].id, "en-03")
        self.assertEqual(qs[1].id, "en-07")


# ─── validate_start_params (server-side input gate) ──────────────────

class TestValidateStartParams(unittest.TestCase):
    def test_valid_with_selection(self):
        p, err = validate_start_params({
            "language": "de",
            "selected_ids": ["de-01", "de-02"],
            "count": 2,
        })
        self.assertIsNone(err)
        self.assertEqual(p["language"], "de")
        self.assertEqual(p["selected_ids"], ["de-01", "de-02"])
        self.assertEqual(p["count"], 2)

    def test_valid_without_selection(self):
        p, err = validate_start_params({"language": "en"})
        self.assertIsNone(err)
        self.assertNotIn("selected_ids", p)
        self.assertNotIn("count", p)

    def test_missing_language(self):
        _, err = validate_start_params({})
        self.assertIn("language", err)

    def test_empty_language(self):
        _, err = validate_start_params({"language": ""})
        self.assertIn("language", err)

    def test_language_not_string(self):
        _, err = validate_start_params({"language": 42})
        self.assertIn("language", err)

    def test_partial_only_selected_ids(self):
        _, err = validate_start_params({"language": "de", "selected_ids": ["de-01"]})
        self.assertIn("both be present", err)

    def test_partial_only_count(self):
        _, err = validate_start_params({"language": "de", "count": 3})
        self.assertIn("both be present", err)

    def test_selected_ids_not_list(self):
        _, err = validate_start_params({
            "language": "de", "selected_ids": "de-01", "count": 1,
        })
        self.assertIn("list", err)

    def test_selected_ids_contain_non_string(self):
        _, err = validate_start_params({
            "language": "de", "selected_ids": ["de-01", 42], "count": 2,
        })
        self.assertIn("string", err)

    def test_count_is_bool(self):
        _, err = validate_start_params({
            "language": "de", "selected_ids": ["de-01"], "count": True,
        })
        self.assertIn("integer", err)

    def test_count_is_float(self):
        _, err = validate_start_params({
            "language": "de", "selected_ids": ["de-01"], "count": 1.5,
        })
        self.assertIn("integer", err)

    def test_count_is_string(self):
        _, err = validate_start_params({
            "language": "de", "selected_ids": ["de-01"], "count": "1",
        })
        self.assertIn("integer", err)


# ─── InterviewController with selection ──────────────────────────────

class TestInterviewControllerSelection(unittest.TestCase):
    def test_default_uses_full_bank(self):
        ctrl = InterviewController("de")
        self.assertEqual(ctrl.total, len(GERMAN_QUESTIONS))
        self.assertEqual(ctrl.questions, GERMAN_QUESTIONS)

    def test_selected_subset(self):
        ctrl = InterviewController("de", selected_ids=["de-03", "de-07", "de-11"], count=3)
        self.assertEqual(ctrl.total, 3)
        self.assertEqual(ctrl.questions[0].id, "de-03")
        self.assertEqual(ctrl.questions[1].id, "de-07")
        self.assertEqual(ctrl.questions[2].id, "de-11")

    def test_invalid_selection_raises(self):
        with self.assertRaises(ValueError):
            InterviewController("de", selected_ids=["de-01", "de-01"], count=2)

    def test_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            InterviewController("de", selected_ids=["xx-99"], count=1)

    def test_backward_compat_no_selection(self):
        ctrl = InterviewController("en")
        self.assertEqual(ctrl.total, len(ENGLISH_QUESTIONS))


# ─── selected-order execution ────────────────────────────────────────

class TestSelectedOrderExecution(unittest.TestCase):
    def test_questions_asked_in_selection_order(self):
        ids = ["de-12", "de-01", "de-06"]
        ctrl = InterviewController("de", selected_ids=ids, count=3)
        order = []
        while not ctrl.complete:
            q = ctrl.current_question
            order.append(q.id)
            ctrl.record_answer(f"Answer to {q.id}")
        self.assertEqual(order, ids)
        self.assertEqual(len(ctrl.answers), 3)
        self.assertTrue(ctrl.complete)

    def test_total_matches_selection(self):
        ctrl = InterviewController("en", selected_ids=["en-02", "en-08"], count=2)
        self.assertEqual(ctrl.total, 2)
        snap = ctrl.snapshot()
        self.assertEqual(snap["total"], 2)

    def test_record_stores_correct_number(self):
        ctrl = InterviewController("de", selected_ids=["de-05", "de-10"], count=2)
        rec = ctrl.record_answer("First answer")
        self.assertEqual(rec["number"], 1)
        self.assertEqual(rec["id"], "de-05")
        rec2 = ctrl.record_answer("Second answer")
        self.assertEqual(rec2["number"], 2)
        self.assertEqual(rec2["id"], "de-10")


# ─── evaluation data enrichment ──────────────────────────────────────

class TestEvaluationData(unittest.TestCase):
    def test_fallback_includes_answer_text(self):
        """The fallback evaluator must include _answer_text in feedback."""
        from app.interview_report import _fallback
        ctrl = InterviewController("de", selected_ids=["de-01", "de-02"], count=2)
        ctrl.record_answer("Ich bin ein Kandidat.")
        ctrl.record_answer("Ich arbeite gern im Kundenservice.")
        fb = _fallback(ctrl)
        for entry in fb["answer_feedback"]:
            self.assertIn("_answer_text", entry)
            self.assertTrue(len(entry["_answer_text"]) > 0)


# ─── PDF download path validation ────────────────────────────────────

class TestPdfPathValidation(unittest.TestCase):
    """Test the path-safety logic used by _serve_report.

    We test the pure logic (resolve + parent check) without needing
    the HTTP server running.
    """
    def _is_allowed(self, raw_path: str, allowed_dir: Path) -> bool:
        """Replicate the server's check."""
        path = Path(raw_path).resolve()
        if path.suffix.lower() != ".pdf" or not path.is_file():
            return False
        allowed = allowed_dir.resolve()
        return allowed in path.parents

    def test_pdf_in_reports_dir(self):
        with tempfile.TemporaryDirectory() as d:
            reports = Path(d) / "reports"
            reports.mkdir()
            pdf = reports / "Interview-Report-DE-20260916.pdf"
            pdf.write_bytes(b"%PDF-1.4 test")
            self.assertTrue(self._is_allowed(str(pdf), reports))

    def test_pdf_outside_reports_dir(self):
        with tempfile.TemporaryDirectory() as d:
            reports = Path(d) / "reports"
            reports.mkdir()
            outside = Path(d) / "secrets.pdf"
            outside.write_bytes(b"secret")
            self.assertFalse(self._is_allowed(str(outside), reports))

    def test_non_pdf_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            reports = Path(d) / "reports"
            reports.mkdir()
            txt = reports / "data.txt"
            txt.write_text("not a pdf")
            self.assertFalse(self._is_allowed(str(txt), reports))

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            reports = Path(d) / "reports"
            reports.mkdir()
            outside = Path(d) / "evil.pdf"
            outside.write_bytes(b"%PDF-1.4 evil")
            traversal = str(reports / ".." / "evil.pdf")
            self.assertFalse(self._is_allowed(traversal, reports))


# ─── snapshot / progress ─────────────────────────────────────────────

class TestSnapshot(unittest.TestCase):
    def test_snapshot_with_selection(self):
        ctrl = InterviewController("de", selected_ids=["de-03", "de-09"], count=2)
        snap = ctrl.snapshot()
        self.assertTrue(snap["active"])
        self.assertEqual(snap["total"], 2)
        self.assertEqual(snap["answered"], 0)
        self.assertFalse(snap["complete"])
        self.assertEqual(snap["current_question"]["id"], "de-03")

    def test_complete_snapshot(self):
        ctrl = InterviewController("en", selected_ids=["en-01"], count=1)
        ctrl.record_answer("Hello I am applying.")
        snap = ctrl.snapshot()
        self.assertTrue(snap["complete"])
        self.assertEqual(snap["answered"], 1)
        self.assertIsNone(snap["current_question"])


# ─── valid_language ──────────────────────────────────────────────────

class TestValidLanguage(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(valid_language("de"), "de")
        self.assertEqual(valid_language("en"), "en")
        self.assertEqual(valid_language("  DE  "), "de")

    def test_invalid(self):
        self.assertIsNone(valid_language("fr"))
        self.assertIsNone(valid_language(""))
        self.assertIsNone(valid_language(None))


if __name__ == "__main__":
    unittest.main()

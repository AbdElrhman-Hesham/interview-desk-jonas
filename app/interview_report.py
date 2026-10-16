"""Interview assessment and polished local PDF report."""
from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
from pathlib import Path

from . import config


SCORE_LABELS = {
    "de": {
        "communication": "Kommunikation",
        "language": "Sprachliche Sicherheit",
        "customer_service": "Kundenorientierung",
        "problem_solving": "Problemlösung",
        "professionalism": "Professionalität",
    },
    "en": {
        "communication": "Communication",
        "language": "Language control",
        "customer_service": "Customer focus",
        "problem_solving": "Problem solving",
        "professionalism": "Professionalism",
    },
}


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                  flags=re.IGNORECASE | re.DOTALL)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _clamp_score(value, default=0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except Exception:
        return default


def _text_list(value, max_items: int, max_chars: int = 500) -> list[str]:
    """Model 'lists' sometimes arrive as one string; never slice that into letters."""
    if isinstance(value, str):
        value = [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(x)[:max_chars] for x in value[:max_items]]


def _score_verdict(score: int) -> str:
    return "Developing" if score < 70 else "Interview ready"


def _fallback(controller) -> dict:
    answers = controller.answers
    total = max(1, controller.total)
    completed = len(answers) / total
    counts = [len((a.get("answer") or "").split()) for a in answers]
    substantial = sum(1 for n in counts if n >= 18) / max(1, len(counts))
    avg_words = sum(counts) / max(1, len(counts))
    base = round(35 + completed * 35 + substantial * 20 + min(avg_words, 45) / 45 * 10)
    base = _clamp_score(base)
    lang = controller.language
    labels = SCORE_LABELS[lang]
    # ⚑ The baseline only counts completion and answer length. Nothing assessed the
    # language or the content, so every dimension gets the same number and the
    # verdict says so -- no invented per-skill differences, never "Interview ready".
    if lang == "de":
        verdict = "Nur Basiswert – nicht bewertet"
        note = "Basiswert aus Vollständigkeit und Antwortlänge; keine Sprachbewertung."
    else:
        verdict = "Baseline only – not assessed"
        note = "Baseline from completion and answer length; not a language assessment."
    if lang == "de":
        summary = ("Automatisch erzeugte Basisauswertung. Die Antworten wurden vollständig "
                   "gespeichert; eine detaillierte sprachliche Modellbewertung war nicht verfügbar.")
        strengths = ["Das Interview wurde strukturiert bearbeitet.",
                     "Die Antworten wurden klar den Fragen zugeordnet."]
        improvements = ["Antworten mit einem konkreten Beispiel und Ergebnis ergänzen.",
                        "Für jede Antwort eine klare Struktur aus Situation, Handlung und Ergebnis nutzen."]
        fallback_feedback = "Ergänzen Sie ein konkretes Beispiel und ein messbares Ergebnis."
    else:
        summary = ("Automatically generated baseline assessment. All answers were saved; "
                   "a detailed language-model assessment was not available.")
        strengths = ["The interview was completed in a structured way.",
                     "Answers were matched clearly to their questions."]
        improvements = ["Add a specific example and outcome to each answer.",
                        "Use a clear situation, action, and result structure."]
        fallback_feedback = "Add a specific example and a measurable result."
    return {
        "overall_score": base,
        "verdict": verdict,
        "summary": summary,
        "dimensions": [{"key": k, "label": label, "score": base, "note": note}
                       for k, label in labels.items()],
        "strengths": strengths,
        "improvements": improvements,
        "answer_feedback": [
            {"question_id": a["id"], "score": base,
             "feedback": fallback_feedback, "better_answer": "",
             "_answer_text": (a.get("answer") or "")[:1400],
             "_question_text": a.get("text", "")}
            for a in answers
        ],
        "assessment_mode": "fallback",
    }


def assess(llm, controller) -> dict:
    """Ask the already-running local model for one structured final judgment."""
    if not controller.answers:
        return _fallback(controller)
    lang = controller.language
    output_language = "German" if lang == "de" else "English"
    transcript = []
    for a in controller.answers:
        answer = (a.get("answer") or "").strip()[:1400]
        transcript.append(
            f"[{a['id']}] {a['text']}\nCANDIDATE: {answer or '[no answer]'}")
    rubric = """Return ONLY valid JSON with this exact shape:
{
  "overall_score": 0-100,
  "verdict": "short verdict",
  "summary": "2-3 sentence evidence-based summary",
  "dimensions": [
    {"key":"communication","label":"...","score":0-100,"note":"one sentence"},
    {"key":"language","label":"...","score":0-100,"note":"one sentence"},
    {"key":"customer_service","label":"...","score":0-100,"note":"one sentence"},
    {"key":"problem_solving","label":"...","score":0-100,"note":"one sentence"},
    {"key":"professionalism","label":"...","score":0-100,"note":"one sentence"}
  ],
  "strengths": ["three specific strengths"],
  "improvements": ["four prioritized improvements"],
  "answer_feedback": [
    {"question_id":"exact id","score":0-100,"feedback":"specific concise feedback","better_answer":"a concise stronger answer in the interview language"}
  ]
}
Score only evidence in the transcript. Do not invent experience. Penalize vague,
memorized, unsafe, dishonest, or customer-hostile answers. Reward active listening,
empathy, clear verification, realistic next steps, ownership, privacy, de-escalation,
and STAR-style examples. Evaluate grammar, vocabulary, fluency and clarity at the
standard expected for a voice call-center role. Include feedback for every answer."""
    messages = [
        {"role": "system", "content":
         f"You are a strict but constructive call-center hiring assessor. Write all report text in {output_language}."},
        {"role": "user", "content": rubric + "\n\nINTERVIEW TRANSCRIPT\n" + "\n\n".join(transcript)},
    ]
    try:
        parts = []
        for delta, _meta in llm.stream_chat(messages, max_tokens=1900):
            if delta:
                parts.append(delta)
        obj = _extract_json("".join(parts))
    except Exception:
        obj = None
    if not obj:
        return _fallback(controller)

    fallback = _fallback(controller)
    obj["overall_score"] = _clamp_score(obj.get("overall_score"),
                                         fallback["overall_score"])
    dims = []
    by_key = {d.get("key"): d for d in (obj.get("dimensions") or [])
              if isinstance(d, dict)}
    for key, label in SCORE_LABELS[lang].items():
        raw = by_key.get(key, {})
        dims.append({"key": key, "label": label,
                     "score": _clamp_score(raw.get("score"), obj["overall_score"]),
                     "note": str(raw.get("note") or "")[:500]})
    obj["dimensions"] = dims
    # The model did assess: missing fields must not borrow the fallback's
    # "model assessment was not available" wording.
    obj["verdict"] = str(obj.get("verdict") or _score_verdict(obj["overall_score"]))[:120]
    obj["summary"] = str(obj.get("summary") or "")[:1400]
    obj["strengths"] = _text_list(obj.get("strengths"), 5)
    obj["improvements"] = _text_list(obj.get("improvements"), 6)
    feedback = {str(x.get("question_id")): x for x in (obj.get("answer_feedback") or [])
                if isinstance(x, dict) and x.get("question_id")}
    obj["answer_feedback"] = []
    for a in controller.answers:
        raw = feedback.get(a["id"], {})
        obj["answer_feedback"].append({
            "question_id": a["id"],
            "score": _clamp_score(raw.get("score"), obj["overall_score"]),
            "feedback": str(raw.get("feedback") or "")[:700],
            "better_answer": str(raw.get("better_answer") or "")[:1200],
            "_answer_text": (a.get("answer") or "")[:1400],
            "_question_text": a.get("text", ""),
        })
    obj["assessment_mode"] = "local_model"
    return obj


def report_dir() -> Path:
    override = os.environ.get("INTERVIEW_REPORT_DIR")
    desktop_reports = (Path(os.environ.get("OneDrive", Path.home())) /
                       "Desktop" / "Call Center Interview Coach" / "reports")
    candidates = ([Path(override)] if override else []) + [
        desktop_reports,
        config.ROOT / "reports",
        config.DATA_DIR / "reports",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write-test"
            probe.write_text("ok", encoding="ascii")
            probe.unlink()
            return candidate
        except Exception:
            continue
    raise OSError("No writable report directory is available")


def create_pdf(controller, evaluation: dict, session_id: str,
               started_at: str | None = None) -> Path:
    """Render the final report. ReportLab is imported lazily for fast startup."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (HRFlowable, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)

    lang = controller.language
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = report_dir() / f"Interview-Report-{lang.upper()}-{stamp}.pdf"

    font = "Helvetica"
    bold = "Helvetica-Bold"
    arial = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf"
    arial_bold = arial.with_name("arialbd.ttf")
    try:
        if arial.exists():
            pdfmetrics.registerFont(TTFont("InterviewArial", str(arial)))
            font = "InterviewArial"
        if arial_bold.exists():
            pdfmetrics.registerFont(TTFont("InterviewArialBold", str(arial_bold)))
            bold = "InterviewArialBold"
    except Exception:
        pass

    navy = colors.HexColor("#07111F")
    blue = colors.HexColor("#3A7BFF")
    mint = colors.HexColor("#54E1B5")
    slate = colors.HexColor("#526276")
    pale = colors.HexColor("#EEF3F9")
    line = colors.HexColor("#D8E1EC")

    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=18 * mm,
                            leftMargin=18 * mm, topMargin=20 * mm,
                            bottomMargin=18 * mm,
                            title="Call Center Interview Assessment")
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], fontName=bold,
                           fontSize=24, leading=28, textColor=navy, spaceAfter=5 * mm)
    h1 = ParagraphStyle("ReportH1", parent=styles["Heading1"], fontName=bold,
                        fontSize=14, leading=18, textColor=navy,
                        spaceBefore=5 * mm, spaceAfter=3 * mm)
    body = ParagraphStyle("ReportBody", parent=styles["BodyText"], fontName=font,
                          fontSize=9.5, leading=14, textColor=navy)
    small = ParagraphStyle("ReportSmall", parent=body, fontSize=8, leading=11,
                           textColor=slate)
    score_style = ParagraphStyle("Score", parent=body, fontName=bold,
                                 fontSize=28, leading=30, alignment=TA_CENTER,
                                 textColor=blue)

    labels = ({
        "report": "Bewerbungsgespräch - Auswertung",
        "language": "Interviewsprache", "date": "Erstellt am",
        "overall": "Gesamtergebnis", "summary": "Zusammenfassung",
        "strengths": "Stärken", "improvements": "Nächste Schritte",
        "dimensions": "Bewertungsbereiche", "answers": "Fragen und Antworten",
        "answer": "Ihre Antwort", "feedback": "Feedback",
        "better": "Stärkere Formulierung", "question": "Frage",
    } if lang == "de" else {
        "report": "Call Center Interview Assessment",
        "language": "Interview language", "date": "Created",
        "overall": "Overall result", "summary": "Summary",
        "strengths": "Strengths", "improvements": "Next steps",
        "dimensions": "Assessment areas", "answers": "Questions and answers",
        "answer": "Your answer", "feedback": "Feedback",
        "better": "Stronger answer", "question": "Question",
    })

    def p(value, style=body):
        return Paragraph(html.escape(str(value or "")).replace("\n", "<br/>"), style)

    def rich(value, style=body):
        """Markup assembled here only; caller must escape user/model text."""
        return Paragraph(str(value or ""), style)

    def page_decor(canvas, _doc):
        canvas.saveState()
        canvas.setFillColor(navy)
        canvas.rect(0, A4[1] - 9 * mm, A4[0], 9 * mm, fill=1, stroke=0)
        canvas.setFont(font, 7.5)
        canvas.setFillColor(slate)
        canvas.drawString(18 * mm, 9 * mm, f"{session_id}  |  Private local report")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, str(canvas.getPageNumber()))
        canvas.restoreState()

    story = [p(labels["report"], title)]
    meta = Table([
        [p(labels["language"], small), p(controller.meta["label"], body),
         p(labels["date"], small), p(dt.datetime.now().strftime("%Y-%m-%d %H:%M"), body)],
    ], colWidths=[29 * mm, 48 * mm, 24 * mm, 55 * mm])
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), pale), ("BOX", (0, 0), (-1, -1), .5, line),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [meta, Spacer(1, 7 * mm)]

    overall = evaluation.get("overall_score", 0)
    score_box = Table([
        [p(f"{overall}/100", score_style),
         rich(f"<b>{html.escape(labels['overall'])}</b><br/>{html.escape(str(evaluation.get('verdict') or ''))}", body)],
    ], colWidths=[45 * mm, 111 * mm])
    score_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#EAF1FF")),
        ("BACKGROUND", (1, 0), (1, 0), pale), ("BOX", (0, 0), (-1, -1), .7, line),
        ("LINEAFTER", (0, 0), (0, 0), .7, line), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 11), ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story += [score_box, p(labels["summary"], h1), p(evaluation.get("summary"))]

    story.append(p(labels["dimensions"], h1))
    rows = []
    for d in evaluation.get("dimensions") or []:
        rows.append([p(d.get("label"), body), p(f"{d.get('score', 0)}/100", body), p(d.get("note"), small)])
    if rows:
        table = Table(rows, colWidths=[43 * mm, 23 * mm, 90 * mm], repeatRows=0)
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), .4, line), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (0, -1), pale), ("TEXTCOLOR", (1, 0), (1, -1), blue),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(table)

    def bullet_section(name, items, accent):
        data = [[rich(f"<font color='{accent}'>●</font>", body), p(item, body)] for item in items]
        story.append(p(name, h1))
        if data:
            t = Table(data, colWidths=[7 * mm, 149 * mm])
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("TOPPADDING", (0, 0), (-1, -1), 3),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            story.append(t)

    bullet_section(labels["strengths"], evaluation.get("strengths") or [], "#1FA97A")
    bullet_section(labels["improvements"], evaluation.get("improvements") or [], "#3A7BFF")
    story += [PageBreak(), p(labels["answers"], title)]

    feedback = {x.get("question_id"): x for x in evaluation.get("answer_feedback") or []}
    for idx, answer in enumerate(controller.answers, 1):
        fb = feedback.get(answer["id"], {})
        # Keep these as ordinary flowables rather than a single-cell table.  A
        # candidate may give a long spoken answer and ReportLab tables cannot
        # split one tall cell over two pages; flowables can paginate safely.
        story += [
            p(f"{idx:02d}  {labels['question']}  |  {answer['category']}", small),
            p(answer["text"], h1),
            rich(f"<b>{html.escape(labels['answer'])}:</b><br/>{html.escape(answer.get('answer') or '').replace(chr(10), '<br/>')}", body),
        ]
        if fb.get("feedback"):
            story.append(rich(f"<b>{html.escape(labels['feedback'])} ({fb.get('score', 0)}/100):</b><br/>{html.escape(str(fb['feedback']))}", body))
        if fb.get("better_answer"):
            story.append(rich(f"<b>{html.escape(labels['better'])}:</b><br/>{html.escape(str(fb['better_answer']))}", body))
        story += [Spacer(1, 3 * mm), HRFlowable(width="100%", thickness=.6,
                                                color=line), Spacer(1, 4 * mm)]

    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    return path


def finalize(engine, summary: dict, open_pdf: bool = True) -> dict:
    controller = engine.interview
    evaluation = assess(engine.llm, controller)
    path = create_pdf(controller, evaluation,
                      engine.log.session_id if engine.log else "no-session")
    if open_pdf:
        try:
            os.startfile(str(path))  # Windows: show the finished private report.
        except Exception:
            pass
    evaluation.update({
        "kind": "interview_evaluation",
        "language": controller.language,
        "answered": len(controller.answers),
        "total_questions": controller.total,
        "report_path": str(path),
        "summary_metrics": summary,
    })
    return evaluation

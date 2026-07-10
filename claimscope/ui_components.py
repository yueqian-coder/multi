from __future__ import annotations

import json
from html import escape
from urllib.parse import urlparse


AGENT_LABELS = {
    "operationalizer": "Operationalizer",
    "mechanism_analyst": "Mechanism analyst",
    "skeptical_empiricist": "Skeptical empiricist",
    "falsifiability_critic": "Falsifiability critic",
    "scope_critic": "Scope critic",
    "judge": "Judge",
    "heuristic_core_claim_engine": "Baseline extractor",
}


def public_event_rows(result: dict) -> list[dict[str, object]]:
    rows = []
    for event in result.get("events", []):
        role = str(event.get("role", "agent"))
        rows.append(
            {
                "role": AGENT_LABELS.get(role, role.replace("_", " ").title()),
                "status": str(event.get("status", "unknown")),
                "summary": str(event.get("public_summary", "No public summary.")),
                "duration": _duration(event.get("duration_ms", 0)),
                "artifacts": len(event.get("artifacts", {})),
            }
        )
    return rows


def feedback_jsonl(
    input_text: str,
    observed_claim: str,
    expected_claim: str,
    rating: str,
    notes: str,
) -> str:
    return json.dumps(
        {
            "input": input_text,
            "observed_claim": observed_claim,
            "expected_claim": expected_claim,
            "rating": rating,
            "notes": notes,
        },
        ensure_ascii=True,
    ) + "\n"


def provider_host(base_url: str) -> str:
    parsed = urlparse(base_url.strip())
    return parsed.netloc or parsed.path or "not configured"


def confidence_label(value: object) -> str:
    score = float(value or 0)
    if score >= 0.75:
        return "High"
    if score >= 0.5:
        return "Moderate"
    return "Low"


def render_metric(label: str, value: object) -> str:
    return (
        '<div class="metric"><span class="metric-label">'
        + escape(label)
        + '</span><strong>'
        + escape(str(value))
        + "</strong></div>"
    )


def render_candidate(candidate: dict, selected: bool, index: int) -> str:
    score = float(candidate.get("confidence", 0))
    selected_text = "Selected" if selected else "Candidate"
    conditions = ", ".join(candidate.get("conditions", [])) or "No conditions stated"
    radio_class = "radio radio-selected" if selected else "radio"
    return (
        f'<article class="candidate {"candidate-selected" if selected else ""}">'
        f'<div class="candidate-top"><span class="{radio_class}" aria-hidden="true"></span>'
        f'<strong>{selected_text} {index}</strong>'
        f'<span class="muted">Proposed by {escape(str(candidate.get("proposer", "unknown")).replace("_", " "))}</span>'
        f'<b class="candidate-score">{score:.2f}</b></div>'
        f'<div class="candidate-claim">{escape(str(candidate.get("claim", "")))}</div>'
        f'<div class="candidate-meta"><b>Mechanism:</b> {escape(str(candidate.get("method_or_mechanism", "Not specified")))}<br>'
        f'<b>Outcome:</b> {escape(str(candidate.get("expected_effect", "Not specified")))}<br>'
        f'<b>Conditions:</b> {escape(conditions)}</div></article>'
    )


def render_activity(result: dict) -> str:
    rows = public_event_rows(result)
    if not rows:
        rows = [{"role": label, "status": "pending", "summary": "Waiting for arena run.", "duration": "-", "artifacts": 0} for label in ["Operationalizer", "Mechanism analyst", "Skeptical empiricist", "Falsifiability critic", "Scope critic", "Judge"]]
    items = []
    for index, row in enumerate(rows, 1):
        items.append(
            f'<div class="activity-row"><span class="agent-number">{index}</span><div>'
            f'<strong>{escape(str(row["role"]))}</strong>'
            f'<span class="status status-{escape(str(row["status"]))}">{escape(str(row["status"]).title())}</span>'
            f'<p>{escape(str(row["summary"]))}</p></div>'
            f'<span class="muted">{escape(str(row["duration"]))} / {row["artifacts"]} artifacts</span></div>'
        )
    return "".join(items)


def render_critique_summary(result: dict) -> str:
    critiques = result.get("critiques", [])
    if not critiques:
        return '<p class="muted">No critique artifacts returned by this run.</p>'
    items = []
    for critique in critiques:
        scores = critique.get("rubric_scores", {})
        average = sum(float(value) for value in scores.values()) / max(len(scores), 1) / 5
        items.append(f"<div class=\"critique-row\"><strong>{escape(AGENT_LABELS.get(str(critique.get('critic')), str(critique.get('critic', 'Critic'))))}</strong><span>{escape(str(critique.get('public_summary') or critique.get('revision') or 'Public rubric review completed.'))}</span><b>{average:.2f}</b></div>")
    return "".join(items)


def render_discovery_evidence(report: dict) -> str:
    papers = report.get("papers", [])
    if not papers:
        return '<p class="muted">No papers were retrieved.</p>'
    cards = []
    if any(paper.get("is_fixture") for paper in papers):
        cards.append(
            '<div class="warning-box">Synthetic fixture papers are demo data, not research evidence.</div>'
        )
    for paper in papers:
        warning = "Abstract-only source; inspect the full paper before relying on this result." if paper.get("abstract") else "No abstract available."
        fixture_label = " | synthetic fixture" if paper.get("is_fixture") else ""
        cards.append(f"<article class=\"evidence-item\"><div><strong>{escape(str(paper.get('title', 'Untitled')))}</strong> <span class=\"muted\">{escape(str(paper.get('year') or 'n.d.'))}</span></div><p class=\"muted\">{escape(str(paper.get('source', 'unknown')))}{fixture_label} | {escape(', '.join(paper.get('authors', [])[:3]))}</p><p>{escape(str(paper.get('abstract', '')))}</p><small class=\"warning\">{escape(warning)}</small></article>")
    return "".join(cards)


def report_markdown(report: dict) -> str:
    lines = ["# ClaimScope Report", "", f"**Input direction / claim:** {report.get('query', '')}", f"**Normalized claim:** {report.get('claim', '')}", "", "> Evidence labels are heuristic abstract signals, not scientific adjudication.", "", "## Workflow Trace"]
    for index, step in enumerate(report.get("workflow_steps", []), 1):
        lines.append(f"{index}. **{step.get('name', '')}** - `{step.get('status', 'complete')}`")
        lines.append(f"   - Output: {step.get('output', '')}")
    lines.extend(["", "## Retrieved Papers"])
    for paper in report.get("papers", []):
        fixture_label = " [synthetic fixture - not research evidence]" if paper.get("is_fixture") else ""
        lines.append(f"- {paper.get('title', 'Untitled')} ({paper.get('year') or 'n.d.'}){fixture_label}")
        if paper.get("url"):
            lines.append(f"  - URL: {paper['url']}")
    lines.extend(["", "## Assumptions"])
    for assumption in report.get("assumptions", []):
        lines.append(f"- **{assumption.get('text', '')}** - heuristic abstract signal: `{assumption.get('status', 'unknown')}`, risk: `{assumption.get('risk', 'medium')}`")
    lines.extend(["", "## Opportunities"])
    for opportunity in report.get("idea_opportunities", []):
        lines.append(f"- **{opportunity.get('title', '')}** - {opportunity.get('rationale', '')}")
    return "\n".join(lines)


def _duration(milliseconds: object) -> str:
    try:
        seconds = float(milliseconds or 0) / 1000
    except (TypeError, ValueError):
        seconds = 0
    return f"{seconds:.1f}s"

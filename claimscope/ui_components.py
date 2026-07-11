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
AGENT_LABELS_ZH = {
    "operationalizer": "操作化研究员",
    "mechanism_analyst": "机制分析员",
    "skeptical_empiricist": "怀疑主义实证员",
    "falsifiability_critic": "可证伪性评审",
    "scope_critic": "范围边界评审",
    "judge": "裁决者",
    "heuristic_core_claim_engine": "基线抽取器",
}
STATUS_LABELS_ZH = {
    "running": "运行中",
    "complete": "已完成",
    "failed": "失败",
    "degraded": "降级",
    "pending": "等待中",
    "unknown": "未知",
}


def public_event_rows(
    result: dict,
    language: str = "en",
) -> list[dict[str, object]]:
    rows = []
    for event in result.get("events", []):
        role = str(event.get("role", "agent"))
        status = str(event.get("status", "unknown"))
        rows.append(
            {
                "role": _agent_label(role, language),
                "status": status,
                "summary": _public_summary(
                    role,
                    status,
                    str(event.get("public_summary", "No public summary.")),
                    language,
                ),
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


def confidence_label(value: object, language: str = "en") -> str:
    score = float(value or 0)
    if score >= 0.75:
        return "高" if language == "zh" else "High"
    if score >= 0.5:
        return "中" if language == "zh" else "Moderate"
    return "低" if language == "zh" else "Low"


def render_metric(label: str, value: object) -> str:
    return (
        '<div class="metric"><span class="metric-label">'
        + escape(label)
        + '</span><strong>'
        + escape(str(value))
        + "</strong></div>"
    )


def render_candidate(
    candidate: dict,
    selected: bool,
    index: int,
    language: str = "en",
) -> str:
    score = float(candidate.get("confidence", 0))
    selected_text = ("已选" if selected else "候选") if language == "zh" else ("Selected" if selected else "Candidate")
    conditions = ", ".join(candidate.get("conditions", [])) or ("未说明条件" if language == "zh" else "No conditions stated")
    proposed_by = "提出者" if language == "zh" else "Proposed by"
    mechanism = "机制" if language == "zh" else "Mechanism"
    outcome = "预期结果" if language == "zh" else "Outcome"
    conditions_label = "适用条件" if language == "zh" else "Conditions"
    radio_class = "radio radio-selected" if selected else "radio"
    return (
        f'<article class="candidate {"candidate-selected" if selected else ""}">'
        f'<div class="candidate-top"><span class="{radio_class}" aria-hidden="true"></span>'
        f'<strong>{selected_text} {index}</strong>'
        f'<span class="muted">{proposed_by} {_agent_label(str(candidate.get("proposer", "unknown")), language)}</span>'
        f'<b class="candidate-score">{score:.2f}</b></div>'
        f'<div class="candidate-claim">{escape(str(candidate.get("claim", "")))}</div>'
        f'<div class="candidate-meta"><b>{mechanism}:</b> {escape(str(candidate.get("method_or_mechanism", "Not specified")))}<br>'
        f'<b>{outcome}:</b> {escape(str(candidate.get("expected_effect", "Not specified")))}<br>'
        f'<b>{conditions_label}:</b> {escape(conditions)}</div></article>'
    )


def render_activity(result: dict, language: str = "en") -> str:
    rows = public_event_rows(result, language)
    if not rows:
        role_keys = ["operationalizer", "mechanism_analyst", "skeptical_empiricist", "falsifiability_critic", "scope_critic", "judge"]
        rows = [{"role": _agent_label(role, language), "status": "pending", "summary": "等待智能体运行。" if language == "zh" else "Waiting for arena run.", "duration": "-", "artifacts": 0} for role in role_keys]
    items = []
    for index, row in enumerate(rows, 1):
        items.append(
            f'<div class="activity-row"><span class="agent-number">{index}</span><div>'
            f'<strong>{escape(str(row["role"]))}</strong>'
            f'<span class="status status-{escape(str(row["status"]))}">{escape(_status_label(str(row["status"]), language))}</span>'
            f'<p>{escape(str(row["summary"]))}</p></div>'
            f'<span class="muted">{escape(str(row["duration"]))} / {row["artifacts"]} {"项产物" if language == "zh" else "artifacts"}</span></div>'
        )
    return "".join(items)


def render_critique_summary(result: dict, language: str = "en") -> str:
    critiques = result.get("critiques", [])
    if not critiques:
        return '<p class="muted">本次运行没有返回评审产物。</p>' if language == "zh" else '<p class="muted">No critique artifacts returned by this run.</p>'
    items = []
    for critique in critiques:
        scores = critique.get("rubric_scores", {})
        average = sum(float(value) for value in scores.values()) / max(len(scores), 1) / 5
        summary = critique.get('public_summary') or critique.get('revision') or ('公开评分已完成。' if language == 'zh' else 'Public rubric review completed.')
        items.append(f"<div class=\"critique-row\"><strong>{escape(_agent_label(str(critique.get('critic', 'critic')), language))}</strong><span>{escape(str(summary))}</span><b>{average:.2f}</b></div>")
    return "".join(items)


def render_discovery_evidence(report: dict, language: str = "en") -> str:
    papers = report.get("papers", [])
    if not papers:
        return '<p class="muted">没有检索到论文。</p>' if language == "zh" else '<p class="muted">No papers were retrieved.</p>'
    cards = []
    if any(paper.get("is_fixture") for paper in papers):
        cards.append(
            '<div class="warning-box">合成论文仅为演示数据，不属于科研证据。</div>' if language == "zh" else '<div class="warning-box">Synthetic fixture papers are demo data, not research evidence.</div>'
        )
    for paper in papers:
        warning = (("仅使用摘要；采用结论前必须核查全文。" if language == "zh" else "Abstract-only source; inspect the full paper before relying on this result.") if paper.get("abstract") else ("没有可用摘要。" if language == "zh" else "No abstract available."))
        fixture_label = (" | 合成数据" if language == "zh" else " | synthetic fixture") if paper.get("is_fixture") else ""
        cards.append(f"<article class=\"evidence-item\"><div><strong>{escape(str(paper.get('title', 'Untitled')))}</strong> <span class=\"muted\">{escape(str(paper.get('year') or 'n.d.'))}</span></div><p class=\"muted\">{escape(str(paper.get('source', 'unknown')))}{fixture_label} | {escape(', '.join(paper.get('authors', [])[:3]))}</p><p>{escape(str(paper.get('abstract', '')))}</p><small class=\"warning\">{escape(warning)}</small></article>")
    return "".join(cards)


def report_markdown(report: dict, language: str = "en") -> str:
    if language == "zh":
        return _report_markdown_zh(report)
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


def _agent_label(role: str, language: str) -> str:
    labels = AGENT_LABELS_ZH if language == "zh" else AGENT_LABELS
    return labels.get(role, role.replace("_", " ").title())


def _status_label(status: str, language: str) -> str:
    if language == "zh":
        return STATUS_LABELS_ZH.get(status, status)
    return status.title()


def _public_summary(role: str, status: str, summary: str, language: str) -> str:
    if language != "zh":
        return summary
    if status == "running":
        return "正在生成该角色的结构化公开产物。"
    if status == "failed" and role in {
        "operationalizer",
        "mechanism_analyst",
        "skeptical_empiricist",
    }:
        return "该提案角色没有返回可用的公开候选主张。"
    if status == "failed" and role in {
        "falsifiability_critic",
        "scope_critic",
    }:
        return "该评审角色没有返回可用的公开评分。"
    if status == "complete" and role in {
        "operationalizer",
        "mechanism_analyst",
        "skeptical_empiricist",
    }:
        return "已生成一个结构化公开候选主张。"
    if status == "complete" and role in {
        "falsifiability_critic",
        "scope_critic",
    }:
        return "已按公开量表完成候选主张评审。"
    if status == "complete" and role == "judge":
        return "已选择并收敛最终公开核心主张。"
    if status in {"failed", "degraded"} and role == "judge":
        return "裁决角色没有返回可用的最终产物。"
    return summary


def _report_markdown_zh(report: dict) -> str:
    lines = ["# ClaimScope 报告", "", f"**输入研究方向：** {report.get('query', '')}", f"**标准化主张：** {report.get('claim', '')}", "", "> 证据标签是摘要级启发式信号，不是科学裁决。", "", "## 工作流轨迹"]
    for index, step in enumerate(report.get("workflow_steps", []), 1):
        lines.append(f"{index}. **{step.get('name', '')}** - `{step.get('status', 'complete')}`")
        lines.append(f"   - 输出：{step.get('output', '')}")
    lines.extend(["", "## 检索论文"])
    for paper in report.get("papers", []):
        fixture = " [合成数据，不属于科研证据]" if paper.get("is_fixture") else ""
        lines.append(f"- {paper.get('title', '未命名')} ({paper.get('year') or 'n.d.'}){fixture}")
        if paper.get("url"):
            lines.append(f"  - URL: {paper['url']}")
    lines.extend(["", "## 隐含假设"])
    for assumption in report.get("assumptions", []):
        lines.append(f"- **{assumption.get('text', '')}** - 摘要级信号：`{assumption.get('status', 'unknown')}`，风险：`{assumption.get('risk', 'medium')}`")
    lines.extend(["", "## 研究机会"])
    for opportunity in report.get("idea_opportunities", []):
        lines.append(f"- **{opportunity.get('title', '')}** - {opportunity.get('rationale', '')}")
    return "\n".join(lines)

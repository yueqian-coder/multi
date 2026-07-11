from __future__ import annotations

import os
import json
from html import escape
from pathlib import Path

import streamlit as st

from claimscope.llm import OpenAICompatibleClient
from claimscope.service import ClaimScopeService
from claimscope.ui_components import (
    confidence_label,
    feedback_jsonl,
    provider_host,
    render_activity,
    render_candidate,
    render_critique_summary,
    render_discovery_evidence,
    report_markdown,
)


DEFAULT_DIRECTION = "RAG can reliably reduce hallucination in LLM-generated answers"
DEFAULT_BASE_URL = "https://sub.qianyueapi.com/v1"
DEFAULT_MODEL = "gpt-5.4-mini"
ROLE_ORDER = [
    "operationalizer",
    "mechanism_analyst",
    "skeptical_empiricist",
    "falsifiability_critic",
    "scope_critic",
    "judge",
]
PROPOSER_ROLES = ROLE_ORDER[:3]
CRITIC_ROLES = ROLE_ORDER[3:5]

TEXT = {
    "en": {
        "tagline": "Research claim audit",
        "arena": "Core Claim Arena",
        "discovery": "Full Discovery",
        "provider": "Provider",
        "provider_ready": "Provider configured",
        "provider_required": "Provider required",
        "api_key": "API key",
        "base_url": "Base URL",
        "model": "Model",
        "remote_consent": "Allow research directions to be sent to this provider",
        "session_only": "Session-only. The key is never exported or logged.",
        "provider_host": "Provider host",
        "provider_note": "Agent runs send the research direction to the configured provider.",
        "direction": "Research direction",
        "direction_help": "Describe the fuzzy direction you want to stress-test.",
        "direction_example": "Example: retrieval quality, hallucination, and the conditions under which the effect should hold.",
        "core_scope": "This run stops after the Core Claim arena.",
        "retrieval_consent": "Allow generated queries to be sent to arXiv and Semantic Scholar",
        "retrieval_note": "Discovery uses live academic APIs. Queries and source metadata may leave this device.",
        "max_papers": "Maximum papers",
        "run_arena": "Run Arena",
        "run_discovery": "Run Discovery",
        "configure_first": "Configure a Provider and approve remote processing before running.",
        "approve_retrieval": "Approve academic query sharing before running Discovery.",
        "enter_direction": "Enter a research direction first.",
        "run_failed": "The real-agent run did not complete. Check the Provider, model availability, and network, then retry.",
        "discovery_failed": "Discovery did not complete. No synthetic result was substituted.",
        "empty_core": "No claim analyzed",
        "empty_core_detail": "Configure a Provider, enter a direction, and run the arena.",
        "empty_discovery": "No discovery report",
        "empty_discovery_detail": "Run live Discovery to build an assumption and evidence map.",
        "selected_artifact": "Selected artifact",
        "core_claim": "Core Claim",
        "candidate_score": "Candidate score",
        "structural": "Structural completeness",
        "mode": "Mode",
        "falsification": "Falsification test",
        "not_specified": "Not specified",
        "open_slots": "Open claim slots",
        "agent_activity": "Agent activity",
        "candidate_comparison": "Candidate comparison",
        "critique_summary": "Critique summary",
        "public_trace": "Public execution trace",
        "public_trace_note": "Live status and structured public artifacts only. Private chain-of-thought is not collected or displayed.",
        "proposer_stage": "Proposer",
        "critic_stage": "Critic",
        "judge_stage": "Judge",
        "public_artifacts": "Agent public artifacts",
        "public_input": "Public input summary",
        "operationalizer_input": "Operationalize variables, baseline, metric, and controlled comparison.",
        "mechanism_input": "Specify mechanism, target, expected effect, and boundary conditions.",
        "skeptic_input": "Produce the narrowest claim that an adverse result could falsify.",
        "critic_input": "Research direction plus all public candidates; score the six-item rubric and suggest revisions.",
        "judge_input": "All public candidates and critiques; select and refine one falsifiable claim.",
        "candidate": "Candidate",
        "no_candidate": "No valid candidate was returned.",
        "rubric_scores": "Rubric scores",
        "reason_codes": "Reason codes",
        "revision": "Revision suggestion",
        "no_critique": "No valid critique was returned.",
        "selected_candidate": "Selected candidate",
        "unresolved": "Unresolved ambiguities",
        "field": "Field",
        "value": "Value",
        "claim": "Claim",
        "mechanism": "Method or mechanism",
        "target": "Target or task",
        "effect": "Expected effect",
        "conditions": "Boundary conditions",
        "missing": "Missing information",
        "confidence": "Structural score",
        "candidate_id": "Candidate ID",
        "score": "Score",
        "anchor": "Anchor artifact",
        "selected_core": "Selected Core Claim",
        "overview": "Overview",
        "ledger": "Assumption Ledger",
        "evidence": "Evidence",
        "opportunities": "Opportunities",
        "trace": "Trace",
        "export": "Export",
        "assumptions": "Testable assumptions",
        "papers": "Retrieved papers",
        "opportunity_slots": "Opportunity slots",
        "signal": "Signal",
        "risk": "Risk",
        "evidence_count": "Evidence",
        "next": "Next",
        "download_report": "Download Markdown report",
        "evaluate": "Evaluate observed claim",
        "expected": "Expected claim",
        "rating": "Rating",
        "notes": "Notes",
        "download_feedback": "Download feedback JSONL",
    },
    "zh": {
        "tagline": "研究主张审计",
        "arena": "核心主张竞技场",
        "discovery": "完整研究发现",
        "provider": "模型配置",
        "provider_ready": "Provider 已配置",
        "provider_required": "需要配置 Provider",
        "api_key": "API 密钥",
        "base_url": "接口地址",
        "model": "模型",
        "remote_consent": "允许将研究方向发送给此 Provider",
        "session_only": "密钥仅保留在当前会话中，不会导出或写入日志。",
        "provider_host": "接口主机",
        "provider_note": "智能体运行时会把研究方向发送给已配置的 Provider。",
        "direction": "研究方向",
        "direction_help": "输入一个需要被收敛和质疑的模糊研究方向。",
        "direction_example": "示例：检索质量、幻觉，以及该效果成立所需的边界条件。",
        "core_scope": "本次运行仅执行核心主张竞技场，不启动后续检索。",
        "retrieval_consent": "允许将生成的检索式发送给 arXiv 和 Semantic Scholar",
        "retrieval_note": "完整研究发现会调用真实学术接口，检索式与来源元数据可能离开本机。",
        "max_papers": "最多检索论文数",
        "run_arena": "运行竞技场",
        "run_discovery": "运行完整流程",
        "configure_first": "请先配置 Provider，并允许远程处理研究方向。",
        "approve_retrieval": "请先允许发送学术检索式。",
        "enter_direction": "请先输入研究方向。",
        "run_failed": "真实智能体运行未完成。请检查 Provider、模型可用性与网络后重试。",
        "discovery_failed": "完整研究发现未完成，系统没有用合成结果替代。",
        "empty_core": "尚未分析主张",
        "empty_core_detail": "配置 Provider、输入研究方向，然后运行竞技场。",
        "empty_discovery": "尚无研究发现报告",
        "empty_discovery_detail": "运行真实检索流程，构建假设与证据地图。",
        "selected_artifact": "最终产物",
        "core_claim": "核心主张",
        "candidate_score": "候选结构分",
        "structural": "结构完整度",
        "mode": "运行模式",
        "falsification": "证伪测试",
        "not_specified": "未说明",
        "open_slots": "待补全主张字段",
        "agent_activity": "智能体活动",
        "candidate_comparison": "候选主张对比",
        "critique_summary": "评审摘要",
        "public_trace": "公开执行轨迹",
        "public_trace_note": "仅显示实时状态和结构化公开产物，不收集或展示模型私有思维链。",
        "proposer_stage": "提案阶段",
        "critic_stage": "评审阶段",
        "judge_stage": "裁决阶段",
        "public_artifacts": "各智能体公开产物",
        "public_input": "公开输入摘要",
        "operationalizer_input": "操作化变量、比较基线、评价指标与受控对照。",
        "mechanism_input": "明确机制、研究对象、预期效果与边界条件。",
        "skeptic_input": "提出能被不利结果直接证伪的最窄主张。",
        "critic_input": "研究方向与全部公开候选；按六项量表评分并给出修改建议。",
        "judge_input": "全部公开候选与评审产物；选择并收敛一个可证伪主张。",
        "candidate": "候选主张",
        "no_candidate": "该角色没有返回有效候选。",
        "rubric_scores": "量表评分",
        "reason_codes": "原因代码",
        "revision": "修改建议",
        "no_critique": "该角色没有返回有效评审。",
        "selected_candidate": "入选候选",
        "unresolved": "未解决歧义",
        "field": "字段",
        "value": "内容",
        "claim": "主张",
        "mechanism": "方法或机制",
        "target": "对象或任务",
        "effect": "预期效果",
        "conditions": "边界条件",
        "missing": "缺失信息",
        "confidence": "结构分",
        "candidate_id": "候选编号",
        "score": "得分",
        "anchor": "锚点产物",
        "selected_core": "最终核心主张",
        "overview": "概览",
        "ledger": "假设台账",
        "evidence": "证据",
        "opportunities": "研究机会",
        "trace": "执行轨迹",
        "export": "导出",
        "assumptions": "可检验假设",
        "papers": "检索论文",
        "opportunity_slots": "研究机会点",
        "signal": "信号",
        "risk": "风险",
        "evidence_count": "证据数",
        "next": "下一步",
        "download_report": "下载 Markdown 报告",
        "evaluate": "评价当前主张",
        "expected": "期望主张",
        "rating": "评价",
        "notes": "备注",
        "download_feedback": "下载反馈 JSONL",
    },
}

WORKFLOW_LABELS = {
    "en": ["Direction", "Core claim", "Variants", "Assumptions", "Queries", "Evidence", "Opportunities"],
    "zh": ["研究方向", "核心主张", "主张变体", "隐含假设", "证据检索式", "证据卡片", "研究机会"],
}


def t(language: str, key: str) -> str:
    return TEXT.get(language, TEXT["en"]).get(key, key)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#182026; --muted:#637078; --line:#d5dadd; --surface:#ffffff; --canvas:#f6f7f7; --accent:#176b68; --navy:#243e4a; --amber:#8a5a00; --red:#a63b32; --soft:#eef1f1; }
        .stApp { background:var(--canvas); color:var(--ink); font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
        .block-container { max-width:1480px; padding:4.5rem 1.1rem 2.5rem; }
        [data-testid="stHeader"] { background:rgba(255,255,255,.96); border-bottom:1px solid #e4e7e8; }
        h1,h2,h3,h4,p { letter-spacing:0; }
        .brand { color:var(--navy); font-size:24px; line-height:1.05; font-weight:760; white-space:nowrap; }
        .brand-kicker { color:var(--muted); font-size:11px; margin-top:5px; }
        .provider-status { color:var(--muted); font-size:12px; display:flex; align-items:center; justify-content:flex-end; gap:7px; white-space:nowrap; }
        .provider-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--accent); flex:0 0 auto; }
        .provider-dot-required { background:var(--amber); }
        .work-surface { background:var(--surface); border:1px solid var(--line); border-radius:3px; padding:16px; margin-bottom:12px; }
        .section-head { color:var(--navy); font-size:15px; font-weight:720; margin-bottom:4px; }
        .section-kicker { color:var(--muted); font-size:11px; margin-bottom:2px; }
        .muted { color:var(--muted); font-size:12px; }
        .selected-claim { color:var(--ink); font-size:19px; line-height:1.45; font-weight:680; margin:8px 0 16px; max-width:78ch; overflow-wrap:anywhere; }
        .metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border-top:1px solid var(--line); }
        .metric { min-height:70px; border-right:1px solid var(--line); padding:12px; }.metric:first-child { padding-left:0; }.metric:last-child { border:0; }
        .metric-label { display:block; color:var(--muted); font-size:11px; }.metric strong { display:block; font-size:17px; line-height:1.3; margin-top:4px; overflow-wrap:anywhere; }.metric:nth-child(3) strong { font-size:12px; line-height:1.45; font-weight:620; }.metric small { display:block; color:var(--accent); margin-top:3px; }
        .candidate { border-top:1px solid var(--line); padding:12px 2px; margin:0; }.candidate:first-of-type { border-top:0; }.candidate-selected { border-left:3px solid var(--accent); padding-left:11px; }
        .candidate-top { display:flex; align-items:center; gap:9px; font-size:13px; }.radio { display:inline-block; width:13px; height:13px; border:1px solid var(--accent); border-radius:50%; flex:0 0 auto; }.radio-selected { background:var(--accent); box-shadow:inset 0 0 0 3px #fff; }.candidate-score { margin-left:auto; color:var(--accent); }.candidate-claim { font-weight:650; margin:6px 0; line-height:1.4; }.candidate-meta { color:#465159; font-size:12px; line-height:1.55; }
        .trace-head { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; padding-bottom:10px; border-bottom:1px solid var(--line); }.trace-note { color:var(--muted); font-size:11px; max-width:62ch; text-align:right; }
        .stage-line { display:grid; grid-template-columns:130px 1fr; border-bottom:1px solid var(--line); padding:8px 0; }.stage-line:last-child { border-bottom:0; }.stage-line strong { font-size:12px; }.stage-line span { color:var(--muted); font-size:12px; }
        .activity-row { display:grid; grid-template-columns:28px 1fr auto; gap:10px; align-items:start; border-top:1px solid var(--line); padding:10px 0; }.agent-number { border:1px solid var(--line); color:var(--navy); border-radius:2px; width:23px; height:23px; text-align:center; padding-top:2px; font-size:11px; font-weight:700; }.activity-row p { margin:3px 0 0; color:var(--muted); font-size:12px; }.status { margin-left:10px; color:var(--accent); font-size:11px; font-weight:650; }.status-failed { color:var(--red); }.status-pending { color:var(--muted); }.status-degraded { color:var(--amber); }
        .critique-row { display:grid; grid-template-columns:160px 1fr 45px; gap:10px; border-top:1px solid var(--line); padding:10px 0; font-size:13px; }.critique-row b { color:var(--amber); }
        .warning-box { border-left:3px solid var(--amber); padding:9px 11px; background:#fffaf0; color:#6d4a09; font-size:13px; margin:8px 0; }
        .empty-state { min-height:210px; display:grid; align-content:center; border:1px dashed #b9c1c4; background:rgba(255,255,255,.58); padding:30px; text-align:center; }.empty-state strong { color:var(--navy); font-size:15px; }.empty-state p { color:var(--muted); font-size:13px; margin:6px auto 0; max-width:46ch; }
        .workflow-wrap { overflow-x:auto; margin:0 0 12px; }.workflow-rail { min-width:760px; display:grid; grid-template-columns:repeat(7,minmax(100px,1fr)); border-top:1px solid var(--line); border-bottom:1px solid var(--line); background:var(--surface); }.workflow-step { position:relative; min-height:62px; border-right:1px solid var(--line); padding:10px 8px 8px 34px; }.workflow-step:last-child { border:0; }.workflow-step-num { position:absolute; left:9px; top:11px; color:var(--muted); font-size:11px; }.workflow-step strong { display:block; color:var(--ink); font-size:11px; line-height:1.3; }.workflow-step span { color:var(--accent); font-size:10px; }
        .overview-metrics { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); border-top:1px solid var(--line); border-bottom:1px solid var(--line); margin:10px 0 14px; }.overview-metric { min-height:78px; border-right:1px solid var(--line); padding:13px; }.overview-metric:last-child { border:0; }.overview-metric b { color:var(--navy); display:block; font-size:22px; line-height:1; margin-bottom:6px; }.overview-metric span { color:var(--muted); font-size:11px; }
        .artifact, .evidence-item { border-top:1px solid var(--line); padding:12px 0; }.artifact:first-child, .evidence-item:first-child { border-top:0; }.artifact-head { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }.artifact-title { font-size:13px; line-height:1.45; }.badge-row { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:8px; }.badge { color:var(--muted); font-size:10px; white-space:nowrap; }.artifact p,.evidence-item p { font-size:13px; line-height:1.5; margin:8px 0 0; }.artifact-next { color:var(--muted); }.warning { color:var(--amber); }
        [data-testid="stVerticalBlockBorderWrapper"] { border-radius:3px !important; box-shadow:none !important; }
        [data-testid="stButton"] button, [data-testid="stDownloadButton"] button { min-height:44px; border-radius:3px; font-weight:650; }
        [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input { border-radius:3px; }
        [data-testid="stPopover"] button { white-space:nowrap; }
        [data-testid="stExpander"] { border-radius:2px; box-shadow:none; }
        button:focus-visible, input:focus-visible, textarea:focus-visible, [role="tab"]:focus-visible { outline:3px solid rgba(23,107,104,.24) !important; outline-offset:2px; }
        [role="tab"] { min-height:42px; }
        @media (max-width:1050px) { .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); }.metric { border-bottom:1px solid var(--line); }.metric:nth-child(2) { border-right:0; }.metric:nth-child(n+3) { border-bottom:0; } }
        @media (max-width:760px) { .block-container { padding:4rem 0.65rem 2rem; }.brand { font-size:21px; }.provider-status { justify-content:flex-start; white-space:normal; }.metrics { grid-template-columns:1fr 1fr; }.metric { min-height:76px; padding:9px; }.metric:first-child { padding-left:9px; }.overview-metrics { grid-template-columns:1fr; }.overview-metric { border-right:0; border-bottom:1px solid var(--line); }.overview-metric:last-child { border-bottom:0; }.activity-row { grid-template-columns:26px 1fr; }.activity-row > .muted { grid-column:2; }.critique-row { grid-template-columns:1fr auto; }.critique-row span { grid-column:1 / -1; order:3; }.trace-head,.artifact-head { display:block; }.trace-note { text-align:left; margin-top:5px; }.badge-row { justify-content:flex-start; margin-top:6px; }.stage-line { grid-template-columns:105px 1fr; } }
        @media (prefers-reduced-motion:reduce) { *,*::before,*::after { scroll-behavior:auto !important; transition-duration:.01ms !important; animation-duration:.01ms !important; animation-iteration-count:1 !important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def session_defaults() -> None:
    snapshot = load_result_snapshot()
    defaults = {
        "core_claim_result": snapshot,
        "discovery_result": None,
        "direction": str(snapshot.get("direction", DEFAULT_DIRECTION)) if snapshot else DEFAULT_DIRECTION,
        "language": "zh",
        "mode": "arena",
        "provider_key": "",
        "provider_base": os.getenv("OPENAI_BASE_URL", ""),
        "provider_model": os.getenv("MODEL_NAME", DEFAULT_MODEL),
        "provider_allow_remote": os.getenv("CLAIMSCOPE_TRUST_ENV_PROVIDER") == "1",
        "retrieval_consent": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_result_snapshot() -> dict | None:
    path = os.getenv("CLAIMSCOPE_RESULT_SNAPSHOT", "").strip()
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def provider_ready() -> bool:
    session_key = str(st.session_state.get("provider_key", "")).strip()
    environment_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
    base_url = str(st.session_state.get("provider_base", "")).strip()
    approved = bool(st.session_state.get("provider_allow_remote"))
    api_key = session_key or environment_key
    return provider_configuration_ready(api_key, base_url, approved)


def provider_configuration_ready(
    api_key: str,
    base_url: str,
    approved: bool,
) -> bool:
    if not (base_url.strip() and api_key.strip() and approved):
        return False
    try:
        OpenAICompatibleClient(
            api_key=api_key,
            base_url=base_url,
            model=DEFAULT_MODEL,
        )
    except ValueError:
        return False
    return True


def build_service() -> ClaimScopeService:
    session_key = str(st.session_state.get("provider_key", "")).strip()
    api_key = session_key or str(os.getenv("OPENAI_API_KEY", "")).strip()
    base_url = str(st.session_state.get("provider_base", "")).strip()
    model = str(st.session_state.get("provider_model", "")).strip() or DEFAULT_MODEL
    client = OpenAICompatibleClient(api_key=api_key, base_url=base_url, model=model)
    return ClaimScopeService(llm_client=client)


def render_header(language: str) -> tuple[str, str]:
    ready = provider_ready()
    with st.container(border=True):
        brand_column, language_column, mode_column, provider_column = st.columns(
            [0.75, 0.55, 1.3, 1.15], vertical_alignment="center"
        )
        with brand_column:
            st.markdown(
                f'<div class="brand">ClaimScope</div><div class="brand-kicker">{escape(t(language, "tagline"))}</div>',
                unsafe_allow_html=True,
            )
        with language_column:
            if hasattr(st, "segmented_control"):
                st.segmented_control(
                    "Language",
                    ["zh", "en"],
                    format_func=lambda value: "中文" if value == "zh" else "EN",
                    key="language",
                    label_visibility="collapsed",
                )
            else:
                st.radio(
                    "Language",
                    ["zh", "en"],
                    format_func=lambda value: "中文" if value == "zh" else "EN",
                    key="language",
                    horizontal=True,
                    label_visibility="collapsed",
                )
        with mode_column:
            mode_labels = {"arena": t(language, "arena"), "discovery": t(language, "discovery")}
            if hasattr(st, "segmented_control"):
                st.segmented_control(
                    "Mode",
                    ["arena", "discovery"],
                    format_func=lambda value: mode_labels[value],
                    key="mode",
                    label_visibility="collapsed",
                )
            else:
                st.radio(
                    "Mode",
                    ["arena", "discovery"],
                    format_func=lambda value: mode_labels[value],
                    key="mode",
                    horizontal=True,
                    label_visibility="collapsed",
                )
        with provider_column:
            status_column, settings_column = st.columns([1.8, 1], vertical_alignment="center")
            with status_column:
                dot_class = "provider-dot" if ready else "provider-dot provider-dot-required"
                state = t(language, "provider_ready" if ready else "provider_required")
                st.markdown(
                    f'<div class="provider-status"><span class="{dot_class}" aria-hidden="true"></span><span>{escape(state)}</span></div>',
                    unsafe_allow_html=True,
                )
            with settings_column:
                with st.popover(t(language, "provider"), use_container_width=True):
                    st.text_input(t(language, "api_key"), type="password", key="provider_key", help=t(language, "session_only"))
                    st.text_input(
                        t(language, "base_url"),
                        key="provider_base",
                        placeholder=DEFAULT_BASE_URL,
                    )
                    st.text_input(t(language, "model"), key="provider_model")
                    st.checkbox(t(language, "remote_consent"), key="provider_allow_remote")
                    st.caption(f'{t(language, "provider_host")}: {provider_host(str(st.session_state.get("provider_base", "")))}')
                    st.caption(t(language, "provider_note"))
    return str(st.session_state.get("mode", "arena")), str(st.session_state.get("language", language))


def render_research_controls(mode: str, language: str) -> tuple[str, int, bool]:
    ready = provider_ready()
    with st.container(border=True):
        st.markdown(
            f'<div class="section-head">{escape(t(language, "direction"))}</div><p class="muted">{escape(t(language, "direction_help"))}</p>',
            unsafe_allow_html=True,
        )
        direction = st.text_area(t(language, "direction"), key="direction", height=142, label_visibility="collapsed")
        st.caption(t(language, "direction_example"))
        limit = 12
        retrieval_approved = True
        if mode == "arena":
            st.caption(t(language, "core_scope"))
        else:
            st.info(t(language, "retrieval_note"))
            retrieval_approved = st.checkbox(t(language, "retrieval_consent"), key="retrieval_consent")
            limit = st.slider(t(language, "max_papers"), 5, 30, 12)
        disabled = not ready or not retrieval_approved
        run = st.button(
            t(language, "run_arena" if mode == "arena" else "run_discovery"),
            type="primary",
            use_container_width=True,
            disabled=disabled,
        )
        if not ready:
            st.caption(t(language, "configure_first"))
        elif not retrieval_approved:
            st.caption(t(language, "approve_retrieval"))
    return direction, limit, run


def empty_state(title: str, detail: str) -> str:
    return f'<div class="empty-state"><strong>{escape(title)}</strong><p>{escape(detail)}</p></div>'


def render_live_progress(events: dict[str, dict], language: str) -> str:
    ordered = [events[role] for role in ROLE_ORDER if role in events]
    status_by_role = {role: str(events.get(role, {}).get("status", "pending")) for role in ROLE_ORDER}

    def stage_status(roles: list[str]) -> str:
        states = [status_by_role[role] for role in roles]
        if any(state == "running" for state in states):
            return "running" if language == "en" else "运行中"
        if all(state == "complete" for state in states):
            return "complete" if language == "en" else "已完成"
        if any(state in {"failed", "degraded"} for state in states):
            return "failed" if language == "en" else "未完成"
        return "pending" if language == "en" else "等待中"

    stage_rows = [
        (t(language, "proposer_stage"), stage_status(PROPOSER_ROLES)),
        (t(language, "critic_stage"), stage_status(CRITIC_ROLES)),
        (t(language, "judge_stage"), stage_status(["judge"])),
    ]
    stages = "".join(
        f'<div class="stage-line"><strong>{escape(label)}</strong><span>{escape(status)}</span></div>'
        for label, status in stage_rows
    )
    return (
        '<section class="work-surface"><div class="trace-head">'
        f'<div class="section-head">{escape(t(language, "public_trace"))}</div>'
        f'<div class="trace-note">{escape(t(language, "public_trace_note"))}</div></div>'
        + stages
        + render_activity({"events": ordered}, language=language)
        + "</section>"
    )


def render_core_claim(result: dict | None, language: str) -> None:
    if not result:
        st.markdown(empty_state(t(language, "empty_core"), t(language, "empty_core_detail")), unsafe_allow_html=True)
        return
    selected = result.get("selected_candidate") or {}
    confidence = float(selected.get("confidence", 0))
    open_slots = result.get("unresolved_ambiguities", [])
    metrics = [
        f'<div class="metric"><span class="metric-label">{escape(t(language, "candidate_score"))}</span><strong>{confidence:.2f}</strong><small>{escape(t(language, "structural"))}: {escape(confidence_label(confidence, language))}</small></div>',
        f'<div class="metric"><span class="metric-label">{escape(t(language, "mode"))}</span><strong>multi-agent</strong></div>',
        f'<div class="metric"><span class="metric-label">{escape(t(language, "falsification"))}</span><strong>{escape(str(selected.get("falsification_test", t(language, "not_specified"))))}</strong></div>',
        f'<div class="metric"><span class="metric-label">{escape(t(language, "open_slots"))}</span><strong>{len(open_slots)}</strong></div>',
    ]
    st.markdown(
        f'<section class="work-surface"><div class="section-kicker">{escape(t(language, "selected_artifact"))}</div><div class="section-head">{escape(t(language, "core_claim"))}</div><div class="selected-claim">{escape(str(result.get("selected_claim", "")))}</div><div class="metrics">{"".join(metrics)}</div></section>',
        unsafe_allow_html=True,
    )
    if open_slots:
        st.markdown(
            f'<div class="warning-box"><strong>{escape(t(language, "open_slots"))}:</strong> {escape("; ".join(str(item) for item in open_slots))}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<section class="work-surface"><div class="section-head">{escape(t(language, "agent_activity"))}</div>{render_activity(result, language=language)}</section>',
        unsafe_allow_html=True,
    )
    candidates = result.get("candidates", [])
    candidate_markup = "".join(
        render_candidate(
            candidate,
            candidate is selected or candidate.get("proposer") == selected.get("proposer"),
            index,
            language=language,
        )
        for index, candidate in enumerate(candidates, 1)
    )
    st.markdown(
        f'<section class="work-surface"><div class="section-head">{escape(t(language, "candidate_comparison"))}</div>{candidate_markup}</section>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<section class="work-surface"><div class="section-head">{escape(t(language, "critique_summary"))}</div>{render_critique_summary(result, language=language)}</section>',
        unsafe_allow_html=True,
    )
    render_agent_inspector(result, language)


def candidate_rows(candidate: dict, language: str) -> list[dict[str, object]]:
    values = [
        ("claim", candidate.get("claim", "")),
        ("mechanism", candidate.get("method_or_mechanism", "")),
        ("target", candidate.get("target_or_task", "")),
        ("effect", candidate.get("expected_effect", "")),
        ("conditions", "; ".join(candidate.get("conditions", []))),
        ("falsification", candidate.get("falsification_test", "")),
        ("missing", "; ".join(candidate.get("missing_information", []))),
        ("confidence", candidate.get("confidence", 0)),
    ]
    return [{t(language, "field"): t(language, key), t(language, "value"): value or t(language, "not_specified")} for key, value in values]


def render_agent_inspector(result: dict, language: str) -> None:
    candidates = {str(item.get("proposer", "")): item for item in result.get("candidates", [])}
    critiques = result.get("critiques", [])
    selected = result.get("selected_candidate") or {}
    st.markdown(f'<div class="section-head">{escape(t(language, "public_artifacts"))}</div>', unsafe_allow_html=True)
    for role in PROPOSER_ROLES:
        with st.expander(agent_name(role, language)):
            mandate_key = {
                "operationalizer": "operationalizer_input",
                "mechanism_analyst": "mechanism_input",
                "skeptical_empiricist": "skeptic_input",
            }[role]
            st.caption(
                f'{t(language, "public_input")}: {result.get("direction", "")} | '
                f'{t(language, mandate_key)}'
            )
            candidate = candidates.get(role)
            if candidate:
                st.table(candidate_rows(candidate, language))
            else:
                st.warning(t(language, "no_candidate"))
    for role in CRITIC_ROLES:
        with st.expander(agent_name(role, language)):
            st.caption(
                f'{t(language, "public_input")}: {t(language, "critic_input")} '
                f'({len(candidates)} {t(language, "candidate").lower()})'
            )
            role_critiques = [item for item in critiques if item.get("critic") == role]
            if not role_critiques:
                st.warning(t(language, "no_critique"))
            for critique in role_critiques:
                st.markdown(f'**{t(language, "candidate_id")}:** `{critique.get("candidate_id", "")}`')
                rubric_scores = critique.get("rubric_scores", {})
                st.table([{t(language, "field"): localized_term(key, language), t(language, "score"): score} for key, score in rubric_scores.items()])
                reason_codes = critique.get("reason_codes", [])
                revision = critique.get("revision", "")
                st.markdown(f'**{t(language, "reason_codes")}:** {escape(", ".join(str(item) for item in reason_codes) or t(language, "not_specified"))}')
                st.markdown(f'**{t(language, "revision")}:** {escape(str(revision or t(language, "not_specified")))}')
                st.divider()
    with st.expander(agent_name("judge", language)):
        st.caption(
            f'{t(language, "public_input")}: {t(language, "judge_input")} '
            f'({len(candidates)} / {len(critiques)})'
        )
        judge_rows = [
            {t(language, "field"): t(language, "selected_candidate"), t(language, "value"): selected.get("proposer", "")},
            {t(language, "field"): t(language, "claim"), t(language, "value"): result.get("selected_claim", "")},
            {t(language, "field"): t(language, "falsification"), t(language, "value"): selected.get("falsification_test", "")},
            {t(language, "field"): t(language, "unresolved"), t(language, "value"): "; ".join(result.get("unresolved_ambiguities", [])) or t(language, "not_specified")},
        ]
        st.table(judge_rows)


def agent_name(role: str, language: str) -> str:
    labels = {
        "en": {
            "operationalizer": "Operationalizer",
            "mechanism_analyst": "Mechanism analyst",
            "skeptical_empiricist": "Skeptical empiricist",
            "falsifiability_critic": "Falsifiability critic",
            "scope_critic": "Scope critic",
            "judge": "Judge",
        },
        "zh": {
            "operationalizer": "操作化研究员",
            "mechanism_analyst": "机制分析员",
            "skeptical_empiricist": "怀疑主义实证员",
            "falsifiability_critic": "可证伪性评审",
            "scope_critic": "范围边界评审",
            "judge": "裁决者",
        },
    }
    return labels[language].get(role, role.replace("_", " ").title())


def localized_term(value: str, language: str) -> str:
    if language != "zh":
        return value.replace("_", " ")
    labels = {
        "specificity": "具体性",
        "falsifiability": "可证伪性",
        "mechanism": "机制清晰度",
        "scope": "范围边界",
        "measurability": "可测量性",
        "risk_awareness": "风险意识",
        "complete": "已完成",
        "failed": "失败",
        "degraded": "部分完成",
        "skipped": "已跳过",
        "unknown": "未知",
        "supported": "支持信号",
        "unsupported": "反向信号",
        "mixed": "混合信号",
        "high": "高",
        "medium": "中",
        "low": "低",
        "assumption_gap": "假设缺口",
        "negative_evidence_experiment": "负面证据实验",
    }
    return labels.get(value, value.replace("_", " "))


def workflow_rail(report: dict, language: str) -> str:
    steps = report.get("workflow_steps", [])
    items = []
    for index, label in enumerate(WORKFLOW_LABELS[language], 1):
        status = str(steps[index - 1].get("status", "complete")) if index <= len(steps) else "complete"
        items.append(
            f'<div class="workflow-step"><span class="workflow-step-num">{index:02}</span><strong>{escape(label)}</strong><span>{escape(localized_term(status, language))}</span></div>'
        )
    return '<div class="workflow-wrap"><div class="workflow-rail">' + "".join(items) + "</div></div>"


def render_discovery(report: dict | None, language: str) -> None:
    if not report:
        st.markdown(empty_state(t(language, "empty_discovery"), t(language, "empty_discovery_detail")), unsafe_allow_html=True)
        return
    st.markdown(
        f'<section class="work-surface"><div class="section-kicker">{escape(t(language, "anchor"))}</div><div class="section-head">{escape(t(language, "selected_core"))}</div><div class="selected-claim">{escape(str(report.get("claim", "")))}</div></section>',
        unsafe_allow_html=True,
    )
    st.markdown(workflow_rail(report, language), unsafe_allow_html=True)
    overview, ledger, evidence, opportunities, trace, export = st.tabs(
        [t(language, key) for key in ["overview", "ledger", "evidence", "opportunities", "trace", "export"]]
    )
    with overview:
        st.markdown(
            '<div class="overview-metrics">'
            f'<div class="overview-metric"><b>{len(report.get("assumptions", []))}</b><span>{escape(t(language, "assumptions"))}</span></div>'
            f'<div class="overview-metric"><b>{len(report.get("papers", []))}</b><span>{escape(t(language, "papers"))}</span></div>'
            f'<div class="overview-metric"><b>{len(report.get("idea_opportunities", []))}</b><span>{escape(t(language, "opportunity_slots"))}</span></div></div>',
            unsafe_allow_html=True,
        )
        for warning in report.get("warnings", []):
            st.markdown(f'<div class="warning-box">{escape(str(warning))}</div>', unsafe_allow_html=True)
    with ledger:
        for index, assumption in enumerate(report.get("assumptions", []), 1):
            status = localized_term(str(assumption.get("status", "unknown")), language)
            risk = localized_term(str(assumption.get("risk", "medium")), language)
            evidence_count = len(assumption.get("evidence", []))
            st.markdown(
                f'<article class="artifact"><div class="artifact-head"><strong class="artifact-title">{index}. {escape(str(assumption.get("text", "")))}</strong><div class="badge-row"><span class="badge">{escape(t(language, "signal"))}: {escape(status)}</span><span class="badge">{escape(t(language, "risk"))}: {escape(risk)}</span><span class="badge">{evidence_count} {escape(t(language, "evidence_count"))}</span></div></div></article>',
                unsafe_allow_html=True,
            )
    with evidence:
        st.markdown(render_discovery_evidence(report, language=language), unsafe_allow_html=True)
    with opportunities:
        for index, opportunity in enumerate(report.get("idea_opportunities", []), 1):
            kind = localized_term(str(opportunity.get("kind", "opportunity")), language)
            score = opportunity.get("score", "-")
            st.markdown(
                f'<article class="artifact"><div class="artifact-head"><strong class="artifact-title">{index}. {escape(str(opportunity.get("title", "")))}</strong><div class="badge-row"><span class="badge">{escape(kind)}</span><span class="badge">{escape(t(language, "score"))} {escape(str(score))}</span></div></div><p>{escape(str(opportunity.get("rationale", "")))}</p><p class="artifact-next"><strong>{escape(t(language, "next"))}:</strong> {escape(str(opportunity.get("next_step", "")))}</p></article>',
                unsafe_allow_html=True,
            )
    with trace:
        st.markdown(
            f'<section class="work-surface"><div class="section-head">{escape(t(language, "agent_activity"))}</div>{render_activity({"events": report.get("events", [])}, language=language)}</section>',
            unsafe_allow_html=True,
        )
    with export:
        markdown = report_markdown(report, language=language)
        st.download_button(t(language, "download_report"), markdown, "claimscope_report.md", "text/markdown")
        st.code(markdown, language="markdown")


def render_feedback(direction: str, observed: str, language: str) -> None:
    with st.expander(t(language, "evaluate")):
        expected = st.text_area(t(language, "expected"), key="feedback_expected")
        rating = st.selectbox(t(language, "rating"), ["Pass", "Partial", "Fail"], key="feedback_rating")
        notes = st.text_area(t(language, "notes"), key="feedback_notes")
        st.download_button(
            t(language, "download_feedback"),
            feedback_jsonl(direction, observed, expected, rating, notes),
            "claimscope_feedback.jsonl",
            "application/jsonl",
        )


def result_for_direction(
    result: dict | None,
    direction: str,
    field: str,
) -> dict | None:
    if not result:
        return None
    return result if str(result.get(field, "")).strip() == direction.strip() else None


def main() -> None:
    st.set_page_config(page_title="ClaimScope", page_icon="CS", layout="wide")
    session_defaults()
    inject_style()
    language = str(st.session_state.get("language", "zh"))
    mode, language = render_header(language)
    left, right = st.columns([0.92, 1.45])
    with left:
        direction, limit, run = render_research_controls(mode, language)
    with right:
        progress_slot = st.empty()
        result_slot = st.empty()
        with result_slot.container():
            if mode == "arena":
                render_core_claim(
                    result_for_direction(
                        st.session_state.get("core_claim_result"),
                        direction,
                        "direction",
                    ),
                    language,
                )
            else:
                render_discovery(
                    result_for_direction(
                        st.session_state.get("discovery_result"),
                        direction,
                        "query",
                    ),
                    language,
                )

    if run:
        if not direction.strip():
            st.warning(t(language, "enter_direction"))
        elif mode == "arena":
            st.session_state["core_claim_result"] = None
            result_slot.empty()
            live_events: dict[str, dict] = {}

            def on_agent_event(event: object) -> None:
                payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
                role = str(payload.get("role", "agent"))
                live_events[role] = payload
                progress_slot.markdown(render_live_progress(live_events, language), unsafe_allow_html=True)

            progress_slot.markdown(render_live_progress(live_events, language), unsafe_allow_html=True)
            try:
                service = build_service()
                st.session_state["core_claim_result"] = service.extract_core_claim(
                    direction,
                    mode="llm_strict",
                    on_event=on_agent_event,
                )
            except Exception:
                st.error(t(language, "run_failed"))
            else:
                st.rerun()
        else:
            st.session_state["discovery_result"] = None
            result_slot.empty()
            live_events: dict[str, dict] = {}

            def on_discovery_agent_event(event: object) -> None:
                payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
                role = str(payload.get("role", "agent"))
                live_events[role] = payload
                progress_slot.markdown(
                    render_live_progress(live_events, language),
                    unsafe_allow_html=True,
                )

            progress_slot.markdown(
                render_live_progress(live_events, language),
                unsafe_allow_html=True,
            )
            try:
                service = build_service()
                st.session_state["discovery_result"] = service.analyze_research_direction(
                    direction,
                    online=True,
                    limit=limit,
                    strict_agent=True,
                    on_event=on_discovery_agent_event,
                )
            except Exception:
                st.error(t(language, "discovery_failed"))
            else:
                st.rerun()

    if mode == "arena":
        result = result_for_direction(
            st.session_state.get("core_claim_result"),
            direction,
            "direction",
        ) or {}
        render_feedback(direction, str(result.get("selected_claim", "")), language)


if __name__ == "__main__":
    main()

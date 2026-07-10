from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ui_has_agent_arena_and_no_internal_thought_copy():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "Core Claim Arena" in source
    assert "Agent activity" in source
    assert "chain of thought" not in source.lower()


def test_ui_uses_session_state_for_results():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'st.session_state["core_claim_result"]' in source
    assert 'st.session_state["discovery_result"]' in source


def test_ui_contract_has_compact_modes_and_secret_safe_provider_copy():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "Full Discovery" in source
    assert "Download feedback JSONL" in source
    assert "st.session_state[\"provider_key\"]" in source
    assert "st.sidebar" not in source
    assert "linear-gradient" not in source


def test_ui_keeps_mode_selector_below_streamlit_header():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "padding:4.5rem 1.1rem 2.5rem" in source
    assert "padding:4rem 0.65rem 2rem" in source


def test_ui_composes_header_and_research_controls_as_real_containers():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "def render_header() -> str:" in source
    assert "mode = render_header()" in source
    assert "def render_research_controls(mode: str)" in source
    assert "\ufffd" not in source


def test_feedback_jsonl_contains_only_review_fields():
    from claimscope.ui_components import feedback_jsonl

    payload = feedback_jsonl(
        "direction",
        "observed",
        "expected",
        "Partial",
        "needs a narrower target",
    )
    assert payload.strip() == (
        '{"input": "direction", "observed_claim": "observed", '
        '"expected_claim": "expected", "rating": "Partial", '
        '"notes": "needs a narrower target"}'
    )
    assert "provider_key" not in payload


def test_result_summary_uses_public_events_only():
    from claimscope.ui_components import public_event_rows

    rows = public_event_rows(
        {
            "events": [
                {
                    "role": "operationalizer",
                    "status": "complete",
                    "public_summary": "Defined measurable variables.",
                    "duration_ms": 1200,
                    "artifacts": {"candidate_id": "operationalizer"},
                }
            ]
        }
    )
    assert rows[0]["role"] == "Operationalizer"
    assert rows[0]["summary"] == "Defined measurable variables."


def test_activity_markup_is_streamlit_safe_and_keeps_the_full_trace():
    from claimscope.ui_components import render_activity

    events = [
        {
            "role": f"stage_{index}",
            "status": "complete",
            "public_summary": f"Completed stage {index}.",
            "duration_ms": index,
            "artifacts": {"output": index},
        }
        for index in range(1, 9)
    ]
    markup = render_activity({"events": events})

    assert markup.startswith('<div class="activity-row">')
    assert "\n" not in markup
    assert markup.count('class="activity-row"') == len(events)


def test_candidate_markup_uses_css_radio_without_broken_glyphs():
    from claimscope.ui_components import render_candidate

    markup = render_candidate(
        {
            "claim": "A changes B.",
            "confidence": 0.8,
            "proposer": "judge",
            "method_or_mechanism": "A",
            "expected_effect": "B",
            "conditions": [],
        },
        selected=True,
        index=1,
    )

    assert "\n" not in markup
    assert 'class="radio radio-selected"' in markup
    assert markup.isascii()

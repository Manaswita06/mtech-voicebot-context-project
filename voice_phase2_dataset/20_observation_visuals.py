from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="RECAP-Voice | Experimental Observations",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background:
      radial-gradient(circle at 15% 8%, rgba(66, 110, 255, 0.14), transparent 28%),
      radial-gradient(circle at 88% 18%, rgba(168, 85, 247, 0.12), transparent 25%),
      linear-gradient(180deg, #07101f 0%, #091525 48%, #07101b 100%);
    color: #eef5ff;
}

section[data-testid="stSidebar"] {
    background: rgba(5, 12, 25, 0.96);
    border-right: 1px solid rgba(145, 170, 210, 0.13);
}

h1, h2, h3 {
    font-family: "Space Grotesk", sans-serif !important;
    letter-spacing: -0.025em;
}

.hero {
    padding: 26px 28px;
    border: 1px solid rgba(130,170,255,.18);
    border-radius: 24px;
    background:
        linear-gradient(135deg, rgba(22,39,69,.92), rgba(12,24,43,.85)),
        radial-gradient(circle at 80% 20%, rgba(109, 66, 255, .18), transparent 32%);
    box-shadow: 0 24px 70px rgba(0,0,0,.32);
    margin-bottom: 18px;
}

.eyebrow {
    text-transform: uppercase;
    letter-spacing: .18em;
    color: #8eb5ff;
    font-size: .72rem;
    font-weight: 800;
}

.hero h1 {
    font-size: 2.55rem;
    margin: 6px 0 8px;
}

.hero p {
    color: #b8c8e6;
    max-width: 900px;
    font-size: 1.02rem;
    line-height: 1.55;
}

.glass {
    background: rgba(12, 23, 40, .72);
    border: 1px solid rgba(141, 168, 207, .13);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 18px 48px rgba(0,0,0,.18);
}

.kpi {
    background: linear-gradient(180deg, rgba(19,35,60,.90), rgba(10,20,35,.90));
    border: 1px solid rgba(119, 155, 216, .14);
    border-radius: 16px;
    padding: 16px;
    min-height: 108px;
}

.kpi-label {
    color: #91a8ca;
    font-size: .77rem;
    text-transform: uppercase;
    letter-spacing: .11em;
    font-weight: 700;
}

.kpi-value {
    font-family: "Space Grotesk", sans-serif;
    color: #f2f7ff;
    font-size: 1.72rem;
    font-weight: 700;
    margin-top: 7px;
}

.kpi-note {
    color: #7f95b7;
    font-size: .76rem;
    margin-top: 4px;
}

.note {
    color: #8fa7ca;
    font-size: .84rem;
    line-height: 1.55;
}

.source-note {
    color: #6e86a8;
    font-size: .75rem;
    line-height: 1.45;
}

.pill {
    display:inline-block;
    padding: 5px 9px;
    border-radius: 999px;
    background: rgba(75, 121, 255, .12);
    border: 1px solid rgba(95, 145, 255, .18);
    color: #9fc0ff;
    font-size: .73rem;
    font-weight: 700;
    margin: 3px 4px 3px 0;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(154,179,220,.18), transparent);
    margin: 18px 0;
}

.story-title {
    text-align:center;
    margin: 6px 0 2px;
    font-family: "Space Grotesk", sans-serif;
    font-size: 1.45rem;
}

.story-subtitle {
    text-align:center;
    color:#8fa5c6;
    margin-bottom: 18px;
}

/* Animated claim audit */
.audit-stage {
    position: relative;
    min-height: 300px;
    overflow: hidden;
    border-radius: 22px;
    background: linear-gradient(145deg, rgba(10,24,42,.94), rgba(10,16,30,.94));
    border: 1px solid rgba(119,151,209,.12);
}

.audit-title {
    position:absolute; top:16px; left:18px;
    color:#a7bbdc; font-size:.72rem; letter-spacing:.14em;
    text-transform:uppercase; font-weight:800;
}

.claims {
    position:absolute; left:25px; right:25px; top:82px;
    display:flex; gap:8px; flex-wrap:wrap;
    justify-content:center;
}

.claim-dot {
    width:14px; height:14px; border-radius:50%;
    background:#79a8ff;
    opacity:0;
    animation: claimIn 1s cubic-bezier(.2,.7,.2,1) forwards;
}

.claim-dot:nth-child(1){animation-delay:.05s}
.claim-dot:nth-child(2){animation-delay:.10s}
.claim-dot:nth-child(3){animation-delay:.15s}
.claim-dot:nth-child(4){animation-delay:.20s}
.claim-dot:nth-child(5){animation-delay:.25s}
.claim-dot:nth-child(6){animation-delay:.30s}
.claim-dot:nth-child(7){animation-delay:.35s}
.claim-dot:nth-child(8){animation-delay:.40s}
.claim-dot:nth-child(9){animation-delay:.45s}
.claim-dot:nth-child(10){animation-delay:.50s}
.claim-dot:nth-child(11){animation-delay:.55s}
.claim-dot:nth-child(12){animation-delay:.60s}

@keyframes claimIn {
    0% { opacity:0; transform: translateY(-24px) scale(.45); }
    100% { opacity:1; transform: translateY(0) scale(1); }
}

.audit-arrows {
    position:absolute; left:50%; top:125px; transform:translateX(-50%);
    font-size:2rem; color:#5f83ff; animation:pulse 2.1s infinite;
}

@keyframes pulse {
    0%,100% {opacity:.35; transform:translateX(-50%) translateY(0);}
    50% {opacity:1; transform:translateX(-50%) translateY(4px);}
}

.audit-boxes {
    position:absolute; left:25px; right:25px; bottom:28px;
    display:flex; justify-content:center; gap:14px;
}

.audit-box {
    flex:1; max-width:180px;
    padding:16px; text-align:center;
    border-radius:15px;
    animation: boxIn 1s 1.0s both;
}

.audit-box.good {background:rgba(37,211,134,.09); border:1px solid rgba(37,211,134,.25);}
.audit-box.warn {background:rgba(255,184,84,.09); border:1px solid rgba(255,184,84,.25);}

@keyframes boxIn {
    from {opacity:0; transform:translateY(20px) scale(.94);}
    to {opacity:1; transform:none;}
}

.wave {
    height: 118px;
    display:flex;
    gap:3px;
    align-items:center;
    padding: 14px;
    border-radius: 18px;
    background: rgba(6,14,27,.84);
    border:1px solid rgba(117,148,203,.12);
    overflow:hidden;
}

.wave i {
    display:block;
    width:5px;
    border-radius:6px;
    background: linear-gradient(180deg,#7ca6ff,#7b61ff);
    animation: wave 1.15s ease-in-out infinite;
    animation-delay: calc(var(--i) * -0.06s);
    opacity:.78;
}
@keyframes wave {
    0%,100%{height:18px}
    50%{height:var(--h)}
}

.turn-lane {
    height: 18px;
    margin: 9px 0;
    border-radius: 999px;
    background: rgba(255,255,255,.04);
    overflow:hidden;
    position:relative;
}
.turn-segment {
    position:absolute;
    top:0; bottom:0;
    border-radius:999px;
    animation: growSeg 1.3s cubic-bezier(.2,.7,.2,1) both;
}
.turn-label {
    color:#8097b7; font-size:.72rem; margin-bottom:3px;
}
@keyframes growSeg {
    from {transform:scaleX(0); transform-origin:left;}
    to {transform:scaleX(1);}
}

/* Capsule build */
.capsule-scene {
    display:flex;
    align-items:center;
    justify-content:center;
    gap:26px;
    min-height:320px;
    padding:20px;
    border-radius:22px;
    background:linear-gradient(145deg,rgba(11,22,39,.98),rgba(7,13,25,.95));
    border:1px solid rgba(120,151,208,.12);
}
.raw-lines { width:42%; }
.raw-line {
    height:13px; border-radius:99px;
    background:linear-gradient(90deg,rgba(125,157,230,.36),rgba(125,157,230,.09));
    margin:12px 0;
    animation: lineDrift 3.4s ease-in-out infinite;
}
.raw-line:nth-child(2){width:82%; animation-delay:.1s}
.raw-line:nth-child(3){width:71%; animation-delay:.2s}
.raw-line:nth-child(4){width:91%; animation-delay:.3s}
.raw-line:nth-child(5){width:63%; animation-delay:.4s}
.raw-line:nth-child(6){width:78%; animation-delay:.5s}
@keyframes lineDrift {
    0%,100%{transform:translateX(0); opacity:.55}
    50%{transform:translateX(10px); opacity:.9}
}
.arrow-flow {
    color:#789aff;
    font-size:2.8rem;
    animation:pulse 1.8s infinite;
}
.capsule {
    width:42%;
    min-height:220px;
    padding:18px;
    border-radius:20px;
    background:
      radial-gradient(circle at 76% 22%, rgba(118,94,255,.22), transparent 28%),
      linear-gradient(160deg,#13294b,#0b1b31);
    border:1px solid rgba(121,167,255,.34);
    box-shadow:0 20px 50px rgba(44,89,180,.20);
    animation: capsuleFloat 4s ease-in-out infinite;
}
@keyframes capsuleFloat {
    0%,100%{transform:translateY(0)}
    50%{transform:translateY(-5px)}
}
.capsule-chip {
    display:inline-block;
    margin:4px;
    padding:7px 10px;
    border-radius:999px;
    background:rgba(124,161,255,.10);
    border:1px solid rgba(124,161,255,.22);
    color:#b4c9ef;
    font-size:.74rem;
}

/* Handoff lanes */
.handoff-lane {
    margin:11px 0 18px;
}
.lane-name {
    display:flex; justify-content:space-between;
    color:#9eafd0; font-size:.78rem;
    margin-bottom:5px;
}
.lane-track {
    height:24px;
    border-radius:999px;
    background:rgba(255,255,255,.04);
    position:relative;
    overflow:hidden;
}
.lane-fill {
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,#567fff,#a56cff);
    width:var(--w);
    animation: laneGrow 1.6s cubic-bezier(.2,.7,.2,1) both;
}
@keyframes laneGrow { from{width:0} to{width:var(--w)} }

.question-dots {
    min-height:58px;
    display:flex; gap:8px; align-items:center; flex-wrap:wrap;
}
.qdot {
    width:22px;height:22px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    color:#fff;font-size:.72rem;font-weight:800;
    background:rgba(255,184,84,.18);
    border:1px solid rgba(255,184,84,.35);
    animation: pop .5s both;
}
@keyframes pop {from{opacity:0;transform:scale(.3)}to{opacity:1;transform:scale(1)}}

/* Compression */
.memory-grid {
    display:grid;
    grid-template-columns: repeat(7, minmax(40px,1fr));
    gap:8px;
    padding:20px;
    border-radius:20px;
    background:rgba(7,16,29,.88);
    border:1px solid rgba(117,149,208,.11);
}
.memory-cell {
    height:52px;
    border-radius:12px;
    background:linear-gradient(145deg,#466fe5,#704cff);
    animation: memoryPulse 2.6s ease-in-out infinite;
}
.memory-cell:nth-child(2n){animation-delay:.15s}
.memory-cell:nth-child(3n){animation-delay:.3s}
@keyframes memoryPulse {
    0%,100%{transform:translateY(0);opacity:.75}
    50%{transform:translateY(-4px);opacity:1}
}

.big-question {
    text-align:center;
    font-family:"Space Grotesk",sans-serif;
    font-weight:700;
    font-size:2.05rem;
    color:#edf4ff;
    margin:16px 0 4px;
}

.small-capsule {
    display:inline-block;
    height:16px;
    border-radius:999px;
    background:linear-gradient(90deg,#4f7cff,#9164ff);
    animation: shrinkPulse 3.2s ease-in-out infinite;
}
@keyframes shrinkPulse {
    0%,20%{width:220px}
    35%,55%{width:165px}
    70%,82%{width:110px}
    100%{width:62px}
}

.disclaimer {
    border-left:3px solid #8b6dff;
    padding:10px 13px;
    margin-top:12px;
    color:#93a8c9;
    background:rgba(139,109,255,.06);
    font-size:.78rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def fmt(value: float, digits: int = 2, suffix: str = "") -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}{suffix}"


def pct(value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return value * 100 if abs(value) <= 1.0 else value


def find_first(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_observations(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in ("value",):
        if c in df.columns:
            df[c + "_num"] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_handoff(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def load_aggregate(path: str) -> dict[str, Any]:
    return load_json(Path(path))


def normalize_condition(name: str) -> str:
    n = str(name).lower()
    if "b0" in n or "transcript" in n:
        return "B0 — Full transcript"
    if "b1" in n or "summary" in n:
        return "B1 — Generic summary"
    if "b2" in n or "structured" in n or "state" in n:
        return "B2 — Structured state"
    if "b3" in n or "recap" in n:
        return "B3 — RECAP-Voice"
    return str(name)


def stage_rows(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["stage"].astype(str).str.startswith(prefix)].copy()


def mean_obs(df: pd.DataFrame, stage_prefix: str, observation: str) -> float:
    x = stage_rows(df, stage_prefix)
    x = x[x["observation"].eq(observation)]
    if x.empty:
        return float("nan")
    return pd.to_numeric(x["value"], errors="coerce").mean()


def median_obs(df: pd.DataFrame, stage_prefix: str, observation: str) -> float:
    x = stage_rows(df, stage_prefix)
    x = x[x["observation"].eq(observation)]
    if x.empty:
        return float("nan")
    return pd.to_numeric(x["value"], errors="coerce").median()


def render_kpi(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
<div class="kpi">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-note">{note}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def animated_wave():
    bars = []
    heights = [18, 28, 52, 76, 34, 61, 87, 42, 72, 30, 56, 91, 47, 70, 35, 63, 84, 44, 58, 27, 78, 49, 67, 38]
    for i, h in enumerate(heights):
        bars.append(f'<i style="--i:{i};--h:{h}px"></i>')
    st.markdown('<div class="wave">' + "".join(bars) + "</div>", unsafe_allow_html=True)


def audit_animation(total: int = 12, verified_fraction: float = .8):
    good = max(1, min(total - 1, round(total * verified_fraction)))
    bad = total - good
    dots = "".join('<span class="claim-dot"></span>' for _ in range(total))
    st.markdown(
        f"""
<div class="audit-stage">
  <div class="audit-title">CLAIM → EVIDENCE → AUDIT</div>
  <div class="claims">{dots}</div>
  <div class="audit-arrows">↓</div>
  <div class="audit-boxes">
    <div class="audit-box good">
      <div style="font-size:.76rem;color:#7ce0ae;text-transform:uppercase;letter-spacing:.12em;font-weight:800;">Verified</div>
      <div style="font-size:2rem;font-family:'Space Grotesk';font-weight:700;margin-top:4px;">{good}</div>
      <div style="font-size:.72rem;color:#7897a8;">evidence references pass</div>
    </div>
    <div class="audit-box warn">
      <div style="font-size:.76rem;color:#ffc36f;text-transform:uppercase;letter-spacing:.12em;font-weight:800;">Review</div>
      <div style="font-size:2rem;font-family:'Space Grotesk';font-weight:700;margin-top:4px;">{bad}</div>
      <div style="font-size:.72rem;color:#a78c6a;">requires attention</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def capsule_animation():
    chips = [
        "CUSTOMER GOAL", "INTENT", "ENTITIES", "ACTIONS",
        "EVIDENCE", "UNCERTAINTY", "INTERACTION"
    ]
    chip_html = "".join(f'<span class="capsule-chip">{c}</span>' for c in chips)
    raw = "".join('<div class="raw-line"></div>' for _ in range(6))
    st.markdown(
        f"""
<div class="capsule-scene">
  <div class="raw-lines">{raw}</div>
  <div class="arrow-flow">→</div>
  <div class="capsule">
      <div style="font-family:'Space Grotesk';font-weight:700;font-size:1.02rem;color:#f1f6ff;">RECAP-Voice</div>
      <div style="font-size:.72rem;color:#8299bd;margin-top:2px;">compact handover memory</div>
      <div style="margin-top:20px;">{chip_html}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def handoff_lanes(aggregate: dict[str, Any]):
    labels = []
    for raw_name, metrics in aggregate.items():
        label = normalize_condition(raw_name)
        f1 = safe_float(metrics.get("mean_slot_carryover_f1"))
        labels.append((label, f1))

    # Prefer canonical order.
    order = {
        "B0 — Full transcript": 0,
        "B1 — Generic summary": 1,
        "B2 — Structured state": 2,
        "B3 — RECAP-Voice": 3,
    }
    labels.sort(key=lambda x: order.get(x[0], 99))

    html = []
    for idx, (label, f1) in enumerate(labels):
        width = max(4, min(100, pct(f1) if not pd.isna(f1) else 0))
        html.append(
            f"""
<div class="handoff-lane">
  <div class="lane-name">
    <span>{label}</span>
    <span>{fmt(pct(f1),1,'%')}</span>
  </div>
  <div class="lane-track">
    <div class="lane-fill" style="--w:{width:.1f}%; animation-delay:{idx*0.12}s"></div>
  </div>
</div>
"""
        )
    st.markdown("".join(html), unsafe_allow_html=True)


def memory_cells(n: int):
    return "".join('<div class="memory-cell"></div>' for _ in range(n))


# ---------------------------------------------------------------------
# SIDEBAR / DATA DISCOVERY
# ---------------------------------------------------------------------

st.sidebar.markdown("## RECAP-Voice")
st.sidebar.caption("Experimental observation visualizer")

root_text = st.sidebar.text_input(
    "Project/data root",
    value=".",
    help="Run the app from the root of the RECAP-Voice project, or enter the path.",
)
ROOT = Path(root_text).expanduser().resolve()

obs_path = find_first([
    ROOT / "research_observations" / "observations.csv",
    ROOT / "observations.csv",
])
handoff_csv = find_first([
    ROOT / "handoff_benchmark" / "handoff_results.csv",
])
aggregate_path = find_first([
    ROOT / "handoff_benchmark" / "aggregate.json",
])

with st.sidebar.expander("Detected files", expanded=False):
    st.write(f"Observations: {'✓' if obs_path else '—'}")
    st.write(f"Handoff benchmark: {'✓' if handoff_csv else '—'}")
    st.write(f"Benchmark aggregate: {'✓' if aggregate_path else '—'}")

uploaded_obs = st.sidebar.file_uploader(
    "Optional: upload observations.csv",
    type=["csv"],
    help="Use this when the app is not running inside the experiment project root.",
)
uploaded_handoff = st.sidebar.file_uploader(
    "Optional: upload handoff_results.csv",
    type=["csv"],
)
uploaded_aggregate = st.sidebar.file_uploader(
    "Optional: upload aggregate.json",
    type=["json"],
)

if uploaded_obs:
    obs_df = pd.read_csv(uploaded_obs)
    for c in ("value",):
        obs_df[c + "_num"] = pd.to_numeric(obs_df[c], errors="coerce")
else:
    obs_df = load_observations(str(obs_path)) if obs_path else pd.DataFrame()

if uploaded_handoff:
    handoff_df = pd.read_csv(uploaded_handoff)
else:
    handoff_df = load_handoff(str(handoff_csv)) if handoff_csv else pd.DataFrame()

if uploaded_aggregate:
    aggregate = json.load(uploaded_aggregate)
elif aggregate_path:
    aggregate = load_aggregate(str(aggregate_path))
elif not handoff_df.empty:
    aggregate = {}
    for cond, g in handoff_df.groupby("condition"):
        aggregate[cond] = {
            "n_conversations": len(g),
            "mean_slot_carryover_f1": g["slot_carryover_f1"].mean(),
            "mean_unnecessary_reask_rate": g["unnecessary_reask_rate"].mean(),
            "mean_packet_bytes": g["packet_bytes"].mean(),
            "mean_approx_tokens": g["approx_tokens"].mean(),
        }
else:
    aggregate = {}

page = st.sidebar.radio(
    "Explore",
    [
        "Research Story",
        "13 · Evidence Audit",
        "14 · Acoustic Context",
        "15 · Handover Capsule",
        "16 · Handoff Benchmark",
        "17 · Ablation & Compression",
        "Raw Observations",
    ],
)

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

st.markdown(
    """
<div class="hero">
  <div class="eyebrow">M.Tech Research Demonstration · RECAP-Voice</div>
  <h1>What survives a conversational handoff?</h1>
  <p>
    An interactive visual layer for the measured observations produced by
    stages 13–17: evidence auditing, acoustic interaction context, compact
    handover capsules, handoff preservation, and ablation/compression.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

if obs_df.empty:
    st.warning(
        "No research_observations/observations.csv was found. "
        "Upload observations.csv in the sidebar or run stage 19 before using the quantitative views."
    )

# ---------------------------------------------------------------------
# RESEARCH STORY
# ---------------------------------------------------------------------

if page == "Research Story":
    st.markdown('<div class="story-title">The experiment as one visual story</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="story-subtitle">Conversation → audit → memory → handoff → compression</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Animated story", "Dataset snapshot", "Research guardrails"])

    with tab1:
        c1, c2 = st.columns([1.2, 1], gap="large")
        with c1:
            st.markdown("### 01 · Claims become auditable")
            verifier_rate = mean_obs(obs_df, "13_verifier", "evidence_support_rate")
            audit_animation(
                total=12,
                verified_fraction=(verifier_rate if not pd.isna(verifier_rate) else .8),
            )
        with c2:
            render_kpi(
                "Evidence support",
                fmt(pct(verifier_rate), 1, "%"),
                "mean structural support rate" if not pd.isna(verifier_rate) else "load stage 13 observations",
            )
            render_kpi(
                "Claims audited",
                fmt(mean_obs(obs_df, "13_verifier", "claims_checked"), 1),
                "mean per conversation",
            )

        st.markdown("### 02 · Interaction becomes measurable context")
        animated_wave()

        ac_metrics = [
            ("Conversation duration", mean_obs(obs_df, "14_acoustic", "duration_seconds"), "seconds"),
            ("Turn count", mean_obs(obs_df, "14_acoustic", "turn_count"), "turns"),
            ("Silence ratio", pct(mean_obs(obs_df, "14_acoustic", "silence_ratio")), "%"),
            ("Customer talk ratio", pct(mean_obs(obs_df, "14_acoustic", "customer_talk_ratio")), "%"),
        ]
        cols = st.columns(4)
        for col, (lab, val, unit) in zip(cols, ac_metrics):
            with col:
                render_kpi(lab, fmt(val, 1, f" {unit}"), "dataset mean")

        st.markdown("### 03 · Rich conversation collapses into compact memory")
        capsule_animation()

        cap_bytes = median_obs(obs_df, "15_capsule", "capsule_bytes")
        st.markdown(
            f"""
<div class="glass">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div class="eyebrow">Serialized handover capsule</div>
      <div style="font-family:'Space Grotesk';font-size:2rem;font-weight:700;margin-top:3px;">
        {fmt(cap_bytes,0)} bytes
      </div>
    </div>
    <div style="text-align:right;color:#8da5c8;font-size:.8rem;">
      median across observed capsules
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("### 04 · The same information is tested after handoff")
        if aggregate:
            handoff_lanes(aggregate)

            # Re-ask dots
            b3 = next((v for k,v in aggregate.items() if "b3" in str(k).lower() or "recap" in str(k).lower()), None)
            b3_reask = safe_float(b3.get("mean_unnecessary_reask_rate")) if b3 else float("nan")
            qn = max(1, min(12, round(pct(b3_reask) / 8))) if not pd.isna(b3_reask) else 3
            qs = "".join(f'<div class="qdot" style="animation-delay:{i*0.08}s">?</div>' for i in range(qn))
            st.markdown(
                f"""
<div class="glass">
  <div class="eyebrow">UNNECESSARY RE-ASK</div>
  <div class="question-dots">{qs}</div>
  <div class="note">
    The benchmark defines unnecessary re-ask rate as the fraction of gold slots not preserved.
    This is an information-preservation metric, not an LLM response-quality score.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("### 05 · Memory can be systematically reduced")
        st.markdown(
            '<div class="big-question">How small can memory become before useful information disappears?</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="text-align:center;margin:16px 0 12px;"><span class="small-capsule"></span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
<div class="memory-grid">
""" + memory_cells(14) + """
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="disclaimer">Stages 17 observations currently measure packet size for ablation/compression variants. Handoff quality for each ablation/compression level must be benchmarked separately before making a performance claim.</div>',
            unsafe_allow_html=True,
        )

    with tab2:
        if not obs_df.empty:
            st.metric("Observation rows", f"{len(obs_df):,}")
            st.metric("Conversations represented", f"{obs_df['conversation_id'].nunique():,}")
            st.metric("Observed stages", ", ".join(sorted(obs_df["stage"].astype(str).str.split(":").str[0].unique())))
            st.dataframe(
                obs_df.head(200),
                width=True,
                hide_index=True,
            )
        else:
            st.info("Upload or generate observations.csv.")

    with tab3:
        st.markdown(
            """
<div class="glass">
<div class="pill">Stage 13</div>
<div class="note">
Structural evidence-reference verification. A claim is treated as VERIFIED when its
evidence references are structurally valid; this does not perform semantic entailment.
</div>
<div class="divider"></div>
<div class="pill">Stage 14</div>
<div class="note">
Silence, timing, overlap, turn-taking and speaking-rate measurements. These are
interaction observations and are not used here as emotion, honesty or intent inference.
</div>
<div class="divider"></div>
<div class="pill">Stage 17</div>
<div class="note">
Ablation/compression packet sizes are descriptive observations. Do not treat packet-size
changes as quality changes without a corresponding handoff benchmark.
</div>
</div>
""",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------
# 13
# ---------------------------------------------------------------------

elif page == "13 · Evidence Audit":
    st.subheader("Stage 13 — Evidence Audit")
    st.caption("Claims are audited against their referenced evidence and flagged for review when structural checks fail.")

    rate = mean_obs(obs_df, "13_verifier", "evidence_support_rate")
    checked = mean_obs(obs_df, "13_verifier", "claims_checked")
    verified = mean_obs(obs_df, "13_verifier", "verified_claims")
    review = mean_obs(obs_df, "13_verifier", "review_claims")

    c1,c2,c3,c4 = st.columns(4)
    for col, label, value, unit, note in [
        (c1, "Support rate", pct(rate), "%", "structural evidence support"),
        (c2, "Claims checked", checked, "", "mean per conversation"),
        (c3, "Verified", verified, "", "mean per conversation"),
        (c4, "Review", review, "", "mean per conversation"),
    ]:
        with col:
            render_kpi(label, fmt(value, 1, unit), note)

    st.markdown("### Animated evidence audit")
    audit_animation(16, rate if not pd.isna(rate) else .8)

    if not obs_df.empty:
        review_fields = stage_rows(obs_df, "13_verifier")
        review_fields = review_fields[review_fields["observation"].eq("review_fields")]
        if not review_fields.empty:
            st.markdown("### Fields surfaced for review")
            counts = {}
            for text in review_fields["value"].dropna().astype(str):
                for field in text.split(";"):
                    field = field.strip()
                    if field:
                        counts[field] = counts.get(field, 0) + 1
            if counts:
                chart = pd.DataFrame(sorted(counts.items(), key=lambda x: -x[1]), columns=["field","count"])
                st.bar_chart(chart.set_index("field"))

    st.markdown(
        '<div class="disclaimer">Important: this stage verifies evidence references and timestamps structurally. It does not independently establish semantic entailment between a claim and a quote.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# 14
# ---------------------------------------------------------------------

elif page == "14 · Acoustic Context":
    st.subheader("Stage 14 — Conversation Vital Signs")
    st.caption("Acoustic and turn-taking measurements extracted from the selected canonical timing source.")

    metrics = {
        "duration_seconds": mean_obs(obs_df, "14_acoustic", "duration_seconds"),
        "turn_count": mean_obs(obs_df, "14_acoustic", "turn_count"),
        "silence_ratio": mean_obs(obs_df, "14_acoustic", "silence_ratio"),
        "customer_talk_ratio": mean_obs(obs_df, "14_acoustic", "customer_talk_ratio"),
        "customer_wpm": mean_obs(obs_df, "14_acoustic", "customer_words_per_minute"),
        "agent_wpm": mean_obs(obs_df, "14_acoustic", "agent_words_per_minute"),
        "overlap_count": mean_obs(obs_df, "14_acoustic", "overlap_count"),
        "overlap_seconds": mean_obs(obs_df, "14_acoustic", "overlap_seconds"),
    }

    cols = st.columns(4)
    items = [
        ("Duration", fmt(metrics["duration_seconds"],1," s")),
        ("Turns", fmt(metrics["turn_count"],1)),
        ("Silence", fmt(pct(metrics["silence_ratio"]),1,"%")),
        ("Customer share", fmt(pct(metrics["customer_talk_ratio"]),1,"%")),
    ]
    for col,(label,val) in zip(cols,items):
        with col:
            render_kpi(label,val,"dataset mean")

    st.markdown("### Animated conversation waveform")
    animated_wave()

    st.markdown("### Turn-taking structure")
    # Visual aggregate conversation profile, not a reconstructed individual call.
    turn_count = int(round(metrics["turn_count"])) if not pd.isna(metrics["turn_count"]) else 8
    turn_count = max(4,min(10,turn_count))
    segments = []
    for i in range(turn_count):
        role = "CUSTOMER" if i % 2 == 0 else "AGENT"
        start = i / turn_count * 100
        width = 100 / turn_count - 1.4
        opacity = .84 if role == "AGENT" else .62
        color = "#7c9dff" if role == "AGENT" else "#7f6cff"
        segments.append(
            f'<div class="turn-segment" style="left:{start:.1f}%;width:{width:.1f}%;background:{color};opacity:{opacity};animation-delay:{i*0.08}s"></div>'
        )
    st.markdown(
        f"""
<div class="glass">
  <div class="turn-label">CUSTOMER / AGENT TURN STRUCTURE · aggregate illustrative timeline</div>
  <div class="turn-lane">{"".join(segments)}</div>
  <div class="turn-label">Long pauses ≥ 0.5 s: {fmt(mean_obs(obs_df,"14_acoustic","silence_gaps_over_half_second"),1)} · Overlap events: {fmt(metrics["overlap_count"],1)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Speech-rate balance")
    rates = pd.DataFrame(
        {
            "role": ["Customer", "Agent"],
            "WPM": [metrics["customer_wpm"], metrics["agent_wpm"]],
        }
    )
    st.bar_chart(rates.set_index("role"))

# ---------------------------------------------------------------------
# 15
# ---------------------------------------------------------------------

elif page == "15 · Handover Capsule":
    st.subheader("Stage 15 — Compact Handover Memory")
    st.caption("The composer combines state, verified claims, uncertainty and acoustic context into a channel-agnostic capsule.")

    vals = {
        "entity_count": mean_obs(obs_df, "15_capsule", "entity_count"),
        "actions_count": mean_obs(obs_df, "15_capsule", "actions_count"),
        "promise_count": mean_obs(obs_df, "15_capsule", "promise_count"),
        "unresolved_item_count": mean_obs(obs_df, "15_capsule", "unresolved_item_count"),
        "review_flag_count": mean_obs(obs_df, "15_capsule", "review_flag_count"),
        "capsule_bytes": median_obs(obs_df, "15_capsule", "capsule_bytes"),
    }

    cols = st.columns(6)
    kpis = [
        ("Entities", vals["entity_count"]),
        ("Actions", vals["actions_count"]),
        ("Promises", vals["promise_count"]),
        ("Unresolved", vals["unresolved_item_count"]),
        ("Review flags", vals["review_flag_count"]),
        ("Capsule size", vals["capsule_bytes"]),
    ]
    for col,(label,val) in zip(cols,kpis):
        with col:
            render_kpi(label, fmt(val,1," bytes" if label=="Capsule size" else ""), "observed")

    st.markdown("### Animated capsule construction")
    capsule_animation()

    st.markdown("### What the capsule carries")
    capsule_df = pd.DataFrame(
        [
            ["Goal", 1],
            ["Intent", 1],
            ["Entities", vals["entity_count"]],
            ["Actions", vals["actions_count"]],
            ["Promises", vals["promise_count"]],
            ["Unresolved", vals["unresolved_item_count"]],
            ["Review flags", vals["review_flag_count"]],
        ],
        columns=["field","count"],
    )
    st.bar_chart(capsule_df.set_index("field"))

# ---------------------------------------------------------------------
# 16
# ---------------------------------------------------------------------

elif page == "16 · Handoff Benchmark":
    st.subheader("Stage 16 — Information Survival After Handoff")
    st.caption("B0/B1/B2/B3 are compared on gold-slot carryover, unnecessary re-ask rate, serialized context size and approximate token count.")

    if not aggregate:
        st.warning("No handoff benchmark data found. Provide handoff_benchmark/aggregate.json or upload the CSV.")
    else:
        order = [
            "B0 — Full transcript",
            "B1 — Generic summary",
            "B2 — Structured state",
            "B3 — RECAP-Voice",
        ]

        normalized = {}
        for raw_name, data in aggregate.items():
            normalized[normalize_condition(raw_name)] = data

        st.markdown("### Animated handoff lanes")
        handoff_lanes(aggregate)

        rows = []
        for label in order:
            if label in normalized:
                d = normalized[label]
                rows.append(
                    {
                        "Condition": label,
                        "Carryover F1": pct(safe_float(d.get("mean_slot_carryover_f1"))),
                        "Unnecessary re-ask": pct(safe_float(d.get("mean_unnecessary_reask_rate"))),
                        "Packet bytes": safe_float(d.get("mean_packet_bytes")),
                        "Approx tokens": safe_float(d.get("mean_approx_tokens")),
                    }
                )
        bench = pd.DataFrame(rows)
        if not bench.empty:
            st.dataframe(bench, width=True, hide_index=True)

            st.markdown("### The trade-off: preservation vs context size")
            chart_df = bench[["Condition","Carryover F1","Packet bytes"]].copy()
            chart_df["Carryover F1"] = chart_df["Carryover F1"] / 100.0
            chart_df = chart_df.set_index("Condition")
            st.line_chart(chart_df[["Carryover F1"]])

            st.markdown("### Unnecessary re-ask")
            rq = bench[["Condition","Unnecessary re-ask"]].set_index("Condition")
            st.bar_chart(rq)

        st.markdown(
            '<div class="disclaimer">This benchmark is an information-preservation benchmark. The current stage-16 implementation does not yet measure real downstream LLM response quality.</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------
# 17
# ---------------------------------------------------------------------

elif page == "17 · Ablation & Compression":
    st.subheader("Stage 17 — What happens when memory is reduced?")
    st.caption("Current observations measure serialized packet size for ablation and compression variants.")

    if obs_df.empty:
        st.info("Upload observations.csv to populate this view.")
    else:
        x = stage_rows(obs_df, "17_").copy()
        x["value_num"] = pd.to_numeric(x["value"], errors="coerce")
        x = x[x["observation"].eq("packet_bytes")].copy()

        if x.empty:
            st.info("No stage-17 packet-size observations found.")
        else:
            # Aggregate labels.
            x["variant"] = x["stage"].astype(str).str.replace(
                r"^17_[^:]+:", "", regex=True
            )
            summary = (
                x.groupby("variant")["value_num"]
                .agg(["mean","median","count"])
                .reset_index()
                .sort_values("mean", ascending=False)
            )

            st.markdown("### Animated memory reduction")
            max_bytes = summary["mean"].max()
            cards = []
            for i,row in summary.iterrows():
                width = 100 * row["mean"] / max_bytes if max_bytes else 0
                cards.append(
                    f"""
<div class="handoff-lane">
  <div class="lane-name">
    <span>{row['variant']}</span>
    <span>{fmt(row['mean'],0)} bytes</span>
  </div>
  <div class="lane-track">
    <div class="lane-fill" style="--w:{width:.1f}%;animation-delay:{len(cards)*0.15}s"></div>
  </div>
</div>
"""
                )
            st.markdown("".join(cards), unsafe_allow_html=True)

            st.dataframe(
                summary.rename(columns={"mean":"Mean bytes","median":"Median bytes","count":"N"}),
                width=True,
                hide_index=True,
            )

            st.markdown("### Leave-one-block-out")
            ab = x[x["stage"].str.startswith("17_ablation")]
            if not ab.empty:
                ab2 = (
                    ab.assign(
                        variant=ab["stage"].str.replace(
                            r"^17_ablation:", "", regex=True
                        )
                    )
                    .groupby("variant")["value_num"]
                    .mean()
                    .sort_values(ascending=False)
                )
                st.bar_chart(ab2)

            st.markdown("### Compression ladder")
            comp = x[x["stage"].str.startswith("17_compression")]
            if not comp.empty:
                comp2 = (
                    comp.assign(
                        variant=comp["stage"].str.replace(
                            r"^17_compression:", "", regex=True
                        )
                    )
                    .groupby("variant")["value_num"]
                    .mean()
                    .reset_index()
                )
                st.bar_chart(comp2.set_index("variant"))

            st.markdown(
                '<div class="disclaimer">A smaller packet is an observed storage change, not evidence of preserved service quality. To claim that a compressed or ablated memory still works, run the same stage-16 handoff benchmark on those variants.</div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------
# RAW
# ---------------------------------------------------------------------

elif page == "Raw Observations":
    st.subheader("Raw observation table")
    if obs_df.empty:
        st.info("No observations.csv loaded.")
    else:
        st.dataframe(obs_df, width=True, hide_index=True)

        st.download_button(
            "Download currently loaded observations.csv",
            obs_df.to_csv(index=False).encode("utf-8"),
            file_name="observations.csv",
            mime="text/csv",
        )

readme = r"""
# RECAP-Voice Observation Visualizer

This is a standalone Streamlit visualization app for the **measured observations from stages 13–17**.

It does **not** run another LLM demo and does **not** replace the existing experimental pipeline.

## What it visualizes

### Stage 13 — Evidence Audit
Animated claim → evidence → audit flow, with:
- evidence support rate
- claims checked
- verified claims
- review claims
- review fields

The app preserves the stage-13 interpretation: this is **structural evidence-reference verification**, not semantic entailment.

### Stage 14 — Conversation Vital Signs
Animated waveform and turn-taking view using:
- duration
- turn count
- silence ratio
- customer talk ratio
- customer/agent WPM
- overlap count
- overlap duration
- long-pause observations

### Stage 15 — Handover Capsule
Animated:
**full conversation → compact RECAP-Voice memory**

It visualizes the observed counts for:
- entities
- actions
- promises
- unresolved items
- review flags
- serialized capsule bytes

### Stage 16 — Handoff Benchmark
Animated four-lane comparison:
- B0 Full transcript
- B1 Generic summary
- B2 Structured state
- B3 RECAP-Voice

The app displays:
- slot carryover F1
- unnecessary re-ask rate
- packet bytes
- approximate tokens

### Stage 17 — Ablation & Compression
Animated packet-size reduction for:
- leave-one-block-out variants
- compression levels

The UI deliberately does **not** claim quality improvement/degradation from packet size alone.

## Expected folder layout

Run the app from the root of your RECAP-Voice project:

```text
project_root/
├── research_observations/
│   └── observations.csv
├── handoff_benchmark/
│   ├── handoff_results.csv
│   └── aggregate.json
├── ablation_packets/
│   ├── ablations/
│   └── compression/
└── recap_observation_visualizer.py
```

The app can also accept `observations.csv`, `handoff_results.csv`, and `aggregate.json` through the sidebar upload controls.

## Run

```bash
pip install -r requirements.txt
streamlit run recap_observation_visualizer.py
```

## Recommended final-presentation flow

Use **Research Story** first.

Then show:
1. Evidence Audit
2. Conversation Vital Signs
3. Capsule Construction
4. Handoff Benchmark
5. Ablation & Compression

The important narrative becomes:

**Conversation → Auditable Evidence → Compact Memory → Handoff → Information Survival**
"""

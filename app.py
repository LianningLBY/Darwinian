"""
Darwinian — 前端界面

启动命令：
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import threading
import queue
from datetime import datetime
from typing import Any

import streamlit as st
from dotenv import load_dotenv

try:
    from streamlit_autorefresh import st_autorefresh as _st_autorefresh
    _HAS_AUTOREFRESH = True
except ModuleNotFoundError:
    _HAS_AUTOREFRESH = False

load_dotenv()

# ──────────────────────────────────────────────
# 页面基础配置
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Darwinian · 自动化科研系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 全局 CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
body, .stApp { background-color: #07090d !important; }
.main .block-container {
    padding-top: 0.75rem;
    max-width: 1440px;
    background: #07090d;
}

/* ── Hide Streamlit default header chrome ── */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer { display: none !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #07090d !important;
    border-right: 1px solid #131929 !important;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ── Header ── */
.app-header {
    padding: 18px 0 14px;
    margin-bottom: 24px;
    border-bottom: 1px solid #131929;
}
.app-header-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.app-header-left { display: flex; flex-direction: column; gap: 2px; }
.app-header-wordmark {
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #dde1ea;
}
.app-header-sub {
    font-size: 0.78rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #7a8a9e;
    font-weight: 400;
}
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6366f1;
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 3px;
    padding: 4px 11px;
    background: rgba(99,102,241,0.06);
}
.live-badge .live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #6366f1;
    animation: pulsedot 1.6s ease-in-out infinite;
}
.static-badge {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6b7a8d;
    border: 1px solid #1e2a38;
    border-radius: 3px;
    padding: 4px 11px;
}

/* ── Section titles ── */
.section-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #7a8a9e;
    margin: 24px 0 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #131929;
}

/* ── Agent cards ── */
.agent-card {
    position: relative;
    padding: 14px 16px;
    border-radius: 6px;
    margin-bottom: 6px;
    background: #0b0e15;
    border: 1px solid #131929;
    border-left: 3px solid #1e2a38;
    transition: all 0.25s ease;
}
.agent-card.running {
    border-left: 4px solid #6366f1;
    background: #0e1020;
    border-color: #252a4a;
    box-shadow: 0 0 0 1px rgba(99,102,241,0.15), 0 4px 24px rgba(99,102,241,0.12);
    padding: 16px 16px 14px;
}
/* 底部扫描进度条 */
.agent-card.running::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0;
    height: 2px;
    width: 100%;
    background: linear-gradient(90deg, transparent, #6366f1, transparent);
    border-radius: 0 0 6px 6px;
    animation: scan 2s linear infinite;
}
@keyframes scan {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}
.agent-card.done {
    border-left-color: #10b981;
    border-color: #131d1a;
}
.agent-card.error {
    border-left-color: #ef4444;
    border-color: #1f1316;
}
.agent-card.pending {
    opacity: 0.4;
}

/* Status pill — top right corner */
.agent-status-pill {
    position: absolute;
    top: 11px;
    right: 12px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 3px;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.pill-queued  { color: #5a6a7e; background: transparent; border: 1px solid #2a3a50; }
.pill-running {
    color: #c7d2fe;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    box-shadow: 0 0 8px rgba(99,102,241,0.2);
}
.pill-done    { color: #10b981; background: rgba(16,185,129,0.07); border: 1px solid rgba(16,185,129,0.2); }
.pill-error   { color: #ef4444; background: rgba(239,68,68,0.07);  border: 1px solid rgba(239,68,68,0.2); }

/* Agent card body */
.agent-headline {
    font-size: 1rem;
    font-weight: 600;
    color: #dde1ea;
    margin-bottom: 4px;
    padding-right: 100px;
    letter-spacing: 0.1px;
}
.agent-card.running .agent-headline {
    color: #c7d2fe;
    font-size: 1.05rem;
}
.agent-card.done    .agent-headline { color: #6ee7b7; }
.agent-excerpt {
    font-size: 0.85rem;
    color: #a0b0c8;
    line-height: 1.5;
}
.agent-card.running .agent-excerpt { color: #b4c0e0; }

/* running pulse */
.pulse-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #818cf8;
    display: inline-block;
    animation: pulsedot 1.2s ease-in-out infinite;
    box-shadow: 0 0 6px rgba(99,102,241,0.6);
}
@keyframes pulsedot {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.5); }
}

/* ── Agent detail ── */
.agent-detail {
    margin-top: 10px;
    padding: 10px 12px;
    background: #07090d;
    border-radius: 5px;
    border: 1px solid #131929;
    font-size: 0.88rem;
}
.ad-row { display: flex; align-items: flex-start; gap: 8px; margin-bottom: 6px; }
.ad-label {
    color: #8a9ab0;
    white-space: nowrap;
    min-width: 64px;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding-top: 2px;
    font-weight: 500;
}
.ad-value { color: #b8c8dc; line-height: 1.5; font-size: 0.88rem; }
.ad-badge {
    padding: 3px 9px;
    border-radius: 3px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.ad-pass { background: rgba(16,185,129,0.10); color: #10b981; border: 1px solid rgba(16,185,129,0.2); }
.ad-fail { background: rgba(239,68,68,0.08);  color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }
.ad-warn { background: rgba(217,119,6,0.08);  color: #d97706; border: 1px solid rgba(217,119,6,0.2); }
.ad-feedback { color: #a0b0c8; margin-top: 5px; line-height: 1.6; font-size: 0.86rem; }
.ad-ev {
    color: #a0b0c8;
    margin-bottom: 3px;
    padding-left: 10px;
    border-left: 2px solid #1e2a38;
    font-size: 0.85rem;
}
.ad-hyp {
    margin-bottom: 6px;
    padding: 8px 10px;
    background: #0b0e15;
    border-radius: 4px;
    border: 1px solid #131929;
}
.ad-hyp-name { font-weight: 600; color: #818cf8; display: block; margin-bottom: 2px; font-size: 0.88rem; }
.ad-hyp-desc { color: #a0b0c8; font-size: 0.84rem; line-height: 1.5; }

/* ── Publish matrix ── */
.publish-matrix {
    display: flex;
    flex-direction: row;
    gap: 0;
    margin-top: 4px;
    border: 1px solid #131929;
    border-radius: 5px;
    overflow: hidden;
}
.matrix-item {
    flex: 1;
    padding: 16px 10px;
    text-align: center;
    background: #0b0e15;
    border-right: 1px solid #131929;
    transition: background 0.2s;
}
.matrix-item:last-child { border-right: none; }
.matrix-item.green { background: #080f0d; }
.matrix-item.grey  { opacity: 0.5; }
.matrix-icon {
    font-size: 1.3rem;
    display: block;
    margin-bottom: 7px;
    line-height: 1;
}
.matrix-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #9aaabf;
    margin-bottom: 6px;
    display: block;
}
.matrix-pass   { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.8px; color: #10b981; text-transform: uppercase; }
.matrix-pending { font-size: 0.75rem; font-weight: 500; letter-spacing: 0.8px; color: #8a9ab0; text-transform: uppercase; }

/* ── Log console ── */
.log-console {
    background: #030407;
    border: 1px solid #131929;
    border-radius: 6px;
    overflow: hidden;
}
.log-console-header {
    padding: 7px 12px;
    background: #07090d;
    border-bottom: 1px solid #131929;
    display: flex;
    align-items: center;
    gap: 6px;
}
.log-dot { width: 8px; height: 8px; border-radius: 50%; }
.log-console-filename {
    font-size: 0.75rem;
    color: #6b7a8d;
    margin-left: 8px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Menlo', monospace;
    letter-spacing: 0.5px;
}
.log-console-body {
    padding: 10px 14px;
    height: 240px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', 'Fira Code', 'Menlo', monospace;
    font-size: 0.83rem;
    line-height: 1.75;
    background: #030407;
}

/* ── Metrics table ── */
.metrics-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}
.metrics-table thead tr { background: #07090d; }
.metrics-table th {
    text-align: left;
    padding: 9px 12px;
    border-bottom: 1px solid #131929;
    color: #6b7a8d;
    font-weight: 500;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metrics-table td {
    padding: 9px 12px;
    border-bottom: 1px solid #0e1118;
    color: #a0b0c8;
}
.metrics-table tbody tr:nth-child(even) td { background: #09090e; }
.metrics-table tbody tr:hover td { background: #0c0e16; }
.metrics-table .better { color: #10b981; font-weight: 600; }
.metrics-table .worse  { color: #ef4444; }
.metrics-table td:first-child { color: #c0d0e4; font-weight: 500; }

/* ── Failed ledger ── */
.ledger-item {
    padding: 10px 14px;
    border-radius: 5px;
    border: 1px solid #131929;
    border-left: 2px solid #ef4444;
    background: #0b0e15;
    margin-bottom: 5px;
    font-size: 0.88rem;
    color: #a0b0c8;
    line-height: 1.5;
}
.ledger-type {
    display: inline-block;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #ef4444;
    margin-right: 8px;
    padding: 2px 7px;
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.15);
    border-radius: 3px;
    vertical-align: middle;
}

/* ── Loop badge ── */
.loop-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: #6366f1;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
}

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Agent 元数据
# ──────────────────────────────────────────────
AGENTS = [
    ("bottleneck_miner",    "🔍", "Agent 1 · 瓶颈挖掘机",    "检索文献，提取 Limitations，识别核心矛盾"),
    ("hypothesis_generator","💡", "Agent 2 · 方案合成器",    "生成跨域解决思路，输出强类型假设对象"),
    ("theoretical_critic",  "⚖️",  "Agent 3 · 理论审查官",    "审查数学可行性与新颖性，输出 PASS/MATH_ERROR/NOT_NOVEL"),
    ("code_architect",      "🏗️",  "Agent 4 · 代码架构师",    "编写 Baseline + Proposed Method 实验代码"),
    ("diagnostician",       "🩺", "Agent 5 · 诊断分析师",    "区分 code_bug vs logic_flaw，驱动修复循环"),
    ("poison_generator",    "☠️",  "Agent 6 · 毒药数据生成器","从策略库选择扰动组合，生成对抗性测试代码"),
    ("publish_evaluator",   "🏆", "Agent 7 · 成果验收员",    "终局裁判，更新 publish_matrix，生成研究报告"),
]

# 每个 Agent 的 LLM 具体任务说明（显示在 OUTPUT TRACE 中）
AGENT_TASKS: dict[str, str] = {
    "bottleneck_miner":     "📚 文献检索 · 提取 Limitations · 识别核心矛盾",
    "hypothesis_generator": "💡 跨域假设生成 · 构建解决方案树",
    "theoretical_critic":   "⚖️ 数学可行性审查 · 新颖性验证",
    "code_architect":       "🏗️ 实验代码生成 · Baseline vs Proposed",
    "diagnostician":        "🩺 错误诊断 · 区分代码缺陷 / 逻辑缺陷",
    "poison_generator":     "☠️ 对抗性数据生成 · 鲁棒性测试",
    "publish_evaluator":    "🏆 结果验收 · 发表可行性评估 · 生成报告",
    "dataset_finder":       "🗄️ 数据集搜索 · 匹配最优数据集",
}

# ──────────────────────────────────────────────
# Session State 初始化
# ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "running":          False,
        "finished":         False,
        "agent_status":     {a[0]: "pending" for a in AGENTS},  # pending/running/done/error
        "agent_output":     {a[0]: "" for a in AGENTS},
        "failed_ledger":    [],
        "publish_matrix":   {"novelty": False, "baseline": False, "robustness": False, "explain": False},
        "baseline_metrics": {},
        "proposed_metrics": {},
        "poison_metrics":   {},
        "final_verdict":    None,
        "final_report":     "",
        "outer_loop":       0,
        "max_loops":        5,
        "log_queue":        queue.Queue(),
        "current_hypothesis": "",
        "critic_verdict":   "",
        "logs":             [],   # list of (timestamp_str, level, message)
        "agent_detail":     {a[0]: {} for a in AGENTS},
        "current_stream":   "",   # 当前 LLM 流式输出缓冲
        "stream_history":   [],   # 已完成的 LLM 输出历史 [{"agent", "think", "response"}]
        "_stream_agent":      "",   # 当前正在流式输出的 agent 名称
        "_stream_agent_task": "",   # 当前 LLM 具体任务说明
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ──────────────────────────────────────────────
# 辅助渲染函数
# ──────────────────────────────────────────────
def render_header():
    is_running = st.session_state.get("running", False)
    if is_running:
        right_html = '<span class="live-badge"><span class="live-dot"></span>LIVE</span>'
    else:
        right_html = '<span class="static-badge">IDLE</span>'
    st.markdown(f"""
    <div class="app-header">
        <div class="app-header-inner">
            <div class="app-header-left">
                <span class="app-header-wordmark">Darwinian</span>
                <span class="app-header-sub">Intelligence</span>
            </div>
            {right_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_agent_detail(agent_id: str) -> str:
    """把 agent_detail[agent_id] 渲染为 HTML 片段。"""
    import html as _html
    def _e(s: str) -> str:          # HTML-escape 防止 LLM 内容破坏结构
        return _html.escape(str(s))

    d = st.session_state.agent_detail.get(agent_id, {})
    if not d:
        return ""
    parts = []

    if agent_id == "bottleneck_miner":
        if d.get("core_problem"):
            parts.append(f'<div class="ad-row"><span class="ad-label">核心矛盾</span>'
                         f'<span class="ad-value">{_e(d["core_problem"])}</span></div>')
        for ev in d.get("evidence", [])[:3]:
            parts.append(f'<div class="ad-ev">📄 {_e(ev)}</div>')

    elif agent_id == "hypothesis_generator":
        for h in d.get("hypotheses", []):
            parts.append(f'<div class="ad-hyp"><span class="ad-hyp-name">{_e(h["name"])}</span>'
                         f'<span class="ad-hyp-desc">{_e(h["desc"])}</span></div>')

    elif agent_id == "theoretical_critic":
        verdict = d.get("verdict", "")
        vcls = {"PASS": "ad-pass", "MATH_ERROR": "ad-fail", "NOT_NOVEL": "ad-warn"}.get(verdict, "")
        parts.append(f'<div class="ad-row"><span class="ad-label">裁决</span>'
                     f'<span class="ad-badge {vcls}">{_e(verdict)}</span></div>')
        if d.get("feedback"):
            parts.append(f'<div class="ad-feedback">{_e(d["feedback"][:200])}</div>')

    elif agent_id == "code_architect":
        if d.get("dataset"):
            parts.append(f'<div class="ad-row"><span class="ad-label">数据集</span>'
                         f'<span class="ad-value">{_e(d["dataset"])}</span></div>')
        if d.get("method"):
            parts.append(f'<div class="ad-row"><span class="ad-label">方法</span>'
                         f'<span class="ad-value">{_e(d["method"])}</span></div>')

    elif agent_id == "diagnostician":
        verdict = d.get("verdict", "")
        vcls = {"SUCCESS": "ad-pass", "CODE_BUG": "ad-warn", "LOGIC_FLAW": "ad-fail"}.get(verdict, "")
        parts.append(f'<div class="ad-row"><span class="ad-label">诊断</span>'
                     f'<span class="ad-badge {vcls}">{_e(verdict)}</span></div>')
        if d.get("diagnosis"):
            parts.append(f'<div class="ad-feedback">{_e(d["diagnosis"][:200])}</div>')

    elif agent_id == "poison_generator":
        if d.get("strategy"):
            parts.append(f'<div class="ad-row"><span class="ad-label">扰动策略</span>'
                         f'<span class="ad-value">{_e(d["strategy"])}</span></div>')

    elif agent_id == "publish_evaluator":
        verdict = d.get("verdict", "")
        vcls = "ad-pass" if verdict == "publish_ready" else "ad-fail"
        label = "✅ 达到发表标准" if verdict == "publish_ready" else "⚠️ 未通过"
        parts.append(f'<div class="ad-row"><span class="ad-label">最终裁决</span>'
                     f'<span class="ad-badge {vcls}">{label}</span></div>')

    if not parts:
        return ""
    return '<div class="agent-detail">' + "".join(parts) + "</div>"


def render_agent_card(agent_id: str, icon: str, name: str, desc: str):
    status = st.session_state.agent_status.get(agent_id, "pending")
    if status == "running":
        pill_html = '<span class="agent-status-pill pill-running"><span class="pulse-dot"></span>RUNNING</span>'
    elif status == "done":
        pill_html = '<span class="agent-status-pill pill-done">DONE</span>'
    elif status == "error":
        pill_html = '<span class="agent-status-pill pill-error">ERROR</span>'
    else:
        pill_html = '<span class="agent-status-pill pill-queued">QUEUED</span>'
    detail_html = _render_agent_detail(agent_id) if status == "done" else ""

    st.markdown(f"""
    <div class="agent-card {status}">
        {pill_html}
        <div class="agent-headline">{icon}&nbsp;&nbsp;{name}</div>
        <div class="agent-excerpt">{desc}</div>
        {detail_html}
    </div>
    """, unsafe_allow_html=True)


def render_publish_matrix():
    m = st.session_state.publish_matrix
    items = [
        ("Novelty",        m["novelty"],    "💎"),
        ("Baseline",       m["baseline"],   "📈"),
        ("Robustness",     m["robustness"], "🛡️"),
        ("Explainability", m["explain"],    "📖"),
    ]
    html = '<div class="publish-matrix">'
    for label, passed, icon in items:
        css = "green" if passed else "grey"
        status_span = (
            f'<span class="matrix-pass">PASS</span>'
            if passed else
            f'<span class="matrix-pending">PENDING</span>'
        )
        html += (
            f'<div class="matrix-item {css}">'
            f'<span class="matrix-icon">{icon}</span>'
            f'<span class="matrix-label">{label}</span>'
            f'{status_span}'
            f'</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_failed_ledger():
    import html as _html_mod
    ledger = st.session_state.failed_ledger
    if not ledger:
        st.markdown(
            '<span style="font-size:0.72rem;color:#5a6a7e;letter-spacing:0.5px;">No failures recorded.</span>',
            unsafe_allow_html=True,
        )
        return
    for rec in ledger[-5:]:  # 只显示最近 5 条
        ftype   = _html_mod.escape(str(rec.get("failure_type", "?")))
        summary = _html_mod.escape(str(rec.get("error_summary", "")))
        itr     = _html_mod.escape(str(rec.get("iteration", "?")))
        st.markdown(
            f'<div class="ledger-item">'
            f'<span class="ledger-type">{ftype}</span>'
            f'<span style="color:#8a9ab0;font-size:0.68rem;margin-right:8px;">iter&nbsp;{itr}</span>'
            f'<span style="color:#a0b0c8;">{summary}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_log_console():
    import html as _html_mod
    logs = st.session_state.logs
    level_color = {
        "info":  "#8a9ab8",
        "ok":    "#10b981",
        "warn":  "#d97706",
        "error": "#ef4444",
    }
    level_prefix = {"info": "·", "ok": "·", "warn": "!", "error": "✗"}
    lines = []
    for ts, level, msg in logs[-80:]:
        color  = level_color.get(level, "#8a9ab8")
        prefix = level_prefix.get(level, "·")
        safe_msg = _html_mod.escape(str(msg))
        lines.append(
            f'<span style="color:#7a8a9e;user-select:none">{ts}</span>'
            f'&nbsp;<span style="color:{color}">{prefix}</span>'
            f'&nbsp;<span style="color:{color}">{safe_msg}</span>'
        )
    body = "<br>".join(lines) if lines else '<span style="color:#5a6a7e">waiting for execution...</span>'
    html = (
        '<div class="log-console">'
        '<div class="log-console-header">'
        '<div class="log-dot" style="background:#2a1f1f"></div>'
        '<div class="log-dot" style="background:#1f2217"></div>'
        '<div class="log-dot" style="background:#0e1a15"></div>'
        '<span class="log-console-filename">runtime.log</span>'
        '</div>'
        f'<div class="log-console-body">{body}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_metrics_table():
    import html as _html_mod
    base = st.session_state.baseline_metrics
    prop = st.session_state.proposed_metrics
    pois = st.session_state.poison_metrics
    if not base and not prop:
        st.markdown(
            '<span style="font-size:0.72rem;color:#5a6a7e;">Awaiting experiment results...</span>',
            unsafe_allow_html=True,
        )
        return

    all_keys = sorted(set(list(base.keys()) + list(prop.keys())))
    rows = ""
    for key in all_keys:
        b_val   = base.get(key)
        p_val   = prop.get(key)
        poi_val = pois.get(key)
        b_str   = f"{b_val:.4f}" if b_val is not None else "—"
        p_css   = ""
        if p_val is not None and b_val is not None:
            p_css = "better" if p_val > b_val else "worse"
        p_str   = f"{p_val:.4f}" if p_val is not None else "—"
        poi_str = f"{poi_val:.4f}" if poi_val is not None else "—"
        safe_key = _html_mod.escape(str(key))
        rows += (
            f'<tr>'
            f'<td>{safe_key}</td>'
            f'<td>{b_str}</td>'
            f'<td class="{p_css}">{p_str}</td>'
            f'<td>{poi_str}</td>'
            f'</tr>'
        )

    st.markdown(f"""
    <table class="metrics-table">
        <thead><tr><th>Metric</th><th>Baseline</th><th>Proposed</th><th>Poison</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LLM 回调：把 LLM 调用进度实时写入 queue
# ──────────────────────────────────────────────
def _make_queue_callback(q: queue.Queue):
    """动态构建继承自 BaseCallbackHandler 的回调，避免在模块顶层 import langchain。"""
    from langchain_core.callbacks import BaseCallbackHandler

    class _QueueCallback(BaseCallbackHandler):
        def __init__(self):
            super().__init__()
            self._q = q
            self._buf: list[str] = []
            self._buf_len = 0

        def _flush(self):
            if self._buf:
                self._q.put(("stream_chunk", "".join(self._buf)))
                self._buf = []
                self._buf_len = 0

        def on_llm_start(self, serialized, prompts, **kwargs):
            self._buf = []
            self._buf_len = 0
            self._q.put(("stream_start", None))
            self._q.put(("log", ("info", "  ⌛ LLM 开始推理（等待模型响应）...")))

        def on_llm_new_token(self, token: str, **kwargs):
            self._buf.append(token)
            self._buf_len += len(token)
            if self._buf_len >= 60:   # 每攒够 60 字符推一次，平衡实时性与队列压力
                self._flush()

        def on_llm_end(self, response, **kwargs):
            self._flush()
            try:
                usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
                total = usage.get("total_tokens") or usage.get("total_token_count")
                suffix = f"  共 {total} tokens" if total else ""
            except Exception:
                suffix = ""
            self._q.put(("stream_end", None))
            self._q.put(("log", ("ok", f"  ✓ LLM 响应完成，解析输出中...{suffix}")))

        def on_llm_error(self, error, **kwargs):
            self._flush()
            self._q.put(("stream_end", None))
            self._q.put(("log", ("error", f"  ✗ LLM 出错: {str(error)[:120]}")))

    return _QueueCallback()


# ──────────────────────────────────────────────
# 运行逻辑（在后台线程中执行图）
# ──────────────────────────────────────────────
def _run_graph(research_direction: str, dataset_schema: dict, api_key: str,
               model: str, provider: str, max_loops: int, q: queue.Queue,
               user_data_path: str = ""):
    """在独立线程中运行 LangGraph，通过 queue 向主线程传递状态更新。"""
    try:
        # 检查所有必要依赖
        required = [
            ("pydantic",          "pydantic"),
            ("pydantic-core",     "pydantic_core"),
            ("langchain",         "langchain"),
            ("langchain-openai",  "langchain_openai"),
            ("langchain-anthropic","langchain_anthropic"),
            ("langgraph",         "langgraph"),
        ]
        missing = []
        for pkg, imp in required:
            try:
                __import__(imp)
            except ModuleNotFoundError:
                missing.append(pkg)
        if missing:
            q.put(("error",
                f"缺少依赖包：{', '.join(missing)}\n\n"
                "请在终端运行：\n"
                "  pip install -r requirements.txt\n\n"
                f"或单独安装：\n"
                f"  pip install {' '.join(missing)}"
            ))
            return

        cb = _make_queue_callback(q)

        if provider == "Anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=model, max_tokens=8192, streaming=True, callbacks=[cb])

        elif provider == "OpenAI":
            os.environ["OPENAI_API_KEY"] = api_key
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model, max_tokens=8192, streaming=True, callbacks=[cb])

        elif provider == "MiniMax":
            # MiniMax 提供 OpenAI 兼容接口
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url="https://api.minimax.chat/v1",
                max_tokens=8192,
                streaming=True,
                callbacks=[cb],
            )

        else:
            raise ValueError(f"未知 provider: {provider}")

        q.put(("log", ("info", f"模型: {provider} / {model}")))
        q.put(("log", ("info", f"研究方向: {research_direction[:60]}")))
        q.put(("log", ("info", "正在构建 LangGraph 主图...")))

        from darwinian.state import ResearchState
        from darwinian.graphs.main_graph import build_main_graph

        graph = build_main_graph(llm)
        if user_data_path:
            q.put(("log", ("info", f"用户数据集: {user_data_path}")))
        initial_state = ResearchState(
            research_direction=research_direction,
            dataset_schema=dataset_schema,
            max_outer_loops=max_loops,
            user_data_path=user_data_path,
        )

        q.put(("log", ("ok", "图构建完成，开始执行...")))

        # 使用 stream 逐节点获取更新
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            q.put(("chunk", chunk))

        q.put(("done", None))

    except Exception as e:
        import traceback
        q.put(("error", traceback.format_exc()))


def _add_log(level: str, message: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append((ts, level, message))
    if len(st.session_state.logs) > 200:
        st.session_state.logs = st.session_state.logs[-200:]


def _apply_chunk(chunk: dict):
    """将单个 stream chunk 更新到 session_state。"""
    for node_name, update in chunk.items():
        # 标记 Agent 节点状态
        agent_id = _node_to_agent_id(node_name)
        if agent_id:
            st.session_state.agent_status[agent_id] = "done"
            _extract_agent_output(agent_id, update)

        if not isinstance(update, dict):
            continue

        # failed_ledger
        if "failed_ledger" in update:
            st.session_state.failed_ledger = [
                r.model_dump() if hasattr(r, "model_dump") else r
                for r in update["failed_ledger"]
            ]

        # publish_matrix
        if "publish_matrix" in update:
            pm = update["publish_matrix"]
            if hasattr(pm, "model_dump"):
                pm = pm.model_dump()
            st.session_state.publish_matrix = {
                "novelty":   pm.get("novelty_passed", False),
                "baseline":  pm.get("baseline_improved", False),
                "robustness":pm.get("robustness_passed", False),
                "explain":   pm.get("explainability_generated", False),
            }

        # experiment_result
        if "experiment_result" in update:
            er = update["experiment_result"]
            if hasattr(er, "model_dump"):
                er = er.model_dump()
            if er:
                st.session_state.baseline_metrics = er.get("baseline_metrics", {})
                st.session_state.proposed_metrics = er.get("proposed_metrics", {})

        # poison_test_result
        if "poison_test_result" in update:
            ptr = update["poison_test_result"]
            if hasattr(ptr, "model_dump"):
                ptr = ptr.model_dump()
            if ptr:
                st.session_state.poison_metrics = ptr.get("perturbed_metrics", {})

        # final_verdict & report
        if "final_verdict" in update and update["final_verdict"]:
            fv = update["final_verdict"]
            st.session_state.final_verdict = fv.value if hasattr(fv, "value") else str(fv)
        if "final_report" in update and update["final_report"]:
            st.session_state.final_report = update["final_report"]

        # outer_loop_count
        if "outer_loop_count" in update:
            st.session_state.outer_loop = update["outer_loop_count"]

        # critic_verdict
        if "critic_verdict" in update and update["critic_verdict"]:
            cv = update["critic_verdict"]
            st.session_state.critic_verdict = cv.value if hasattr(cv, "value") else str(cv)

        # current_hypothesis
        if "current_hypothesis" in update and update["current_hypothesis"]:
            h = update["current_hypothesis"]
            if hasattr(h, "model_dump"):
                h = h.model_dump()
            st.session_state.current_hypothesis = h.get("core_problem", "") if h else ""


def _node_to_agent_id(node_name: str) -> str | None:
    mapping = {
        "bottleneck_miner":    "bottleneck_miner",
        "hypothesis_generator":"hypothesis_generator",
        "theoretical_critic":  "theoretical_critic",
        "code_architect":      "code_architect",
        "diagnostician":       "diagnostician",
        "poison_generator":    "poison_generator",
        "publish_evaluator":   "publish_evaluator",
    }
    return mapping.get(node_name)


def _extract_agent_output(agent_id: str, update: Any):
    """从节点更新中提取结构化数据写入 agent_detail。"""
    if not isinstance(update, dict):
        return
    det = st.session_state.agent_detail

    if agent_id == "bottleneck_miner" and "current_hypothesis" in update:
        h = update["current_hypothesis"]
        if hasattr(h, "core_problem"):
            det[agent_id] = {
                "core_problem": h.core_problem,
                "evidence": list(h.literature_support) if hasattr(h, "literature_support") else [],
            }

    elif agent_id == "hypothesis_generator" and "current_hypothesis" in update:
        h = update["current_hypothesis"]
        if hasattr(h, "abstraction_tree") and h.abstraction_tree:
            det[agent_id] = {
                "hypotheses": [
                    {"name": b.name, "desc": getattr(b, "description", "")[:100]}
                    for b in h.abstraction_tree[:4]
                ]
            }

    elif agent_id == "theoretical_critic" and "critic_verdict" in update:
        v = update["critic_verdict"]
        det[agent_id] = {
            "verdict": v.value if hasattr(v, "value") else str(v),
            "feedback": update.get("critic_feedback", ""),
        }

    elif agent_id == "code_architect" and "experiment_code" in update:
        ec = update["experiment_code"]
        det[agent_id] = {
            "dataset": getattr(ec, "dataset_name", "") or getattr(ec, "dataset_loader_code", "")[:60],
            "method": getattr(ec, "proposed_method_code", "")[:80],
        }

    elif agent_id == "diagnostician" and "experiment_result" in update:
        er = update["experiment_result"]
        verdict = getattr(er, "execution_verdict", None)
        det[agent_id] = {
            "verdict": verdict.value if hasattr(verdict, "value") else str(verdict),
            "diagnosis": getattr(er, "diagnosis", ""),
        }

    elif agent_id == "poison_generator" and "poison_test_result" in update:
        ptr = update["poison_test_result"]
        det[agent_id] = {
            "strategy": getattr(ptr, "perturbation_strategy", ""),
        }

    elif agent_id == "publish_evaluator" and "final_verdict" in update:
        fv = update["final_verdict"]
        det[agent_id] = {
            "verdict": fv.value if hasattr(fv, "value") else str(fv),
        }


# ──────────────────────────────────────────────
# 侧边栏
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 实验配置")
    st.divider()

    research_direction = st.text_area(
        "研究方向",
        value="图神经网络在分子属性预测中的应用",
        height=90,
        placeholder="描述你想探索的研究方向...",
    )

    with st.expander("📂 数据集（可选）", expanded=False):
        st.caption("不填则系统自动搜索 HuggingFace 公开数据集")

        uploaded_file = st.file_uploader(
            "上传数据集",
            type=["csv", "parquet", "npz", "npy", "json", "jsonl"],
            help="支持 CSV / Parquet / NumPy / JSON 格式",
        )

        # 保存上传文件到临时目录，持久化路径到 session_state
        if uploaded_file is not None:
            import tempfile, pathlib
            upload_dir = pathlib.Path(tempfile.gettempdir()) / "darwinian_uploads"
            upload_dir.mkdir(exist_ok=True)
            save_path = upload_dir / uploaded_file.name
            save_path.write_bytes(uploaded_file.getvalue())
            st.session_state["_upload_path"] = str(save_path)
            st.success(f"已上传：{uploaded_file.name}", icon="✅")
        elif "_upload_path" not in st.session_state:
            st.session_state["_upload_path"] = ""

        if st.session_state.get("_upload_path"):
            if st.button("✕ 清除上传文件", use_container_width=False):
                st.session_state["_upload_path"] = ""
                st.rerun()

        st.caption("或填写数据集描述辅助搜索：")
        dataset_schema_str = st.text_area(
            "dataset_schema",
            value="",
            height=80,
            placeholder='{"task": "classification", "domain": "molecular", "metric": "ROC-AUC"}',
            label_visibility="collapsed",
        )

    st.divider()
    st.markdown("### 🔑 模型配置")

    PROVIDER_MODELS = {
        "Anthropic": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        "OpenAI":    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "MiniMax":   ["MiniMax-Text-01", "abab6.5s-chat", "MiniMax-M2.7"],
    }
    PROVIDER_ENV = {
        "Anthropic": "ANTHROPIC_API_KEY",
        "OpenAI":    "OPENAI_API_KEY",
        "MiniMax":   "MINIMAX_API_KEY",
    }

    provider_choice = st.selectbox("服务商", list(PROVIDER_MODELS.keys()), index=0)
    model_choice = st.selectbox("模型", PROVIDER_MODELS[provider_choice], index=0)

    env_key = PROVIDER_ENV[provider_choice]
    api_key = st.text_input(
        "API Key",
        value=os.getenv(env_key, ""),
        type="password",
        placeholder=f"{env_key}",
    )

    max_loops = st.slider("最大外层循环次数", min_value=1, max_value=10, value=5)

    st.divider()

    can_run = bool(research_direction.strip() and api_key.strip())

    if st.button(
        "🚀 开始研究" if not st.session_state.running else "⏳ 研究进行中...",
        use_container_width=True,
        type="primary",
        disabled=st.session_state.running or not can_run,
    ):
        # 重置状态
        for k, v in {
            "running": True, "finished": False,
            "agent_status": {a[0]: "pending" for a in AGENTS},
            "agent_output": {a[0]: "" for a in AGENTS},
            "failed_ledger": [], "publish_matrix": {"novelty": False, "baseline": False, "robustness": False, "explain": False},
            "baseline_metrics": {}, "proposed_metrics": {}, "poison_metrics": {},
            "final_verdict": None, "final_report": "",
            "outer_loop": 0, "max_loops": max_loops,
            "log_queue": queue.Queue(),
            "current_hypothesis": "", "critic_verdict": "",
            "logs": [], "_error": "",
            "agent_detail": {a[0]: {} for a in AGENTS},
            "current_stream": "",
            "stream_history": [],
            "_stream_agent": "",
            "_stream_agent_task": "",
        }.items():
            st.session_state[k] = v

        # 解析 dataset_schema
        try:
            ds = json.loads(dataset_schema_str)
        except Exception:
            ds = {}

        user_data_path = st.session_state.get("_upload_path", "")

        # 启动后台线程
        t = threading.Thread(
            target=_run_graph,
            args=(research_direction, ds, api_key, model_choice, provider_choice,
                  max_loops, st.session_state.log_queue),
            kwargs={"user_data_path": user_data_path},
            daemon=True,
        )
        t.start()
        st.rerun()

    if not can_run:
        st.caption("⚠️ 请填写研究方向和 API Key")

    st.divider()
    # 循环进度
    if st.session_state.outer_loop > 0:
        st.markdown(
            f'第 <span class="loop-badge">{st.session_state.outer_loop} / {st.session_state.max_loops}</span> 轮',
            unsafe_allow_html=True,
        )
    if st.session_state.current_hypothesis:
        st.markdown("**当前核心问题**")
        st.caption(st.session_state.current_hypothesis[:200])
    if st.session_state.critic_verdict:
        color = {"PASS": "🟢", "MATH_ERROR": "🔴", "NOT_NOVEL": "🟡"}.get(
            st.session_state.critic_verdict, "⬜"
        )
        st.caption(f"审查结论：{color} {st.session_state.critic_verdict}")


# ──────────────────────────────────────────────
# 主工作区
# ──────────────────────────────────────────────
render_header()

# 从 queue 拉取更新（每次 rerun 时执行）
if st.session_state.running:
    q = st.session_state.log_queue
    processed = 0
    while not q.empty() and processed < 20:
        msg_type, payload = q.get_nowait()
        processed += 1
        if msg_type == "chunk":
            for node_name in payload:
                aid = _node_to_agent_id(node_name)
                if aid and st.session_state.agent_status[aid] == "pending":
                    st.session_state.agent_status[aid] = "running"
                    label = next((n for a, _, n, _ in AGENTS if a == aid), node_name)
                    _add_log("info", f"→ 开始执行 {label}")
                elif aid and st.session_state.agent_status[aid] == "running":
                    label = next((n for a, _, n, _ in AGENTS if a == aid), node_name)
                    _add_log("ok", f"✓ 完成 {label}")
            _apply_chunk(payload)
            # 节点完成后追加摘要日志
            for node_name, update in payload.items():
                if not isinstance(update, dict):
                    continue
                if "critic_verdict" in update and update["critic_verdict"]:
                    v = update["critic_verdict"]
                    _add_log("info", f"  审查结论: {v.value if hasattr(v,'value') else v}")
                if "current_hypothesis" in update and update["current_hypothesis"]:
                    h = update["current_hypothesis"]
                    if hasattr(h, "core_problem") and h.core_problem:
                        _add_log("info", f"  核心问题: {h.core_problem[:80]}")
                if "experiment_result" in update and update["experiment_result"]:
                    er = update["experiment_result"]
                    if hasattr(er, "execution_verdict") and er.execution_verdict:
                        _add_log("info", f"  执行结果: {er.execution_verdict.value if hasattr(er.execution_verdict,'value') else er.execution_verdict}")
                if "final_verdict" in update and update["final_verdict"]:
                    fv = update["final_verdict"]
                    _add_log("ok" if str(fv) == "publish_ready" else "warn",
                             f"  最终裁决: {fv.value if hasattr(fv,'value') else fv}")
                if "failed_ledger" in update:
                    ledger = update["failed_ledger"]
                    if ledger:
                        last = ledger[-1]
                        summary = last.error_summary if hasattr(last, "error_summary") else str(last)
                        _add_log("warn", f"  写入账本: {summary[:80]}")
        elif msg_type == "stream_start":
            # 记录当前正在运行的 agent 名称及任务说明
            running_aid  = next(
                (aid for aid, _, _, _ in AGENTS
                 if st.session_state.agent_status.get(aid) == "running"),
                "",
            )
            running_name = next(
                (name for aid, _, name, _ in AGENTS
                 if st.session_state.agent_status.get(aid) == "running"),
                "LLM",
            )
            task_desc = AGENT_TASKS.get(running_aid, "LLM 推理中")
            st.session_state._stream_agent      = running_name
            st.session_state._stream_agent_task = task_desc
            st.session_state.current_stream     = ""
            _add_log("info", f"  → {task_desc}")
        elif msg_type == "stream_chunk":
            st.session_state.current_stream += payload
        elif msg_type == "stream_end":
            # 将完成的输出存入历史，拆分 think / response
            import re as _re2
            content = st.session_state.current_stream
            if content.strip():
                tm = _re2.search(r"<think>([\s\S]*?)</think>([\s\S]*)", content)
                if tm:
                    think_part = tm.group(1).strip()
                    resp_part  = tm.group(2).strip()
                else:
                    think_part = ""
                    resp_part  = content.strip()
                st.session_state.stream_history.append({
                    "agent":    st.session_state.get("_stream_agent", "LLM"),
                    "task":     st.session_state.get("_stream_agent_task", ""),
                    "think":    think_part,
                    "response": resp_part,
                })
            st.session_state.current_stream = ""
        elif msg_type == "log":
            _add_log(payload[0], payload[1])
        elif msg_type == "done":
            st.session_state.running = False
            st.session_state.finished = True
            _add_log("ok", "━━ 研究流程结束 ━━")
        elif msg_type == "error":
            st.session_state.running = False
            st.session_state.finished = True
            st.session_state["_error"] = payload
            _add_log("error", f"运行出错: {payload.splitlines()[-1] if payload else '未知错误'}")

# ── 布局：单列纵向 ──
def _section(title: str):
    import html as _html_mod
    safe_title = _html_mod.escape(str(title)).upper()
    st.markdown(f'<div class="section-title">{safe_title}</div>', unsafe_allow_html=True)

def _stream_block(stream_text: str):
    import re as _re
    think_match = _re.search(r"<think>([\s\S]*?)</think>([\s\S]*)", stream_text)
    base_style = ("font-size:0.85rem;white-space:pre-wrap;font-family:'JetBrains Mono','Fira Code',monospace;"
                  "max-height:320px;overflow-y:auto;line-height:1.75;padding:14px 16px;"
                  "background:#06080f;border:1px solid #1a2035;border-radius:8px;")
    if think_match:
        think_part = think_match.group(1).strip()
        rest_part  = think_match.group(2).strip()
        with st.expander("🤔 Reasoning trace", expanded=False):
            st.markdown(f'<div style="{base_style}color:#a0b0c8">{think_part}</div>',
                        unsafe_allow_html=True)
        if rest_part:
            st.markdown(f'<div style="{base_style}color:#b8c8dc">{rest_part}</div>',
                        unsafe_allow_html=True)
    else:
        in_think = "<think>" in stream_text and "</think>" not in stream_text
        color = "#a0b0c8" if in_think else "#b8c8dc"
        if in_think:
            st.markdown('<span style="font-size:0.72rem;color:#5a6a7e">⟳ reasoning...</span>',
                        unsafe_allow_html=True)
        st.markdown(f'<div style="{base_style}color:{color}">{stream_text[-3000:]}</div>',
                    unsafe_allow_html=True)


_section("AGENT PIPELINE")
for agent_id, icon, name, desc in AGENTS:
    render_agent_card(agent_id, icon, name, desc)

stream_text    = st.session_state.get("current_stream", "")
stream_history = st.session_state.get("stream_history", [])

if stream_text or stream_history:
    _section("LLM OUTPUT TRACE")
    mono = "'JetBrains Mono','Fira Code','Menlo',monospace"
    box  = (f"font-size:0.85rem;white-space:pre-wrap;font-family:{mono};"
            "max-height:340px;overflow-y:auto;line-height:1.75;"
            "padding:14px 16px;background:#06080f;border:1px solid #1a2035;border-radius:8px;")

    # 已完成的历史条目（全部折叠）
    for i, entry in enumerate(stream_history):
        task_suffix = f"  —  {entry['task']}" if entry.get("task") else ""
        label = f"#{i+1}  {entry['agent']}{task_suffix}"
        with st.expander(label, expanded=False):
            if entry["think"]:
                st.markdown(
                    f'<div style="{box}color:#7a8a9e"><span style="font-size:0.72rem;'
                    f'color:#5a6a7e;display:block;margin-bottom:8px">⟳ Reasoning</span>'
                    f'{entry["think"]}</div>',
                    unsafe_allow_html=True,
                )
            if entry["response"]:
                st.markdown(
                    f'<div style="{box}color:#b8c8dc;margin-top:8px">{entry["response"]}</div>',
                    unsafe_allow_html=True,
                )

    # 正在进行的实时流（展开）
    if stream_text:
        _live_task = st.session_state.get("_stream_agent_task", "")
        _live_label = (f"⟳  {st.session_state.get('_stream_agent','LLM')}  —  {_live_task}  — 输出中..."
                       if _live_task else
                       f"⟳  {st.session_state.get('_stream_agent','LLM')}  — 输出中...")
        with st.expander(_live_label, expanded=True):
            _stream_block(stream_text)

_section("RUNTIME LOG")
render_log_console()

if "_error" in st.session_state and st.session_state["_error"]:
    with st.expander("✗ Error details", expanded=True):
        st.code(st.session_state["_error"], language="python")

_section("PUBLISH CHECKLIST")
render_publish_matrix()

if st.session_state.final_verdict:
    verdict = st.session_state.final_verdict
    if verdict == "publish_ready":
        st.success("Research meets publication standard.", icon="✅")
    else:
        st.warning("Robustness test failed — logged to cognitive ledger.", icon="🔄")

_section("EXPERIMENT METRICS")
render_metrics_table()

ledger_count = len(st.session_state.failed_ledger)
_section(f"COGNITIVE LEDGER  ·  {ledger_count} records")
render_failed_ledger()

# ── 研究报告（全宽展示）──
if st.session_state.final_report:
    st.divider()
    st.markdown("### 📄 研究报告")
    with st.container():
        st.markdown(st.session_state.final_report)
    st.download_button(
        label="⬇️ 下载报告 (.md)",
        data=st.session_state.final_report,
        file_name=f"darwinian_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
    )

# ── 自动刷新 ──
if st.session_state.running:
    if _HAS_AUTOREFRESH:
        _st_autorefresh(interval=1500, key="running_refresh")
    else:
        st.warning(
            "⚠️ 未安装 streamlit-autorefresh，页面无法自动刷新。\n\n"
            "请运行：`pip install streamlit-autorefresh`，然后重启 Streamlit。\n\n"
            "或点击下方按钮手动刷新。",
        )
        if st.button("🔄 手动刷新", key="manual_refresh"):
            st.rerun()

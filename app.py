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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── 全局 ── */
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.main .block-container { padding-top: 1rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background: #0a0d14 !important; border-right: 1px solid #1e2330; }

/* ── 顶栏 ── */
.app-header {
    position: relative;
    padding: 20px 0 16px;
    margin-bottom: 28px;
}
.app-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4, transparent);
    border-radius: 1px;
}
.app-header-inner { display: flex; align-items: center; gap: 16px; }
.app-header-logo {
    width: 48px; height: 48px; border-radius: 12px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; box-shadow: 0 0 20px rgba(99,102,241,0.4);
    flex-shrink: 0;
}
.app-header h1 {
    margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.app-header .subtitle {
    color: #475569; font-size: 0.82rem; margin-top: 3px; letter-spacing: 0.3px;
}
.app-header .badge-row { display: flex; gap: 8px; margin-top: 6px; flex-wrap: wrap; }
.hbadge {
    font-size: 0.68rem; padding: 2px 8px; border-radius: 20px; font-weight: 500;
    border: 1px solid #1e2330; color: #64748b; background: #0f1420;
}

/* ── Section 标题 ── */
.section-title {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: #475569;
    margin: 20px 0 12px; display: flex; align-items: center; gap: 8px;
}
.section-title::after {
    content: ''; flex: 1; height: 1px; background: #1e2330;
}

/* ── Agent 卡片 ── */
.agent-card {
    display: flex; align-items: flex-start; gap: 14px;
    padding: 14px 16px;
    border-radius: 10px;
    margin-bottom: 6px;
    border: 1px solid #1e2330;
    background: #0d1017;
    transition: all 0.25s ease;
    position: relative; overflow: hidden;
}
.agent-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px; background: transparent;
    transition: background 0.25s;
}
.agent-card.running::before  { background: linear-gradient(180deg, #6366f1, #8b5cf6); }
.agent-card.done::before     { background: linear-gradient(180deg, #10b981, #059669); }
.agent-card.error::before    { background: #ef4444; }

.agent-card.running {
    border-color: rgba(99,102,241,0.35);
    background: linear-gradient(135deg, #0d1017 0%, #0f1128 100%);
    box-shadow: 0 0 0 1px rgba(99,102,241,0.1), 0 4px 24px rgba(99,102,241,0.06);
}
.agent-card.running .agent-name { color: #a5b4fc; }
.agent-card.done   { border-color: rgba(16,185,129,0.25); }
.agent-card.done .agent-name   { color: #6ee7b7; }
.agent-card.error  { border-color: rgba(239,68,68,0.3); }
.agent-card.pending { opacity: 0.38; }

.agent-icon {
    font-size: 1.25rem; line-height: 1;
    width: 36px; height: 36px; border-radius: 8px;
    background: #131820; display: flex; align-items: center;
    justify-content: center; flex-shrink: 0; margin-top: 1px;
    border: 1px solid #1e2330;
}
.agent-card.running .agent-icon { background: #131828; border-color: rgba(99,102,241,0.3); }
.agent-card.done    .agent-icon { background: #0b1a14; border-color: rgba(16,185,129,0.2); }

.agent-name { font-weight: 600; font-size: 0.88rem; color: #cbd5e1; letter-spacing: 0.1px; }
.agent-desc { font-size: 0.76rem; color: #334155; margin-top: 3px; }
.agent-status-row { display: flex; align-items: center; gap: 8px; }

/* running 脉冲点 */
.pulse-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #6366f1;
    animation: pulsedot 1.4s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes pulsedot {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.7); }
}

/* ── Agent 详情 ── */
.agent-detail {
    margin-top: 10px; padding: 10px 12px;
    background: rgba(6,9,15,0.8); border-radius: 7px;
    border: 1px solid #1a2035; font-size: 0.79rem;
}
.ad-row { display: flex; align-items: flex-start; gap: 8px; margin-bottom: 6px; }
.ad-label {
    color: #334155; white-space: nowrap; min-width: 56px;
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.6px;
    padding-top: 2px;
}
.ad-value { color: #94a3b8; line-height: 1.5; }
.ad-badge {
    padding: 2px 9px; border-radius: 5px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.3px;
}
.ad-pass { background: rgba(16,185,129,0.12); color: #34d399; border: 1px solid rgba(16,185,129,0.25); }
.ad-fail { background: rgba(239,68,68,0.1);   color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
.ad-warn { background: rgba(251,191,36,0.1);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
.ad-feedback { color: #64748b; margin-top: 5px; line-height: 1.6; font-size: 0.78rem; }
.ad-ev {
    color: #334155; margin-bottom: 3px; padding-left: 10px;
    border-left: 2px solid #1e2330;
}
.ad-hyp {
    margin-bottom: 7px; padding: 7px 10px;
    background: #0a0d14; border-radius: 5px; border: 1px solid #1a2035;
}
.ad-hyp-name { font-weight: 600; color: #818cf8; display: block; margin-bottom: 2px; }
.ad-hyp-desc { color: #475569; font-size: 0.76rem; line-height: 1.5; }

/* ── 发表矩阵 ── */
.publish-matrix {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 10px; margin-top: 4px;
}
.matrix-item {
    padding: 16px 14px; border-radius: 10px;
    border: 1px solid #1e2330; text-align: center;
    background: #0a0d14; position: relative; overflow: hidden;
    transition: all 0.3s;
}
.matrix-item.green {
    border-color: rgba(16,185,129,0.3);
    background: linear-gradient(135deg, #061410 0%, #091a13 100%);
    box-shadow: 0 0 20px rgba(16,185,129,0.05);
}
.matrix-item.grey { opacity: 0.5; }
.matrix-label {
    font-size: 0.72rem; font-weight: 500; letter-spacing: 0.6px;
    text-transform: uppercase; color: #475569; margin-bottom: 8px;
}
.matrix-icon { font-size: 1.6rem; display: block; margin-bottom: 4px; }
.matrix-status {
    font-size: 0.72rem; font-weight: 600;
    color: #1e2330;
}
.matrix-item.green .matrix-status { color: #34d399; }

/* ── 日志控制台 ── */
.log-console {
    background: #060810; border: 1px solid #1a2035;
    border-radius: 10px; overflow: hidden;
}
.log-console-header {
    padding: 8px 14px; background: #0a0d16;
    border-bottom: 1px solid #1a2035;
    display: flex; align-items: center; gap: 8px;
}
.log-dot { width: 8px; height: 8px; border-radius: 50%; }
.log-console-body {
    padding: 10px 14px; height: 220px; overflow-y: auto;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.75rem; line-height: 1.75;
}

/* ── 指标表 ── */
.metrics-table {
    width: 100%; border-collapse: collapse; font-size: 0.82rem;
}
.metrics-table thead tr {
    background: #080b12;
}
.metrics-table th {
    text-align: left; padding: 9px 14px;
    border-bottom: 1px solid #1e2330;
    color: #334155; font-weight: 500;
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.8px;
}
.metrics-table td { padding: 9px 14px; border-bottom: 1px solid #10151f; color: #64748b; }
.metrics-table tbody tr:hover td { background: #080b12; }
.metrics-table .better { color: #34d399; font-weight: 600; }
.metrics-table .worse  { color: #f87171; }
.metrics-table td:first-child { color: #94a3b8; font-weight: 500; }

/* ── 认知账本 ── */
.ledger-item {
    padding: 10px 14px; border-radius: 8px;
    border: 1px solid #1e2330; border-left: 3px solid #ef4444;
    background: #080b12; margin-bottom: 6px;
    font-size: 0.79rem; color: #64748b;
}
.ledger-type {
    font-weight: 600; font-size: 0.68rem; letter-spacing: 0.6px;
    text-transform: uppercase; color: #f87171; margin-right: 8px;
    padding: 1px 6px; background: rgba(239,68,68,0.1);
    border-radius: 4px;
}

/* ── 循环徽章 ── */
.loop-badge {
    display: inline-block; padding: 3px 12px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    background: linear-gradient(135deg, #0f1428, #13192e);
    color: #818cf8; border: 1px solid rgba(99,102,241,0.25);
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ──────────────────────────────────────────────
# 辅助渲染函数
# ──────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="app-header">
        <div class="app-header-inner">
            <div class="app-header-logo">🧬</div>
            <div>
                <h1>Darwinian</h1>
                <div class="subtitle">State-Driven Multi-Agent Automated Research System</div>
                <div class="badge-row">
                    <span class="hbadge">LangGraph</span>
                    <span class="hbadge">7 Agents</span>
                    <span class="hbadge">Adversarial Testing</span>
                    <span class="hbadge">Auto-Publish Eval</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_agent_detail(agent_id: str) -> str:
    """把 agent_detail[agent_id] 渲染为 HTML 片段。"""
    d = st.session_state.agent_detail.get(agent_id, {})
    if not d:
        return ""
    parts = []

    if agent_id == "bottleneck_miner":
        if d.get("core_problem"):
            parts.append(f'<div class="ad-row"><span class="ad-label">核心矛盾</span>'
                         f'<span class="ad-value">{d["core_problem"]}</span></div>')
        for ev in d.get("evidence", [])[:3]:
            parts.append(f'<div class="ad-ev">📄 {ev}</div>')

    elif agent_id == "hypothesis_generator":
        for h in d.get("hypotheses", []):
            parts.append(f'<div class="ad-hyp"><span class="ad-hyp-name">{h["name"]}</span>'
                         f'<span class="ad-hyp-desc">{h["desc"]}</span></div>')

    elif agent_id == "theoretical_critic":
        verdict = d.get("verdict", "")
        vcls = {"PASS": "ad-pass", "MATH_ERROR": "ad-fail", "NOT_NOVEL": "ad-warn"}.get(verdict, "")
        parts.append(f'<div class="ad-row"><span class="ad-label">裁决</span>'
                     f'<span class="ad-badge {vcls}">{verdict}</span></div>')
        if d.get("feedback"):
            parts.append(f'<div class="ad-feedback">{d["feedback"][:200]}</div>')

    elif agent_id == "code_architect":
        if d.get("dataset"):
            parts.append(f'<div class="ad-row"><span class="ad-label">数据集</span>'
                         f'<span class="ad-value">{d["dataset"]}</span></div>')
        if d.get("method"):
            parts.append(f'<div class="ad-row"><span class="ad-label">方法</span>'
                         f'<span class="ad-value">{d["method"]}</span></div>')

    elif agent_id == "diagnostician":
        verdict = d.get("verdict", "")
        vcls = {"SUCCESS": "ad-pass", "CODE_BUG": "ad-warn", "LOGIC_FLAW": "ad-fail"}.get(verdict, "")
        parts.append(f'<div class="ad-row"><span class="ad-label">诊断</span>'
                     f'<span class="ad-badge {vcls}">{verdict}</span></div>')
        if d.get("diagnosis"):
            parts.append(f'<div class="ad-feedback">{d["diagnosis"][:200]}</div>')

    elif agent_id == "poison_generator":
        if d.get("strategy"):
            parts.append(f'<div class="ad-row"><span class="ad-label">扰动策略</span>'
                         f'<span class="ad-value">{d["strategy"]}</span></div>')

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
        status_html = '<span class="pulse-dot"></span>'
    elif status == "done":
        status_html = '<span style="color:#34d399;font-size:0.75rem">✓ 完成</span>'
    elif status == "error":
        status_html = '<span style="color:#f87171;font-size:0.75rem">✗ 出错</span>'
    else:
        status_html = ""
    detail_html = _render_agent_detail(agent_id) if status == "done" else ""

    st.markdown(f"""
    <div class="agent-card {status}">
        <div class="agent-icon">{icon}</div>
        <div style="flex:1;min-width:0">
            <div class="agent-status-row">
                <span class="agent-name">{name}</span>
                {status_html}
            </div>
            <div class="agent-desc">{desc}</div>
            {detail_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_publish_matrix():
    m = st.session_state.publish_matrix
    items = [
        ("新颖性",   m["novelty"],   "💎", "Novelty"),
        ("基准提升", m["baseline"],  "📈", "Baseline"),
        ("鲁棒性",   m["robustness"],"🛡️", "Robustness"),
        ("可解释性", m["explain"],   "📖", "Explainability"),
    ]
    html = '<div class="publish-matrix">'
    for label, passed, icon, _ in items:
        css = "green" if passed else "grey"
        status = '<span style="color:#34d399;font-size:0.7rem;font-weight:600">PASSED</span>' if passed else '<span style="color:#1e2330;font-size:0.7rem">PENDING</span>'
        html += (f'<div class="matrix-item {css}">'
                 f'<span class="matrix-icon">{icon}</span>'
                 f'<div class="matrix-label">{label}</div>'
                 f'<div class="matrix-status">{status}</div>'
                 f'</div>')
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_failed_ledger():
    ledger = st.session_state.failed_ledger
    if not ledger:
        st.caption("暂无失败记录")
        return
    for rec in ledger[-5:]:  # 只显示最近 5 条
        st.markdown(
            f'<div class="ledger-item">'
            f'<span class="ledger-type">[{rec.get("failure_type","?")}]</span>'
            f'第 {rec.get("iteration","?")} 轮 — {rec.get("error_summary","")}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_log_console():
    logs = st.session_state.logs
    level_color = {"info": "#475569", "ok": "#34d399", "warn": "#fbbf24", "error": "#f87171"}
    level_prefix = {"info": "·", "ok": "✓", "warn": "!", "error": "✗"}
    lines = []
    for ts, level, msg in logs[-80:]:
        color  = level_color.get(level, "#475569")
        prefix = level_prefix.get(level, "·")
        lines.append(
            f'<span style="color:#1e293b;user-select:none">{ts}</span> '
            f'<span style="color:{color};margin-right:6px">{prefix}</span>'
            f'<span style="color:{color if level != "info" else "#475569"}">{msg}</span>'
        )
    body = "<br>".join(lines) if lines else '<span style="color:#1e293b">waiting for execution...</span>'
    html = (
        '<div class="log-console">'
        '<div class="log-console-header">'
        '<div class="log-dot" style="background:#ef4444"></div>'
        '<div class="log-dot" style="background:#f59e0b"></div>'
        '<div class="log-dot" style="background:#10b981"></div>'
        '<span style="font-size:0.68rem;color:#334155;margin-left:6px;font-family:monospace">runtime.log</span>'
        '</div>'
        f'<div class="log-console-body">{body}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_metrics_table():
    base = st.session_state.baseline_metrics
    prop = st.session_state.proposed_metrics
    pois = st.session_state.poison_metrics
    if not base and not prop:
        st.caption("等待实验结果...")
        return

    all_keys = sorted(set(list(base.keys()) + list(prop.keys())))
    rows = ""
    for key in all_keys:
        b_val = base.get(key)
        p_val = prop.get(key)
        poi_val = pois.get(key)
        b_str = f"{b_val:.4f}" if b_val is not None else "—"
        p_css = ""
        if p_val is not None and b_val is not None:
            p_css = "better" if p_val > b_val else "worse"
        p_str = f"{p_val:.4f}" if p_val is not None else "—"
        poi_str = f"{poi_val:.4f}" if poi_val is not None else "—"
        rows += f'<tr><td>{key}</td><td>{b_str}</td><td class="{p_css}">{p_str}</td><td>{poi_str}</td></tr>'

    st.markdown(f"""
    <table class="metrics-table">
        <thead><tr><th>指标</th><th>Baseline</th><th>Proposed</th><th>毒药数据</th></tr></thead>
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
               model: str, provider: str, max_loops: int, q: queue.Queue):
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
        initial_state = ResearchState(
            research_direction=research_direction,
            dataset_schema=dataset_schema,
            max_outer_loops=max_loops,
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

    with st.expander("数据集描述（可选）", expanded=False):
        st.caption("留空则由 Agent 4 根据假设自动选择合适的公开数据集")
        dataset_schema_str = st.text_area(
            "dataset_schema",
            value="",
            height=120,
            placeholder='{"type": "tabular", "task": "classification", "metric": "ROC-AUC"}',
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
        }.items():
            st.session_state[k] = v

        # 解析 dataset_schema
        try:
            ds = json.loads(dataset_schema_str)
        except Exception:
            ds = {}

        # 启动后台线程
        t = threading.Thread(
            target=_run_graph,
            args=(research_direction, ds, api_key, model_choice, provider_choice, max_loops, st.session_state.log_queue),
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
            st.session_state.current_stream = ""
        elif msg_type == "stream_chunk":
            st.session_state.current_stream += payload
        elif msg_type == "stream_end":
            pass  # 保留最终内容，由下一个 stream_start 清空
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

# ── 布局：左 Agent 流程 | 右 结果面板 ──
col_left, col_right = st.columns([1, 1], gap="large")

def _section(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def _stream_block(stream_text: str):
    import re as _re
    think_match = _re.search(r"<think>([\s\S]*?)</think>([\s\S]*)", stream_text)
    base_style = ("font-size:0.77rem;white-space:pre-wrap;font-family:'JetBrains Mono','Fira Code',monospace;"
                  "max-height:280px;overflow-y:auto;line-height:1.7;padding:12px 14px;"
                  "background:#06080f;border:1px solid #1a2035;border-radius:8px;")
    if think_match:
        think_part = think_match.group(1).strip()
        rest_part  = think_match.group(2).strip()
        with st.expander("🤔 Reasoning trace", expanded=False):
            st.markdown(f'<div style="{base_style}color:#334155">{think_part}</div>',
                        unsafe_allow_html=True)
        if rest_part:
            st.markdown(f'<div style="{base_style}color:#64748b">{rest_part}</div>',
                        unsafe_allow_html=True)
    else:
        in_think = "<think>" in stream_text and "</think>" not in stream_text
        color = "#334155" if in_think else "#64748b"
        if in_think:
            st.markdown('<span style="font-size:0.72rem;color:#475569">⟳ reasoning...</span>',
                        unsafe_allow_html=True)
        st.markdown(f'<div style="{base_style}color:{color}">{stream_text[-3000:]}</div>',
                    unsafe_allow_html=True)


with col_left:
    _section("AGENT PIPELINE")
    for agent_id, icon, name, desc in AGENTS:
        render_agent_card(agent_id, icon, name, desc)

    stream_text = st.session_state.get("current_stream", "")
    if stream_text:
        _section("LIVE OUTPUT")
        _stream_block(stream_text)

    _section("RUNTIME LOG")
    render_log_console()

    if "_error" in st.session_state and st.session_state["_error"]:
        with st.expander("✗ Error details", expanded=True):
            st.code(st.session_state["_error"], language="python")

with col_right:
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

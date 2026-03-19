"""
Darwinian — 前端界面

启动命令：
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import time
import threading
import queue
from datetime import datetime
from typing import Any

import streamlit as st
from dotenv import load_dotenv

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
/* 全局字体和背景 */
.main { background: #0e1117; }

/* 顶栏标题 */
.app-header {
    display: flex; align-items: center; gap: 12px;
    padding: 16px 0 8px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 24px;
}
.app-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
.app-header .subtitle { color: #8b949e; font-size: 0.9rem; margin-top: 2px; }

/* Agent 流程卡片 */
.agent-card {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    border: 1px solid #21262d;
    background: #161b22;
    transition: border-color 0.2s;
}
.agent-card.running  { border-color: #388bfd; background: #0d1117; }
.agent-card.done     { border-color: #3fb950; }
.agent-card.error    { border-color: #f85149; }
.agent-card.pending  { opacity: 0.45; }

.agent-icon { font-size: 1.4rem; line-height: 1; margin-top: 2px; }
.agent-name { font-weight: 600; font-size: 0.95rem; color: #e6edf3; }
.agent-desc { font-size: 0.82rem; color: #8b949e; margin-top: 2px; }
.agent-output {
    font-size: 0.8rem; color: #c9d1d9;
    background: #0d1117; border-radius: 4px;
    padding: 6px 10px; margin-top: 8px;
    border-left: 3px solid #388bfd;
    white-space: pre-wrap; word-break: break-all;
}

/* 发表指标矩阵 */
.publish-matrix {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 12px; margin-top: 8px;
}
.matrix-item {
    padding: 14px 16px; border-radius: 8px;
    border: 1px solid #21262d; text-align: center;
}
.matrix-item.green  { border-color: #3fb950; background: #0a1f10; }
.matrix-item.red    { border-color: #f85149; background: #1f0a0a; }
.matrix-item.grey   { border-color: #30363d; background: #161b22; }
.matrix-label { font-size: 0.82rem; color: #8b949e; margin-bottom: 4px; }
.matrix-value { font-size: 1.5rem; }

/* failed_ledger 条目 */
.ledger-item {
    padding: 8px 12px; border-radius: 6px;
    border-left: 3px solid #f85149;
    background: #161b22; margin-bottom: 6px;
    font-size: 0.82rem; color: #c9d1d9;
}
.ledger-type { font-weight: 600; color: #f85149; margin-right: 6px; }

/* 指标对比表 */
.metrics-table {
    width: 100%; border-collapse: collapse; font-size: 0.85rem;
}
.metrics-table th {
    text-align: left; padding: 8px 12px;
    border-bottom: 1px solid #21262d; color: #8b949e;
}
.metrics-table td {
    padding: 8px 12px; border-bottom: 1px solid #161b22; color: #e6edf3;
}
.metrics-table .better { color: #3fb950; font-weight: 600; }
.metrics-table .worse  { color: #f85149; }

/* 循环进度徽章 */
.loop-badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 12px; font-size: 0.78rem; font-weight: 600;
    background: #1f2937; color: #60a5fa; border: 1px solid #374151;
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
        <span style="font-size:2.2rem">🧬</span>
        <div>
            <h1>Darwinian</h1>
            <div class="subtitle">状态驱动型多智能体自动化科研系统 · Powered by LangGraph</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_agent_card(agent_id: str, icon: str, name: str, desc: str):
    status = st.session_state.agent_status.get(agent_id, "pending")
    output = st.session_state.agent_output.get(agent_id, "")

    status_icon = {"pending": "⬜", "running": "⏳", "done": "✅", "error": "❌"}.get(status, "⬜")
    output_html = f'<div class="agent-output">{output}</div>' if output else ""

    st.markdown(f"""
    <div class="agent-card {status}">
        <div class="agent-icon">{icon}</div>
        <div style="flex:1">
            <div style="display:flex;align-items:center;gap:8px">
                <span class="agent-name">{name}</span>
                <span>{status_icon}</span>
            </div>
            <div class="agent-desc">{desc}</div>
            {output_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_publish_matrix():
    m = st.session_state.publish_matrix
    items = [
        ("新颖性",   m["novelty"],   "💎"),
        ("基准提升", m["baseline"],  "📈"),
        ("鲁棒性",   m["robustness"],"🛡️"),
        ("可解释性", m["explain"],   "📖"),
    ]
    html = '<div class="publish-matrix">'
    for label, passed, icon in items:
        css = "green" if passed else "grey"
        val = "✅" if passed else "⬜"
        html += f'<div class="matrix-item {css}"><div class="matrix-label">{label}</div><div class="matrix-value">{icon} {val}</div></div>'
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
    if not logs:
        st.caption("等待运行...")
        return
    level_color = {"info": "#8b949e", "ok": "#3fb950", "warn": "#d29922", "error": "#f85149"}
    lines = []
    for ts, level, msg in logs[-60:]:   # 最多显示最近 60 条
        color = level_color.get(level, "#8b949e")
        lines.append(
            f'<span style="color:#555e6b">{ts}</span> '
            f'<span style="color:{color}">{msg}</span>'
        )
    html = (
        '<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
        'padding:12px 14px;height:220px;overflow-y:auto;font-family:monospace;'
        'font-size:0.78rem;line-height:1.7">'
        + "<br>".join(lines)
        + "</div>"
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
# 运行逻辑（在后台线程中执行图）
# ──────────────────────────────────────────────
def _run_graph(research_direction: str, dataset_schema: dict, api_key: str,
               model: str, provider: str, max_loops: int, q: queue.Queue):
    """在独立线程中运行 LangGraph，通过 queue 向主线程传递状态更新。"""
    try:
        if provider == "Anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model=model, max_tokens=8192)

        elif provider == "OpenAI":
            os.environ["OPENAI_API_KEY"] = api_key
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model, max_tokens=8192)

        elif provider == "MiniMax":
            # MiniMax 提供 OpenAI 兼容接口
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url="https://api.minimax.chat/v1",
                max_tokens=8192,
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
    """从节点更新中提取简短摘要写入 agent_output。"""
    if not isinstance(update, dict):
        return
    if agent_id == "bottleneck_miner" and "current_hypothesis" in update:
        h = update["current_hypothesis"]
        if hasattr(h, "core_problem"):
            st.session_state.agent_output[agent_id] = f"核心矛盾：{h.core_problem[:120]}"
    elif agent_id == "hypothesis_generator" and "current_hypothesis" in update:
        h = update["current_hypothesis"]
        if hasattr(h, "abstraction_tree") and h.abstraction_tree:
            names = ", ".join(b.name for b in h.abstraction_tree[:3])
            st.session_state.agent_output[agent_id] = f"生成方案：{names}"
    elif agent_id == "theoretical_critic" and "critic_verdict" in update:
        v = update["critic_verdict"]
        verdict_str = v.value if hasattr(v, "value") else str(v)
        fb = update.get("critic_feedback", "")[:100]
        st.session_state.agent_output[agent_id] = f"裁决：{verdict_str}　{fb}"
    elif agent_id == "diagnostician" and "experiment_result" in update:
        er = update["experiment_result"]
        if hasattr(er, "diagnosis"):
            st.session_state.agent_output[agent_id] = er.diagnosis[:120]
    elif agent_id == "poison_generator" and "poison_test_result" in update:
        ptr = update["poison_test_result"]
        if hasattr(ptr, "perturbation_strategy"):
            st.session_state.agent_output[agent_id] = f"策略：{ptr.perturbation_strategy}"
    elif agent_id == "publish_evaluator" and "final_verdict" in update:
        fv = update["final_verdict"]
        verdict_str = fv.value if hasattr(fv, "value") else str(fv)
        st.session_state.agent_output[agent_id] = f"最终裁决：{verdict_str}"


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

with col_left:
    st.markdown("### 🔄 研究流程")
    for agent_id, icon, name, desc in AGENTS:
        render_agent_card(agent_id, icon, name, desc)

    # 实时日志
    st.markdown("### 🖥️ 运行日志")
    render_log_console()

    # 错误显示
    if "_error" in st.session_state and st.session_state["_error"]:
        with st.expander("❌ 运行错误详情", expanded=True):
            st.code(st.session_state["_error"], language="python")

with col_right:
    # 发表指标矩阵
    st.markdown("### 📊 发表指标矩阵")
    render_publish_matrix()

    # 最终裁决
    if st.session_state.final_verdict:
        verdict = st.session_state.final_verdict
        if verdict == "publish_ready":
            st.success("🎉 研究达到发表标准！报告已生成。", icon="✅")
        else:
            st.warning("⚠️ 鲁棒性测试未通过，已记入认知账本，准备下一轮。", icon="🔄")

    st.divider()

    # 实验指标对比
    st.markdown("### 📈 实验指标对比")
    render_metrics_table()

    st.divider()

    # 认知账本
    ledger_count = len(st.session_state.failed_ledger)
    st.markdown(f"### 📋 认知账本 ({ledger_count} 条记录)")
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

# ── 自动刷新（运行中每 2 秒刷新一次）──
if st.session_state.running:
    time.sleep(2)
    st.rerun()

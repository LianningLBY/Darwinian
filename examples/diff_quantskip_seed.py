"""
QuantSkip seed diff 实验

两种 mode：

(1) IDEAL mode（默认，无需 API key）
    手工构造"理想 ResearchProposal"（模拟 elaborator 完美输出）→ seed_renderer
    用于验证 schema + renderer 覆盖度。

    PYTHONPATH=src python examples/diff_quantskip_seed.py > /tmp/our_seed.md

(2) LIVE mode（需 .env 中 MINIMAX_API_KEY）
    调 elaborate_proposal_from_pack(skeleton, material_pack, llm) → seed_renderer
    用于看真实 LLM 给的 proposal 离 QuantSkip 多远。

    LIVE_LLM=1 PYTHONPATH=src python examples/diff_quantskip_seed.py > /tmp/llm_seed.md

    diff /tmp/our_seed.md /tmp/llm_seed.md   # IDEAL vs LLM
    diff /tmp/llm_seed.md path/to/quantskip_original.md   # LLM vs ground truth
"""

from __future__ import annotations

import os
import sys

# 让 examples 脚本能直接 import darwinian 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from darwinian.state import (
    AbstractionBranch,
    ExpectedOutcomes,
    MethodologyPhase,
    PaperEvidence,
    QuantitativeClaim,
    ResearchConstraints,
    ResearchMaterialPack,
    ResearchProposal,
    ResourceEstimate,
    StructuralHoleHook,
)
from darwinian.tools.seed_renderer import render_proposal


# ---------------------------------------------------------------------------
# 1. 手工构造 QuantSkip 方向的 ResearchMaterialPack（模拟 Phase A 产出）
# ---------------------------------------------------------------------------

def build_material_pack() -> ResearchMaterialPack:
    """
    模拟 Phase A 调研 Agent 的产出：4 篇核心论文 + 1 个结构洞 + 完整约束。
    全部数据来源于 QuantSkip 原文，可追溯。
    """
    constraints = ResearchConstraints(
        gpu_count=4,
        gpu_model="RTX PRO 6000 96GB",
        gpu_hours_budget=4 * 24 * 7.0,    # 4 卡 × 24h × 7 天
        wall_clock_days=7,
        max_model_params_b=14.0,
        use_existing_benchmarks_only=True,
        require_human_annotation=False,
        forbidden_techniques=["GRPO", "PPO", "DPO", "RLHF", "RLVR", "RLMT"],
        require_no_api_for_main=True,
        target_venues=["NeurIPS 2026", "EMNLP 2026", "AAAI 2027"],
        extra_notes="原始约束：seed 表头里的研究方向描述（推理效率与模型架构改进）",
    )

    paper_evidence = [
        PaperEvidence(
            paper_id="arxiv:2404.16710",
            title="LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding",
            short_name="LayerSkip",
            venue="ACL 2024", year=2024,
            category="Layer-skipping self-speculative methods",
            method_names=["layer dropout", "early exit loss", "self-speculative decoding"],
            datasets=["CNN/DailyMail", "TOPv2", "MMLU"],
            metrics=["speedup", "perplexity"],
            quantitative_claims=[
                QuantitativeClaim(metric_name="speedup", metric_value="2.16x",
                                  setting="Llama-3.1-8B on CNN/DailyMail summarization"),
                QuantitativeClaim(metric_name="speedup", metric_value="1.82x",
                                  setting="Llama-3.1-8B on coding tasks"),
            ],
            headline_result="1.3-2.4x speedup",
            limitations=["Requires finetuning with LayerSkip recipe",
                         "Hyperparameters require tuning"],
            relation_to_direction="extends",
            full_text_used=True,
        ),
        PaperEvidence(
            paper_id="arxiv:2510.del-placeholder",
            title="DEL: Context-Aware Dynamic Exit Layer for Efficient Self-Speculative Decoding",
            short_name="DEL",
            venue="COLM 2025", year=2025,
            category="Layer-skipping self-speculative methods",
            method_names=["dynamic exit layer", "context-aware skip"],
            datasets=["MATH", "GSM8K"],
            metrics=["speedup"],
            quantitative_claims=[
                QuantitativeClaim(metric_name="speedup", metric_value="2.16-2.62x",
                                  setting="Llama-3.1-8B on MATH"),
            ],
            headline_result="2.16-2.62x speedup",
            limitations=["Layer-skip search at full precision only"],
            relation_to_direction="baseline",
            full_text_used=False,
        ),
        PaperEvidence(
            paper_id="arxiv:2603.17891",
            title="RAMP: Reinforcement Adaptive Mixed Precision Quantization for Efficient On-Device LLM Inference",
            short_name="RAMP",
            venue="arxiv preprint 2026", year=2026,
            category="Per-layer mixed-precision quantization (general inference)",
            method_names=["RL-based per-layer bit-width", "zero-shot transfer"],
            datasets=["WikiText-2", "Llama-2-7B"],
            metrics=["perplexity"],
            quantitative_claims=[
                QuantitativeClaim(metric_name="perplexity", metric_value="5.54 vs 5.60",
                                  setting="Llama-2-7B mixed-precision vs uniform"),
            ],
            headline_result="reduces PPL from 5.60 to 5.54 on Llama-2-7B",
            limitations=["Sensitivity signal is perplexity, not draft acceptance rate",
                         "Does not consider layer skipping"],
            relation_to_direction="baseline",
            full_text_used=False,
        ),
        PaperEvidence(
            paper_id="arxiv:qspec-emnlp2025",
            title="QSpec: Speculative Decoding with Complementary Quantization Schemes",
            short_name="QSpec",
            venue="EMNLP 2025", year=2025,
            category="Quantization-based speculative decoding",
            method_names=["W4A4 draft", "W4A16 verify"],
            datasets=["MMLU", "GSM8K"],
            metrics=["speedup"],
            quantitative_claims=[
                QuantitativeClaim(metric_name="speedup", metric_value="1.64-1.80x",
                                  setting="Llama-3-8B"),
            ],
            headline_result="1.64-1.80x speedup",
            limitations=["Uniform quantization across layers",
                         "No per-layer sensitivity analysis"],
            relation_to_direction="baseline",
            full_text_used=False,
        ),
    ]

    structural_hole = StructuralHoleHook(
        entity_a="per-layer quantization sensitivity (accuracy-guided)",
        entity_b="draft token acceptance rate",
        score=12,
        hook_text=(
            "Mixed-precision methods (RAMP, LLM-MQ) measure per-layer sensitivity via "
            "perplexity/accuracy; self-speculative methods (DEL, KnapSpec) optimize layer "
            "skipping using draft acceptance rate at full precision. No work has measured "
            "quantization sensitivity through the lens of draft quality — a local distributional "
            "property fundamentally different from the global functional property of accuracy."
        ),
        relation_type="divergence",
        supporting_paper_ids_a=["arxiv:2603.17891"],
        supporting_paper_ids_b=["arxiv:2404.16710", "arxiv:2510.del-placeholder",
                                "arxiv:qspec-emnlp2025"],
    )

    return ResearchMaterialPack(
        direction=(
            "帮我生成和LLM相关的、2025-2026年火的、有low hanging fruits的idea，"
            "要求能中NeurIPS 2026，用四卡RTX PRO 6000（单卡96GB显存）及以下的资源，"
            "auto research模式下七天内可以完成，尽量不用API key（不用来构建benchmark，"
            "但可以用来作为baseline进行比较），不自己构建benchmark，不需要人工标注，"
            "不涉及任何RL训练（不要GRPO/PPO/DPO/RLHF/RLVR/RLMT）。"
        ),
        constraints=constraints,
        paper_evidence=paper_evidence,
        concept_graph=None,    # 本 demo 不构造完整 ConceptGraph
        structural_hole_hooks=[structural_hole],
        timeline_signals={
            "foundational_pre_2024": ["arxiv:2404.16710"],
            "hot_2025_2026": ["arxiv:2510.del-placeholder", "arxiv:2603.17891",
                              "arxiv:qspec-emnlp2025"],
        },
        prior_failures=[],
    )


# ---------------------------------------------------------------------------
# 2. 手工构造 QuantSkip 风格的 ResearchProposal（模拟 elaborator 完美输出）
# ---------------------------------------------------------------------------

def build_ideal_proposal(pack: ResearchMaterialPack) -> ResearchProposal:
    """
    照着 QuantSkip 原文构造，用于评估 schema + renderer 的覆盖度。
    """
    skeleton = AbstractionBranch(
        name="QuantSkip",
        description="Per-layer quantization sensitivity for draft acceptance rate",
        algorithm_logic=(
            "1) Profile per-layer dual sensitivity (accuracy + draft acceptance) "
            "2) Joint knapsack DP "
            "3) Cross-model transfer validation"
        ),
        math_formulation=r"$\rho_{\text{Spearman}}(s_{\text{acc}}, s_{\text{draft}})$",
        cited_entity_names=["layer dropout", "self-speculative decoding",
                            "RL-based per-layer bit-width"],
        solved_limitation_id="rampacc",
        existing_combination=False,
    )

    phases = [
        MethodologyPhase(
            phase_number=1,
            name="Dual-metric sensitivity profiling",
            description=(
                "For each of 32 transformer layers in Llama-3.1-8B, construct per-layer "
                "quantized variants at INT4 and INT8 using llm-compressor. Measure two "
                "sensitivity signals: (a) draft token acceptance rate degradation in a "
                "self-speculative pipeline, (b) task accuracy degradation (perplexity on "
                "WikiText-2, accuracy on MMLU/GSM8K). Compute Spearman's rho between the "
                "two 32-d sensitivity vectors."
            ),
            inputs=["Llama-3.1-8B FP16", "C4/RedPajama 5K-token calibration",
                    "WikiText-2", "MMLU", "GSM8K"],
            outputs=["32-d draft sensitivity vector", "32-d accuracy sensitivity vector",
                    "Spearman rho", "divergence pattern characterization"],
            expected_compute_hours=48.0,
        ),
        MethodologyPhase(
            phase_number=2,
            name="Draft-aware joint optimization",
            description=(
                "Formulate per-layer precision + skip assignment as generalized knapsack. "
                "Each layer items: {skip, INT4, INT8, FP16}. Solve via DP. Compare 4 "
                "allocation strategies: draft-sensitivity-guided (ours) vs accuracy-"
                "sensitivity-guided (RAMP-style) vs uniform quantization (QSpec-style) "
                "vs skip-only (KnapSpec-style)."
            ),
            inputs=["sensitivity vectors from Phase 1", "per-layer latency profile"],
            outputs=["Pareto-optimal config set", "speedup vs accuracy trade-off curves"],
            expected_compute_hours=24.0,
        ),
        MethodologyPhase(
            phase_number=3,
            name="Cross-model transfer validation",
            description=(
                "Replicate dual-metric profiling on Qwen3-8B. Test (a) does the divergence "
                "pattern transfer? (b) can Llama-3.1-8B's optimal configs transfer zero-shot?"
            ),
            inputs=["Qwen3-8B FP16", "Phase 1 calibration corpus"],
            outputs=["Qwen3-8B sensitivity vectors", "transfer success rate"],
            expected_compute_hours=36.0,
        ),
        MethodologyPhase(
            phase_number=4,
            name="End-to-end evaluation",
            description=(
                "Implement top configs from draft-aware Pareto frontier. Measure tokens/sec, "
                "acceptance rate, task accuracy on MATH/GSM8K/HumanEval. Compare against "
                "LayerSkip, DEL, QSpec, QuantSpec, SpecAttn baselines."
            ),
            inputs=["top-K configs from Phase 2", "MATH", "GSM8K", "HumanEval"],
            outputs=["tokens/sec", "acceptance rate", "task accuracy", "structural analysis"],
            expected_compute_hours=60.0,
        ),
    ]

    return ResearchProposal(
        skeleton=skeleton,
        status="draft", level="top-tier",
        seed=pack.direction,
        created_at="2026-04-19T03:02:07.693636+00:00",
        title="QuantSkip: Do LLM Layers That Tolerate Quantization for Accuracy Also Tolerate It for Draft Quality?",
        elevator_pitch=(
            "We present QuantSkip, an empirical study and optimization framework that asks "
            "a fundamental question: does per-layer quantization sensitivity for draft token "
            "acceptance rate in self-speculative decoding follow the same pattern as per-layer "
            "sensitivity for task accuracy? Existing mixed-precision methods (RAMP, LLM-MQ) "
            "assign per-layer bit-widths by measuring accuracy sensitivity, while self-speculative "
            "methods (DEL, KnapSpec) optimize layer skipping at full precision. We profile each "
            "layer of Llama-3.1-8B at INT4 and INT8, measuring both draft acceptance rate "
            "degradation and accuracy degradation. Whether divergence exists or not, both "
            "outcomes are scientifically publishable."
        ),
        challenges=(
            "Mixed-precision quantization assigns per-layer bit-widths by measuring each "
            "layer's sensitivity to quantization — typically via its impact on perplexity or "
            "task accuracy. Self-speculative decoding methods skip or quantize layers to create "
            "fast draft models, and their key performance metric is draft token acceptance rate. "
            "**The unstudied gap**: does per-layer quantization sensitivity for draft acceptance "
            "rate follow the same pattern as per-layer sensitivity for task accuracy? If these "
            "sensitivity profiles diverge, then existing mixed-precision strategies optimized "
            "for accuracy will produce suboptimal draft models."
        ),
        existing_methods=(
            "**Layer-skipping self-speculative methods**: LayerSkip (Meta, ACL 2024, 1.3-2.4x), "
            "DEL (COLM 2025, 2.16-2.62x), KnapSpec (Feb 2026, knapsack-based).\n\n"
            "**Quantization-based speculative**: QSpec (EMNLP 2025, 1.64-1.80x, W4A4/W4A16), "
            "QuantSpec (ICML 2025, ~2.5x, hierarchical INT4 KV).\n\n"
            "**Per-layer mixed-precision (general inference)**: RAMP (Mar 2026, RL-based per-layer "
            "bit-width with zero-shot transfer, optimizes perplexity), LLM-MQ (gradient-based "
            "sensitivity), OWQ (outlier-aware).\n\n"
            "**Joint quantization + layer execution**: QTALE (Feb 2026, fine-tuning-based, "
            "accuracy preservation), MoBiQuant (Feb 2026, learned routers).\n\n"
            "**Interaction studies**: BitSkip (Oct 2025, sub-additive at 1.58-bit/85M with QAT "
            "— different regime), SpecMQuant (May 2025, finds uniform 4-bit diminishes spec "
            "gains, no per-layer analysis).\n\n"
            "**The gap**: No work measures per-layer quantization sensitivity specifically for "
            "draft token acceptance rate, nor compares this profile against the well-studied "
            "accuracy-sensitivity profile."
        ),
        motivation=(
            "Self-speculative layer-skipping methods (DEL 2.16-2.62x, LayerSkip 1.3-2.4x) and "
            "quantization methods (QSpec 1.64-1.80x, QuantSpec ~2.5x) both accelerate inference "
            "but operate in one dimension. The joint space is exponentially larger but unexplored "
            "for a fundamental reason: nobody knows whether the sensitivity landscape governing "
            "this space differs from the well-studied accuracy-sensitivity landscape. RAMP "
            "demonstrates RL-based per-layer precision can outperform uniform quantization for "
            "perplexity (5.54 vs 5.60 on Llama-2-7B) and transfers zero-shot — but its sensitivity "
            "signal is perplexity, not draft acceptance rate. If draft-quality sensitivity "
            "diverges from accuracy sensitivity, then RAMP's (and all accuracy-guided) precision "
            "assignments will be systematically suboptimal for speculative decoding. "
            "**This divergence is plausible**: draft acceptance rate depends on the output "
            "distribution matching at every token (a local, distributional property), while "
            "task accuracy depends on the final answer being correct (a global, functional "
            "property)."
        ),
        proposed_method=(
            "Three-phase methodology centered on a scientific question: does per-layer "
            "quantization sensitivity for draft quality diverge from sensitivity for task "
            "accuracy?"
        ),
        technical_details=(
            r"Sensitivity vector $s_m \in \mathbb{R}^{32}$ for metric $m \in \{\text{accuracy}, \text{draft}\}$. "
            r"Divergence statistic: Spearman's $\rho(s_{\text{acc}}, s_{\text{draft}})$. "
            r"Joint objective: $\max_{c} Q_{\text{draft}}(c)$ s.t. $\sum_l \text{lat}(c_l) \leq B$, "
            r"where $c \in \{\text{skip}, \text{INT4}, \text{INT8}, \text{FP16}\}^{32}$. Solve via DP."
        ),
        expected_outcomes_structured=ExpectedOutcomes(
            positive_finding=(
                "If we find draft-vs-accuracy sensitivity divergence with low Spearman rho "
                "(< 0.5) and >5% gap between draft-guided and accuracy-guided allocations, "
                "this proves speculative decoding requires its own sensitivity metric, "
                "invalidating direct reuse of RAMP/LLM-MQ for draft optimization."
            ),
            negative_finding=(
                "If sensitivity profiles converge (rho > 0.9) and allocations achieve "
                "comparable speedup, this validates that accuracy-guided mixed-precision "
                "methods can be directly reused for speculative decoding without modification — "
                "a useful and non-obvious result for the community."
            ),
            why_both_publishable=(
                "Either outcome provides actionable guidance: divergence motivates "
                "draft-specific sensitivity research; convergence simplifies the design "
                "space and validates accuracy-as-proxy. The question itself is fundamental "
                "and unaddressed by prior work."
            ),
        ),
        methodology_phases=phases,
        total_estimated_hours=sum(p.expected_compute_hours for p in phases),
        fits_resource_budget=True,
        target_venue="EMNLP 2026",
        target_deadline="2026-05-25",
        fallback_venue="AAAI 2027 (deadline August 1, 2026)",
        key_references=[ev.paper_id for ev in pack.paper_evidence],
        # 注：title 通常已以 short_name 开头（如 "LayerSkip: Enabling..."），
        # 直接用 "title (venue)" 即可，不再前缀 short_name 防重复
        key_references_formatted=[
            f"{ev.title} ({ev.venue})"
            for ev in pack.paper_evidence
        ],
        resource_estimate=ResourceEstimate(
            auto_research={"gpu_hours": 168, "usd_cost": 0, "wall_clock_days": 7},
            human_in_loop={"gpu_hours": 168, "human_hours": 14, "wall_clock_days": 7},
            manual={"gpu_hours": 168, "human_hours": 280, "wall_clock_days": 35},
        ),
    )


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main() -> None:
    pack = build_material_pack()

    if os.environ.get("LIVE_LLM") == "1":
        proposal = _live_elaborate(pack)
    else:
        proposal = build_ideal_proposal(pack)

    if proposal is None:
        print("# (proposal 生成失败)")
        return
    md = render_proposal(proposal, material_pack=pack)
    print(md)


def _live_elaborate(pack: ResearchMaterialPack) -> ResearchProposal | None:
    """LIVE mode：调 MiniMax LLM 跑 elaborate_proposal_from_pack"""
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("# 错误：LIVE_LLM=1 但环境无 MINIMAX_API_KEY", file=sys.stderr)
        return None
    from darwinian.agents.proposal_elaborator import elaborate_proposal_from_pack
    from darwinian.llms import ChatMiniMax
    llm = ChatMiniMax(model="MiniMax-M2.7", api_key=api_key, max_tokens=8192)

    # 用 ideal proposal 的 skeleton 作为输入骨架
    ideal = build_ideal_proposal(pack)
    return elaborate_proposal_from_pack(
        skeleton=ideal.skeleton,
        material_pack=pack,
        llm=llm,
    )


if __name__ == "__main__":
    main()

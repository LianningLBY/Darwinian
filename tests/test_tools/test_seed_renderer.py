"""
seed_renderer 单元测试：纯函数渲染，不依赖 LLM。

校验目标：
  1. 必填字段有时输出含对应内容
  2. 缺字段时输出含占位符 '(待补)'，不崩溃
  3. expected_outcomes_structured 优先于 expected_outcomes str
  4. methodology_phases 非空时按 phase 渲染
  5. metadata 块含状态/级别/种子/创建时间
  6. ResearchMaterialPack.direction 在 proposal.seed 为空时兜底
"""

import pytest

from darwinian.state import (
    AbstractionBranch,
    ExpectedOutcomes,
    MethodologyPhase,
    ResearchConstraints,
    ResearchMaterialPack,
    ResearchProposal,
    ResourceEstimate,
)
from darwinian.tools.seed_renderer import render_proposal


def _minimal_skeleton() -> AbstractionBranch:
    return AbstractionBranch(
        name="x",
        description="x",
        algorithm_logic="x",
        math_formulation="x",
    )


def _minimal_proposal(**overrides) -> ResearchProposal:
    base = dict(
        skeleton=_minimal_skeleton(),
        title="QuantSkip: Do A Also Imply B?",
        elevator_pitch="200-word pitch...",
    )
    base.update(overrides)
    return ResearchProposal(**base)


class TestMetadataBlock:
    def test_default_status_translated(self):
        p = _minimal_proposal()
        md = render_proposal(p)
        assert "**状态**: 待审阅" in md  # 默认 draft → 待审阅

    def test_level_and_created_at(self):
        p = _minimal_proposal(level="top-tier", created_at="2026-04-19T03:02:07")
        md = render_proposal(p)
        assert "**级别**: top-tier" in md
        assert "**创建时间**: 2026-04-19T03:02:07" in md

    def test_seed_from_proposal(self):
        p = _minimal_proposal(seed="生成 NeurIPS 2026 idea")
        md = render_proposal(p)
        assert "生成 NeurIPS 2026 idea" in md

    def test_seed_falls_back_to_material_pack_direction(self):
        p = _minimal_proposal()  # seed 空
        pack = ResearchMaterialPack(direction="LLM inference acceleration")
        md = render_proposal(p, material_pack=pack)
        assert "LLM inference acceleration" in md

    def test_missing_status_uses_placeholder(self):
        p = _minimal_proposal(status="unknown_status")
        md = render_proposal(p)
        # 未知状态 fallback 显示原文（不崩）
        assert "**状态**: unknown_status" in md


class TestSectionPresence:
    def test_all_sections_present_with_placeholders(self):
        p = _minimal_proposal()  # 大量字段为空
        md = render_proposal(p)
        for section in ["## 描述", "## 核心问题", "## 机遇 / Why Now",
                        "## 方法思路", "## 研究现状", "## 目标 Venue",
                        "## 关键参考", "## 资源预估"]:
            assert section in md, f"缺 section: {section}"

    def test_filled_content_appears(self):
        p = _minimal_proposal(
            elevator_pitch="QuantSkip studies whether...",
            challenges="The core challenge is layer sensitivity divergence",
            motivation="DEL achieves 2.16-2.62x; RAMP reduces PPL 5.60→5.54",
            existing_methods="**Layer-skipping**: LayerSkip, DEL, KnapSpec",
        )
        md = render_proposal(p)
        assert "QuantSkip studies whether" in md
        assert "layer sensitivity divergence" in md
        assert "2.16-2.62x" in md
        assert "**Layer-skipping**: LayerSkip" in md


class TestMethodology:
    def test_phases_rendered_with_compute_hours(self):
        phases = [
            MethodologyPhase(
                phase_number=1, name="Profiling",
                description="Profile per-layer sensitivity",
                inputs=["Llama-3.1-8B"], outputs=["sensitivity vector"],
                expected_compute_hours=24.0,
            ),
            MethodologyPhase(
                phase_number=2, name="Knapsack DP",
                description="Joint optimization",
                inputs=["sensitivity vectors"], outputs=["Pareto frontier"],
                expected_compute_hours=8.0,
            ),
        ]
        p = _minimal_proposal(
            proposed_method="High-level intuition: dual-metric profiling",
            methodology_phases=phases,
        )
        md = render_proposal(p)
        assert "(1) **Profiling**" in md
        assert "(2) **Knapsack DP**" in md
        assert "expected_compute_hours: 24.0" in md
        assert "Pareto frontier" in md

    def test_no_phases_falls_back_to_free_text(self):
        p = _minimal_proposal(
            proposed_method="Just a free-text method paragraph",
            technical_details="LaTeX formula here",
        )
        md = render_proposal(p)
        assert "Just a free-text method paragraph" in md
        assert "**Technical details**" in md
        assert "LaTeX formula here" in md


class TestExpectedOutcomes:
    def test_structured_takes_priority(self):
        p = _minimal_proposal(
            expected_outcomes="OLD free text",
            expected_outcomes_structured=ExpectedOutcomes(
                positive_finding="如果发散，证明需 draft-specific metric",
                negative_finding="如果收敛，证明 accuracy-guided 已足够",
                why_both_publishable="两种都给社区 actionable guidance",
            ),
        )
        md = render_proposal(p)
        assert "**正向发现**" in md
        assert "draft-specific metric" in md
        assert "**反向发现**" in md
        assert "**为什么两种结果都可发表**" in md
        # 老 str 不应再被渲染（避免重复）
        assert "OLD free text" not in md

    def test_legacy_str_fallback(self):
        p = _minimal_proposal(expected_outcomes="自由文本结果")
        md = render_proposal(p)
        assert "自由文本结果" in md


class TestVenuesAndReferences:
    def test_target_and_fallback_venue(self):
        p = _minimal_proposal(
            target_venue="EMNLP 2026",
            target_deadline="2026-05-25",
            fallback_venue="AAAI 2027",
        )
        md = render_proposal(p)
        assert "EMNLP 2026" in md
        assert "2026-05-25" in md
        assert "AAAI 2027" in md
        assert "fallback" in md.lower()

    def test_formatted_references_preferred(self):
        p = _minimal_proposal(
            key_references=["arxiv:2404.16710", "arxiv:2603.17891"],
            key_references_formatted=[
                "LayerSkip: Enabling Early Exit Inference (ACL 2024)",
                "RAMP: RL-based Mixed Precision (arXiv 2603.17891)",
            ],
        )
        md = render_proposal(p)
        assert "LayerSkip: Enabling Early Exit Inference" in md
        # 当 formatted 存在时，不应再列 raw paperId
        assert "- arxiv:2404.16710" not in md

    def test_raw_references_fallback(self):
        p = _minimal_proposal(key_references=["arxiv:2404.16710"])
        md = render_proposal(p)
        assert "arxiv:2404.16710" in md


class TestResourceEstimate:
    def test_three_modes_rendered(self):
        p = _minimal_proposal(
            resource_estimate=ResourceEstimate(
                auto_research={"gpu_hours": 96, "usd_cost": 80, "wall_clock_days": 4},
                human_in_loop={"gpu_hours": 96, "human_hours": 7},
                manual={"gpu_hours": 96, "human_hours": 200},
            ),
        )
        md = render_proposal(p)
        assert "auto_research: gpu_hours=96, usd_cost=80, wall_clock_days=4" in md
        assert "human_in_loop:" in md
        assert "manual:" in md

    def test_empty_estimate_placeholder(self):
        p = _minimal_proposal()
        md = render_proposal(p)
        # 资源预估 section 里出现占位符
        idx = md.index("## 资源预估")
        tail = md[idx:]
        assert "(待补)" in tail


# ===========================================================================
# Novelty section (SciMON boost result)
# ===========================================================================

from darwinian.state import NoveltyAssessment


class TestNoveltySection:
    def test_no_assessment_no_section(self):
        p = _minimal_proposal()
        md = render_proposal(p)
        assert "## 新颖性评估" not in md

    def test_assessment_renders(self):
        na = NoveltyAssessment(
            overlap_level="partial",
            closest_work_paper_id="arxiv:2505.22179",
            closest_work_title="SpecMQuant: Speculative Decoding Meets Quantization",
            overlap_summary="both quantize draft model and measure acceptance",
            differentiation_gap="ours measures per-layer cliff, theirs only global",
            novelty_score=0.7,
        )
        p = _minimal_proposal(novelty_assessment=na)
        md = render_proposal(p)
        assert "## 新颖性评估" in md
        assert "`partial`" in md
        assert "novelty_score=0.70" in md
        assert "SpecMQuant" in md
        assert "arxiv:2505.22179" in md
        assert "both quantize draft model" in md
        assert "ours measures per-layer cliff" in md


class TestDebateSection:
    """R19: Debate result 渲染"""

    def _dr(self):
        from darwinian.state import DebateResult, DebateRound
        return DebateResult(
            rounds=[
                DebateRound(
                    round_number=1,
                    advocate_argument="Strong novelty due to recall-vs-forecasting framing",
                    challenger_argument="ACD-WFP 2024 may overlap",
                    judge_assessment="ACD-WFP is partial overlap, not full",
                    estimated_acceptance_rate=0.30,
                    revisions_proposed=["use ISCXVPN2016 instead"],
                ),
                DebateRound(
                    round_number=2,
                    advocate_argument="Refined with ISCXVPN2016",
                    challenger_argument="Still need to address scope",
                    judge_assessment="scope concern partial",
                    estimated_acceptance_rate=0.32,
                    revisions_proposed=["narrow to single primary claim"],
                ),
            ],
            final_acceptance_rate=0.32,
            acceptance_threshold=0.30,
            convergence_delta=0.05,
            converged=True,
        )

    def test_no_debate_no_section(self):
        p = _minimal_proposal()
        md = render_proposal(p)
        assert "Proposal Debate" not in md

    def test_empty_rounds_no_section(self):
        from darwinian.state import DebateResult
        empty = DebateResult(rounds=[])
        p = _minimal_proposal(debate_result=empty)
        md = render_proposal(p)
        assert "Proposal Debate" not in md

    def test_renders_full_debate(self):
        p = _minimal_proposal(debate_result=self._dr())
        md = render_proposal(p)
        assert "## ⚖️ Proposal Debate" in md
        assert "0.32" in md
        assert "converged" in md
        assert "0.30 → 0.32" in md
        assert "Round 1" in md
        assert "Round 2" in md
        assert "🟦 Advocate" in md
        assert "🟥 Challenger" in md
        assert "⚖️ Judge" in md
        assert "ISCXVPN2016" in md
        assert "narrow to single" in md

    def test_not_converged_shows_red(self):
        from darwinian.state import DebateResult, DebateRound
        dr = DebateResult(
            rounds=[
                DebateRound(
                    round_number=1, advocate_argument="weak",
                    challenger_argument="strong", judge_assessment="weak",
                    estimated_acceptance_rate=0.10,
                ),
            ],
            final_acceptance_rate=0.10,
            converged=False,
        )
        p = _minimal_proposal(debate_result=dr)
        md = render_proposal(p)
        assert "🔴" in md
        assert "not converged" in md


class TestMechanismAlignmentSection:
    """R11: Mechanism Alignment Checker 渲染"""

    def _ma_hand_waved(self):
        from darwinian.state import MechanismAlignment, MechanismAlignmentDimension
        return MechanismAlignment(
            is_cross_domain=True,
            source_domain="quantum error correction",
            target_domain="encrypted traffic classification",
            dimensions=[
                MechanismAlignmentDimension(
                    dimension="formal_correspondence", verdict="broken",
                    explanation="qubit Hilbert space != continuous feature space",
                ),
                MechanismAlignmentDimension(
                    dimension="metric_correspondence", verdict="loose",
                    explanation="trace distance vague analog to KL",
                ),
            ],
            overall_verdict="hand_waved",
            recommendation="rewrite motivation without quantum analogy",
        )

    def test_no_alignment_no_section(self):
        p = _minimal_proposal()
        md = render_proposal(p)
        assert "Mechanism Alignment" not in md

    def test_not_applicable_no_section(self):
        from darwinian.state import MechanismAlignment
        ma = MechanismAlignment(
            is_cross_domain=False, overall_verdict="not_applicable",
        )
        p = _minimal_proposal(mechanism_alignment=ma)
        md = render_proposal(p)
        assert "Mechanism Alignment" not in md

    def test_hand_waved_renders(self):
        p = _minimal_proposal(mechanism_alignment=self._ma_hand_waved())
        md = render_proposal(p)
        assert "## ⚠️ Mechanism Alignment" in md
        assert "hand_waved" in md
        assert "quantum error correction" in md
        assert "encrypted traffic classification" in md
        assert "Formal correspondence" in md
        assert "[broken]" in md
        assert "rewrite motivation" in md


class TestRelevanceWarningBanner:
    """R10-Pri-2: Phase A relevance warning 渲染 banner"""

    def test_no_warning_no_banner(self):
        from darwinian.state import ResearchMaterialPack
        p = _minimal_proposal()
        pack = ResearchMaterialPack(direction="d")
        md = render_proposal(p, material_pack=pack)
        assert "Phase A Relevance Warning" not in md

    def test_warning_renders_banner_at_top(self):
        from darwinian.state import ResearchMaterialPack
        p = _minimal_proposal()
        pack = ResearchMaterialPack(
            direction="d",
            relevance_warning="仅找到 3 篇真相关论文，已用 5 篇 orthogonal 兜底",
        )
        md = render_proposal(p, material_pack=pack)
        assert "⚠️ **Phase A Relevance Warning**" in md
        assert "3 篇真相关" in md
        # 必须在 metadata 块之前（即 banner 在 title 后立刻出现）
        banner_pos = md.find("Phase A Relevance Warning")
        meta_pos = md.find("**状态**")
        assert banner_pos < meta_pos


class TestFeasibilityChallengeSection:
    """R9c: Feasibility Challenger 渲染"""

    def _fc(self, **overrides):
        from darwinian.state import FeasibilityChallenge, FeasibilityRisk
        base = dict(
            risks=[
                FeasibilityRisk(
                    category="budget", severity="high",
                    description="phase 2 写 200h 但 budget 只剩 168",
                    mitigation="砍 ablation 数",
                ),
                FeasibilityRisk(
                    category="data", severity="low",
                    description="dataset license unclear",
                ),
            ],
            overall_verdict="rework",
            summary="不修不能跑",
        )
        base.update(overrides)
        return FeasibilityChallenge(**base)

    def test_no_challenge_no_section(self):
        p = _minimal_proposal()
        md = render_proposal(p)
        assert "Feasibility Risks" not in md

    def test_renders_verdict_and_risks(self):
        p = _minimal_proposal(feasibility_challenge=self._fc())
        md = render_proposal(p)
        assert "## ⚠️ Feasibility Risks" in md
        assert "rework" in md
        assert "phase 2" in md
        assert "[HIGH / budget]" in md
        assert "[LOW / data]" in md
        assert "*mitigation*" in md
        assert "砍 ablation" in md

    def test_empty_risks_renders_note(self):
        from darwinian.state import FeasibilityChallenge
        empty_fc = FeasibilityChallenge(risks=[], overall_verdict="go", summary="clean")
        p = _minimal_proposal(feasibility_challenge=empty_fc)
        md = render_proposal(p)
        assert "no risks identified" in md


class TestSpotCheckSection:
    def test_no_unverified_no_section(self):
        p = _minimal_proposal()
        md = render_proposal(p)
        assert "Audit:" not in md

    def test_unverified_renders_section(self):
        p = _minimal_proposal(unverified_numbers=["27%", "42.7%", "9.5x"])
        md = render_proposal(p)
        assert "## ⚠️ Audit:" in md
        assert "`27%`" in md
        assert "`42.7%`" in md
        assert "`9.5x`" in md
        assert "请人工核对" in md

"""
Auto-seed pipeline 端到端 demo —— 真正的"输入方向，全自动出 seed.md"。

Flow:
  Phase A: build_research_material_pack(direction)
    → 用 S2 拉论文 + 一跳引用扩展 + arxiv 全文 + paper_evidence_extractor
    → ResearchMaterialPack
  Phase B: elaborate_proposal_from_pack(skeleton, pack, llm)
    → ResearchProposal
  Render: render_proposal(proposal, material_pack=pack) → seed.md

需要环境变量：
  MINIMAX_API_KEY            必填
  SEMANTIC_SCHOLAR_API_KEY   强烈建议（无 key 走匿名 tier 会 429）

用法：
  PYTHONPATH=src python examples/auto_seed_pipeline.py > /tmp/auto_seed.md

  或自定义方向：
  PYTHONPATH=src python examples/auto_seed_pipeline.py "graph neural network drug discovery" > /tmp/seed.md
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from darwinian.state import (
    AbstractionBranch,
    ResearchConstraints,
)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("# 错误：缺 MINIMAX_API_KEY", file=sys.stderr)
        sys.exit(1)

    direction = sys.argv[1] if len(sys.argv) > 1 else (
        "LLM inference acceleration via speculative decoding and quantization, "
        "must run on 4x RTX PRO 6000 in 7 days, no RL training"
    )

    print(f"[auto_seed] direction: {direction}", file=sys.stderr)

    from darwinian.llms import ChatMiniMax
    extractor_llm = ChatMiniMax(model="MiniMax-M2.7", api_key=api_key, max_tokens=4096)
    evidence_llm  = ChatMiniMax(model="MiniMax-M2.7", api_key=api_key, max_tokens=4096)
    elaborator_llm = ChatMiniMax(model="MiniMax-M2.7", api_key=api_key, max_tokens=8192)

    constraints = ResearchConstraints(
        gpu_count=4,
        gpu_model="RTX PRO 6000 96GB",
        gpu_hours_budget=4 * 24 * 7.0,
        wall_clock_days=7,
        max_model_params_b=14.0,
        use_existing_benchmarks_only=True,
        require_human_annotation=False,
        forbidden_techniques=["GRPO", "PPO", "DPO", "RLHF", "RLVR", "RLMT"],
        require_no_api_for_main=True,
        target_venues=["NeurIPS 2026", "EMNLP 2026", "AAAI 2027"],
    )

    # ---- Phase A ----
    from darwinian.agents.phase_a_orchestrator import build_research_material_pack
    print("[auto_seed] Phase A 启动 —— 这步可能要 5-15 分钟（看论文数量）", file=sys.stderr)
    pack = build_research_material_pack(
        direction=direction,
        constraints=constraints,
        extractor_llm=extractor_llm,
        evidence_llm=evidence_llm,
        top_k_evidence=8,    # 实测先取 8 篇，不至于太久
    )
    print(f"[auto_seed] Phase A 完成: {len(pack.paper_evidence)} 篇 PaperEvidence", file=sys.stderr)

    if not pack.paper_evidence:
        print("# 错误：Phase A 未抽到任何 PaperEvidence", file=sys.stderr)
        sys.exit(2)

    # ---- 构造 skeleton（暂用最简版，待 Agent 2 hypothesis_generator 替代）----
    skeleton = AbstractionBranch(
        name="auto-skeleton",
        description=f"Auto-generated skeleton for direction: {direction[:80]}",
        algorithm_logic="(待 hypothesis_generator 接入后自动产出)",
        math_formulation=r"$f(\cdot)$",
    )

    # ---- Phase B: elaborator ----
    from darwinian.agents.proposal_elaborator import elaborate_proposal_from_pack
    print("[auto_seed] Phase B 启动 —— elaborator", file=sys.stderr)
    proposal = elaborate_proposal_from_pack(
        skeleton=skeleton, material_pack=pack, llm=elaborator_llm,
    )
    if proposal is None:
        print("# 错误：elaborator 3 次重试全失败", file=sys.stderr)
        sys.exit(3)
    print("[auto_seed] Phase B 完成", file=sys.stderr)

    # ---- Phase B.5: SciMON novelty boost ----
    from darwinian.agents.novelty_booster import boost_novelty
    print("[auto_seed] Phase B.5 启动 —— SciMON novelty boost (max 3 rounds)",
          file=sys.stderr)
    proposal, boost_result = boost_novelty(
        proposal=proposal, direction=direction, llm=extractor_llm, max_rounds=3,
    )
    print(f"[auto_seed] Phase B.5 完成: {boost_result.rounds_taken} 轮, "
          f"converged={boost_result.converged}, "
          f"final overlap_level={(boost_result.final_assessment.overlap_level if boost_result.final_assessment else 'n/a')}",
          file=sys.stderr)

    # ---- 渲染 ----
    from darwinian.tools.seed_renderer import render_proposal
    md = render_proposal(proposal, material_pack=pack)
    print(md)


if __name__ == "__main__":
    main()

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
    from darwinian.agents.phase_a_orchestrator import (
        PhaseAAbortError,
        build_research_material_pack,
    )
    print("[auto_seed] Phase A 启动 —— 这步可能要 5-15 分钟（看论文数量）", file=sys.stderr)
    try:
        pack = build_research_material_pack(
            direction=direction,
            constraints=constraints,
            extractor_llm=extractor_llm,
            evidence_llm=evidence_llm,
            top_k_evidence=8,    # 实测先取 8 篇，不至于太久
        )
    except PhaseAAbortError as e:
        # R12: 真相关论文不足，hard abort，省下游 ~80 LLM call 的钱
        print(f"# 错误：Phase A 主动终止\n# {e}", file=sys.stderr)
        sys.exit(4)
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

    # ---- Phase B: multi_elaborate (Tournament 路径，N=5 候选) ----
    from darwinian.agents.proposal_tournament import multi_elaborate, run_tournament
    n_candidates = int(os.environ.get("DARWINIAN_N_CANDIDATES", "5"))
    print(f"[auto_seed] Phase B 启动 —— multi_elaborate (N={n_candidates})",
          file=sys.stderr)
    proposals = multi_elaborate(
        skeleton=skeleton, pack=pack, llm=elaborator_llm,
        n_candidates=n_candidates,
    )
    if not proposals:
        print("# 错误：multi_elaborate 未产出任何 proposal", file=sys.stderr)
        sys.exit(3)
    print(f"[auto_seed] Phase B 完成: 生 {len(proposals)} 个 candidate", file=sys.stderr)

    # ---- Phase B.5: SciMON novelty boost (对每个 candidate) ----
    from darwinian.agents.novelty_booster import boost_novelty
    print(f"[auto_seed] Phase B.5 启动 —— SciMON novelty boost x {len(proposals)}",
          file=sys.stderr)
    boosted_proposals = []
    disqualified_titles = []
    for i, p in enumerate(proposals):
        print(f"[auto_seed] novelty boost candidate {i+1}/{len(proposals)}",
              file=sys.stderr)
        boosted, boost_result = boost_novelty(
            proposal=p, direction=direction, llm=extractor_llm, max_rounds=3,
        )
        boosted_proposals.append(boosted)
        if not boost_result.converged:
            disqualified_titles.append(boosted.title or f"proposal_{i}")
    print(f"[auto_seed] Phase B.5 完成: {len(boosted_proposals)} 个，"
          f"disqualified {len(disqualified_titles)} 个 (撞车未收敛)",
          file=sys.stderr)

    # ---- Phase C: pairwise tournament ----
    print(f"[auto_seed] Phase C 启动 —— pairwise tournament", file=sys.stderr)
    tournament = run_tournament(boosted_proposals, llm=extractor_llm, top_k=2)
    tournament.disqualified_ids = disqualified_titles
    print(f"[auto_seed] Phase C 完成: top_2 = {tournament.top_k_ids}", file=sys.stderr)

    # ---- 渲染：top-1 主输出 + 全部 candidates 概览 ----
    from darwinian.tools.seed_renderer import render_proposal, render_tournament_overview
    title_to_proposal = {(p.title or f"proposal_{i}"): p
                         for i, p in enumerate(boosted_proposals)}
    # 选 top-1 但优先非 disqualified
    top_proposal = None
    for tid in tournament.top_k_ids:
        if tid not in tournament.disqualified_ids and tid in title_to_proposal:
            top_proposal = title_to_proposal[tid]
            break
    if top_proposal is None:
        # 全部 disqualified → 退而求其次取 elo 最高
        top_proposal = title_to_proposal.get(
            tournament.top_k_ids[0] if tournament.top_k_ids else "",
            boosted_proposals[0],
        )

    # ---- R9c: Feasibility Challenger (仅 top-1) ----
    from darwinian.agents.feasibility_challenger import challenge_feasibility
    print(f"[auto_seed] R9c 启动 —— Feasibility Challenger 攻击 top-1", file=sys.stderr)
    fc = challenge_feasibility(
        proposal=top_proposal, constraints=constraints, llm=extractor_llm,
    )
    if fc is not None:
        top_proposal.feasibility_challenge = fc
        print(
            f"[auto_seed] R9c 完成: verdict={fc.overall_verdict}, "
            f"risks={len(fc.risks)} (high={sum(1 for r in fc.risks if r.severity == 'high')})",
            file=sys.stderr,
        )
    else:
        print(f"[auto_seed] R9c 失败（LLM error），跳过", file=sys.stderr)

    # ---- R11: Mechanism Alignment Checker (仅 top-1, 仅 cross-domain 触发) ----
    from darwinian.agents.mechanism_alignment_checker import check_mechanism_alignment
    print(f"[auto_seed] R11 启动 —— Mechanism Alignment 跨域类比 critique", file=sys.stderr)
    ma = check_mechanism_alignment(proposal=top_proposal, llm=extractor_llm)
    if ma is not None:
        top_proposal.mechanism_alignment = ma
        print(
            f"[auto_seed] R11 完成: is_cross_domain={ma.is_cross_domain}, "
            f"verdict={ma.overall_verdict}",
            file=sys.stderr,
        )
    else:
        print(f"[auto_seed] R11 失败（LLM error），跳过", file=sys.stderr)

    md = render_proposal(top_proposal, material_pack=pack)
    md += "\n\n" + render_tournament_overview(tournament, boosted_proposals)
    print(md)

    # ---- S2 调用统计 (Pri-6) ----
    from darwinian.tools.semantic_scholar import get_s2_stats
    stats = get_s2_stats()
    print(
        f"[auto_seed] S2 stats: {stats['total_lookups']} lookups "
        f"(cache hit rate: {stats['cache_hit_rate']*100:.1f}%; "
        f"http: {stats['http_calls']}, 429: {stats['http_429s']}, "
        f"failures: {stats['http_failures']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

"""
Phase A Orchestrator — 编排"读"阶段，从研究方向自动产出 ResearchMaterialPack。

设计目标：把现有 4 个工具串成单一入口，给 elaborator 喂完美素材包：

  build_concept_graph         (S2 搜索 + 一跳引用扩展 + 实体抽取 + 结构洞)
       ↓
  resolve_arxiv_ids           (对 top-K paper 查 externalIds.ArXiv)
       ↓
  fetch_arxiv_latex + render_for_llm  (拉全文 LaTeX 节选)
       ↓
  batch_extract_evidence      (paper_evidence_extractor 深抽取五元组)
       ↓
  组装 ResearchMaterialPack（含 paper_evidence + concept_graph + timeline_signals）

未实现（留给后续）：
  - structural_hole_hooks 的 hook_text 自动生成（需新写 hook_writer LLM 调用）
  - prior_failures 接 state.failed_ledger
"""

from __future__ import annotations

import os
import sys
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from darwinian.state import (
    PaperEvidence,
    PaperInfo,
    ResearchConstraints,
    ResearchMaterialPack,
)
from darwinian.agents.hook_writer import write_structural_hole_hooks
from darwinian.agents.phenomenon_miner import batch_mine_phenomena
from darwinian.agents.contradiction_detector import detect_cross_paper_contradictions
from darwinian.tools.arxiv_latex_fetcher import fetch_arxiv_latex, render_for_llm
from darwinian.tools.paper_evidence_extractor import batch_extract_evidence
from darwinian.tools.semantic_scholar import (
    GRAPH_FIELDS,
    get_citations,
    get_paper_details,
    get_references,
    search_papers,
)
from darwinian.utils.json_parser import parse_llm_json
from darwinian.utils.llm_retry import invoke_with_retry
from darwinian.utils.knowledge_graph import build_concept_graph, _dedup_papers_by_id
from darwinian.utils.similarity import compute_cosine_similarity, get_text_embedding


class PhaseAAbortError(RuntimeError):
    """
    R12: Phase A 检出"研究方向在 S2 上没有足够真相关论文"时抛出。

    场景：v2 / v3 LIVE 实测发现某些 direction（如冷门交叉领域）在 S2 上
    本来就没有 ≥N 篇真相关论文，elaborator 被迫拿 orthogonal 兜底论文当主弹药，
    产出 hand-waved cross-domain 类比。R10 加了 banner 警告但 elaborator 不
    总听话。R12 直接 hard abort，不浪费下游 ~80 LLM call (¥2-3) 跑出垃圾 seed。

    阈值由 env var DARWINIAN_PHASE_A_HARD_ABORT_MIN 控制（默认 0=不 abort）：
    0 = 关闭（保留旧行为：banner 警告但跑完）
    N>0 = 真相关 < N 时 raise，主程序 catch 后让用户换 direction
    """
    pass


# ===========================================================================
# Scheme X: build_seed_pool —— LLM 列方向 seed → S2 verify → 一跳 → rerank
# 替换 S2 keyword search 路径，从根上解决 candidate pool 缺方向论文的问题
# ===========================================================================

_LLM_KEYWORDS_PROMPT = """你是科研助理。给定一个研究方向，**不要列论文**（v6 LIVE 实测 LLM 大量
编造不存在的 arxiv_id：列了 "EncT/ConAda/AdaETC" 等貌似合理的名字，但 arxiv_id 解析到
水下图像/数学/医学论文上）。改为列出 **8-10 组高质量 S2 搜索关键词**，让 S2 真实检索召回。

输出严格 JSON：
{
  "queries": [
    "encrypted traffic classification concept drift",
    "website fingerprinting domain adaptation tor",
    "ET-BERT network traffic transformer",
    "FlowPrint AppScanner mobile encrypted traffic",
    "ISCXVPN2016 encrypted traffic benchmark",
    "CICIDS-2017 network intrusion drift detection",
    "Mirage-2019 mobile encrypted traffic dataset"
  ]
}

【设计原则】
1. 每个 query 是 multi-token search query（3-7 词），不是单个 broad 词
2. queries 互补不冗余——覆盖方向不同子赛道 / 不同方法谱系
3. **领域关键词 + 方法关键词组合**：
   * 方向 = "encrypted traffic classification under concept drift"
     ✅ "encrypted traffic concept drift detection"
     ✅ "website fingerprinting domain adaptation"
     ❌ "machine learning"（太泛）
4. 优先用**论文标题里实际会出现的术语**
5. 可加方法名 / 系统名（如 "ET-BERT" / "FlowPrint" / "DSEC"）让 S2 精确召回

【❗❗ R18 必加 — Benchmark/Dataset 显式查询】
**必须**包含 ≥2 组针对该领域真实 benchmark / dataset 名的 query。
LLM 凭直觉列方法名容易，但容易漏掉 dataset paper —— 这些通常被该领域所有论文 cite，
是召回真相关 prior work 的关键路径。

举例（不同领域的真实 benchmark 名）：
- 加密流量：ISCXVPN2016 / CICIDS-2017 / CICAndMal2017 / Mirage-2019 / CSTNet /
  CIRA-CIC-DoHBrw2020 / NetML-2020 / TLS-Tracker
- 分子属性预测：MoleculeNet / ChEMBL / ZINC / OGB-MolHIV / QM9
- LLM 推理加速：HumanEval / MT-Bench / MMLU / GSM8K

query 里**直接写 dataset 名 + 方向关键词**：
✅ "ISCXVPN2016 encrypted traffic concept drift"
✅ "CICIDS-2017 intrusion detection drift adaptation"
✅ "MoleculeNet GNN molecular property"
❌ "encrypted traffic dataset"（没具体 dataset 名）

如果你不知道该领域真实 benchmark，**宁可少列 query 也不要编**——编了 dataset 名
S2 search 命中 0 篇，浪费 query 槽位。

输出严格 JSON，不要 markdown 包裹。

【❗ 输出节流】
- 不要写 <think> 块，不要 reasoning
- 第一个 token 就开始 `{`
"""


_LLM_SEED_PROMPT = """你是科研助理。给定一个研究方向，列出该方向**最重要**的 8-15 篇论文 seed
（必须是真实存在的、已发表/arxiv 可查的论文），每篇含 arxiv_id + 完整 title + 一句话理由。

输出严格 JSON：
{
  "seed_papers": [
    {"arxiv_id": "2404.16710", "title": "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", "reason": "..."},
    ...
  ]
}

【❗❗ 第一原则 — application domain anchoring（R16 实测必加）】
seed 论文**必须明确属于研究方向所述的具体应用领域**，不要列只是"在该领域可能有用"
的通用 ML 方法论文。

举几个实测踩过的反面例子：
- 方向 = "encrypted traffic classification under concept drift"
  ❌ 错：A2T (NLP adversarial training) / Pure CLIP NeRF (3D vision) / LoRA (通用 PEFT)
       — 这些**不是**加密流量论文，只是"也许能借鉴"的通用技术
  ✅ 对：DSEC (Drift-oriented Self-evolving Encrypted Traffic Classification, 2024) /
       INSOMNIA (encrypted traffic concept drift, 2023) / DA-WF (Domain Adaptation
       Website Fingerprinting) / FlowPrint / AppScanner / FlowPic / DeepFlow /
       FS-Net / ETC-PROTOTYPE
       — 这些是 encrypted traffic / network classification 领域里直接处理 concept
       drift 或 distribution shift 的论文

- 方向 = "molecular property prediction with graph neural networks"
  ❌ 错：原版 GAT/GCN 论文 (2017-2018，太基础) / Llama (LLM 基础) / SimCLR (通用对比学习)
  ✅ 对：MolCLR / GROVER / GEM / Uni-Mol / MoleculeNet / D-MPNN — 都是分子图领域
       直接相关的工作

- 方向 = "speculative decoding for LLM inference"
  ❌ 错：FlashAttention (通用 attention 优化) / GPT-3 paper (基础工作)
  ✅ 对：LayerSkip / DEL / Medusa / EAGLE / EAGLE-2 / QSpec / SpecInfer / Sequoia /
       Lookahead / DistillSpec — 都是 speculative decoding 子赛道里的真工作

【关键约束】
1. arxiv_id 形如 'YYMM.NNNNN'（如 '2404.16710'），不带 v1/v2 后缀，不带 'arxiv:' 前缀
2. title 写**完整官方标题**（含子标题），用于 fallback verify
3. 优先选近 2-3 年（2023-2026）方向核心工作
4. 每个 reason **必须**说明这篇论文在 application domain 里**具体解决什么问题**
   （而非"可能有用"），如 "DSEC: 第一篇专门做 encrypted traffic 实时 drift detection
   并 self-evolving 的方法"
5. 不要列纯通用 ML 方法（Adam / Transformer / LoRA / Adam / PyTorch / Llama 2）
   除非它们就是方向本身
6. **列 8-15 篇，宁缺勿滥**：如果某个方向的应用领域真的没那么多论文，
   宁可只列 5-8 篇真相关，**不要凑数列通用 ML 方法**

输出严格 JSON，不要 markdown 包裹。

【❗ 输出节流（R14 实测必加）】
- **不要写 <think> 块**，**不要 reasoning out loud**
- 不要复述任务、不要列 schema 示例、不要解释你的选择过程
- 第一个 token 就开始 `{`，最后一个 token 是 `}`

(Critical: Output JSON IMMEDIATELY. NO <think> blocks. NO reasoning. NO restating
the task. NO schema examples. Start the response with `{` directly.)
"""


def build_seed_pool(
    direction: str,
    llm: BaseChatModel,
    *,
    n_seeds: int = 15,
    refs_per_seed: int = 8,
    cits_per_seed: int = 8,
    final_pool_size: int = 50,
) -> list[dict]:
    """
    Scheme X 主入口。返回 list of S2-style paper dict。

    R17 重构：默认走 **keyword search 策略**（v6 LIVE 实测 LLM 大量编造 arxiv_id：
    LLM 列了 15 个看起来真的 encrypted traffic 论文 title，但 arxiv_id 实际指向
    水下图像/数学/医学论文。"paper 策略" 完全不可靠）。改成让 LLM 列 keywords
    → S2 search 拉真实存在的论文，绕过 LLM 论文编造问题。

    env DARWINIAN_SEED_STRATEGY:
      "keyword" (默认) = LLM 列 search queries → S2 search
      "paper"  = LLM 列论文 (旧策略，保留作 fallback / 调试用)
    """
    strategy = os.environ.get("DARWINIAN_SEED_STRATEGY", "keyword").lower()
    if strategy == "paper":
        return _build_seed_pool_paper_strategy(
            direction, llm, n_seeds=n_seeds, refs_per_seed=refs_per_seed,
            cits_per_seed=cits_per_seed, final_pool_size=final_pool_size,
        )
    # default: keyword strategy
    return _build_seed_pool_keyword_strategy(
        direction, llm,
        refs_per_seed=refs_per_seed, cits_per_seed=cits_per_seed,
        final_pool_size=final_pool_size,
    )


def _build_seed_pool_keyword_strategy(
    direction: str,
    llm: BaseChatModel,
    *,
    refs_per_seed: int = 8,
    cits_per_seed: int = 8,
    final_pool_size: int = 50,
    per_query_limit: int = 15,
) -> list[dict]:
    """
    R17 keyword 策略：LLM 列 5-7 组 search queries → S2 search → 合并去重 → rerank。

    优点：所有 paper 都是 S2 上真实存在的，避免 LLM 编造 arxiv_id 的问题。
    """
    queries = _llm_list_search_keywords(direction, llm)
    print(f"[seed_pool] LLM 列出 {len(queries)} 组 search queries", file=sys.stderr)
    for i, q in enumerate(queries, 1):
        print(f"[seed_pool]   Q{i}: {q}", file=sys.stderr)

    if not queries:
        return []

    # S2 search 每条 query
    seen_ids: set = set()
    seeds: list[dict] = []
    for q in queries:
        try:
            results = search_papers(q, limit=per_query_limit)
        except Exception as e:
            print(f"[seed_pool] search '{q}' 失败: {type(e).__name__}", file=sys.stderr)
            continue
        added = 0
        for r in results:
            pid = r.get("paperId")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            seeds.append(r)
            added += 1
        print(f"[seed_pool]   Q{queries.index(q)+1} 新增 {added} 篇 "
              f"(累计 {len(seeds)})", file=sys.stderr)

    if not seeds:
        return []

    # R16 sim 过滤（防止 keyword 太泛拉到不相关）
    seeds = _filter_seeds_by_direction_similarity(seeds, direction)
    print(f"[seed_pool] sim 过滤后剩 {len(seeds)} 篇", file=sys.stderr)
    if not seeds:
        return []

    # 一跳扩展 + rerank（沿用 paper 策略的下半截）
    expanded = _expand_seeds_one_hop(seeds, refs_per_seed, cits_per_seed)
    print(f"[seed_pool] 一跳扩展后候选池 {len(expanded)} 篇", file=sys.stderr)
    seed_ids = {s.get("paperId", "") for s in seeds if s.get("paperId")}
    ranked = _rerank_by_direction_relevance(expanded, direction, seed_ids)
    return ranked[:final_pool_size]


def _llm_list_search_keywords(
    direction: str,
    llm: BaseChatModel,
    *,
    max_attempts: int = 3,
) -> list[str]:
    """
    R17: 让 LLM 列 5-7 组 S2 search queries。复用 R14 的 retry + anti-reasoning。
    """
    base_human = (
        f"研究方向：{direction}\n\n"
        f"请列 5-7 组高质量 S2 搜索关键词。"
    )
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        human_msg = base_human
        if attempt > 1:
            human_msg += (
                "\n\n【重试 — 上次输出无法解析】"
                "**直接以 `{` 开头**，不要 <think> 块或 reasoning。"
            )
        try:
            response = invoke_with_retry(llm, [
                SystemMessage(content=_LLM_KEYWORDS_PROMPT),
                HumanMessage(content=human_msg),
            ])
            raw = parse_llm_json(response.content)
            queries = raw.get("queries") or []
            valid = [
                q.strip() for q in queries
                if isinstance(q, str) and q.strip() and len(q.split()) >= 2
            ]
            if valid:
                if attempt > 1:
                    print(
                        f"[seed_pool] _llm_list_search_keywords 第 {attempt}/{max_attempts} "
                        f"次重试成功，{len(valid)} 个 query",
                        file=sys.stderr,
                    )
                return valid
            last_err = "queries 为空或 token 数 <2"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
        print(
            f"[seed_pool] _llm_list_search_keywords 第 {attempt}/{max_attempts} 次失败: "
            f"{last_err}",
            file=sys.stderr,
        )
    return []


def _build_seed_pool_paper_strategy(
    direction: str,
    llm: BaseChatModel,
    *,
    n_seeds: int = 15,
    refs_per_seed: int = 8,
    cits_per_seed: int = 8,
    final_pool_size: int = 50,
) -> list[dict]:
    """
    旧 paper 策略：LLM 列 15 篇论文 (arxiv_id + title) → verify by arxiv_id → 一跳 → rerank。

    v6 LIVE 实测：LLM 大量编造 arxiv_id (列了真的 encrypted traffic title 但
    arxiv_id 解析到水下图像/数学/医学论文上)。R17 已默认改用 keyword 策略，
    本函数保留供 env DARWINIAN_SEED_STRATEGY=paper 调试 / 对照用。
    """
    candidates = _llm_list_seed_papers(direction, llm, n=n_seeds)
    print(f"[seed_pool] LLM 列出 {len(candidates)} 个 seed 候选", file=sys.stderr)
    for i, c in enumerate(candidates[:20], 1):
        print(
            f"[seed_pool]   #{i:2d} arxiv:{c.get('arxiv_id', '?')} | "
            f"{(c.get('title') or '')[:90]}",
            file=sys.stderr,
        )

    seeds: list[dict] = []
    for cand in candidates:
        paper = _verify_and_recover_seed(cand)
        if paper:
            seeds.append(paper)
    print(f"[seed_pool] verify 通过 {len(seeds)}/{len(candidates)} 个 seed", file=sys.stderr)

    if not seeds:
        return []

    # R16: Step 2.5 — 跨域过滤，丢 abstract sim 太低的 seed（防 LLM 列错域）
    seeds = _filter_seeds_by_direction_similarity(seeds, direction)
    print(f"[seed_pool] 过滤跨域后剩 {len(seeds)} 个 seed", file=sys.stderr)
    if not seeds:
        return []

    # Step 3: 一跳扩展
    expanded = _expand_seeds_one_hop(seeds, refs_per_seed, cits_per_seed)
    print(f"[seed_pool] 一跳扩展后候选池 {len(expanded)} 篇", file=sys.stderr)

    # Step 4: rerank
    seed_ids = {s.get("paperId", "") for s in seeds if s.get("paperId")}
    ranked = _rerank_by_direction_relevance(expanded, direction, seed_ids)
    return ranked[:final_pool_size]


def _filter_seeds_by_direction_similarity(
    seeds: list[dict],
    direction: str,
    *,
    min_sim: float = 0.30,
) -> list[dict]:
    """
    R16: verify 通过后再用 embedding sim 二次过滤，防 LLM 列错应用域。

    sim 计算 paper.title+abstract vs direction。低于 min_sim 视为跨域 seed
    (例: direction=encrypted traffic, LLM 列了 A2T 这种 NLP adversarial 论文)，
    丢弃。

    阈值 0.30 是经验值——在加密流量这种窄域 direction 上，真相关 seed sim 通常
    >0.40，泛 ML 论文 <0.25。允许 5 个 seed 全过滤掉返空（让上游 R15 接管 abort）。
    """
    if not seeds:
        return []
    direction_emb = get_text_embedding(direction)
    kept: list[dict] = []
    dropped_sim_log: list[tuple[float, str]] = []
    for s in seeds:
        text = ((s.get("title") or "") + " " + (s.get("abstract") or "")).strip()
        # 文本太短（<30 字符，通常只有 title 没 abstract）→ 无法可靠判断 sim,
        # 保守保留（防止 mock 数据 / S2 abstract 缺失场景误杀真相关 seed）
        if len(text) < 30:
            kept.append(s)
            continue
        sim = compute_cosine_similarity(direction_emb, get_text_embedding(text))
        if sim >= min_sim:
            kept.append(s)
        else:
            dropped_sim_log.append((sim, (s.get("title") or "")[:60]))
    if dropped_sim_log:
        print(
            f"[seed_pool] R16 跨域过滤丢 {len(dropped_sim_log)} 篇 (sim<{min_sim}): "
            + "; ".join(f"sim={s:.2f} {t}" for s, t in dropped_sim_log[:5]),
            file=sys.stderr,
        )
    return kept


def _llm_list_seed_papers(
    direction: str,
    llm: BaseChatModel,
    *,
    n: int = 15,
    max_attempts: int = 3,
) -> list[dict]:
    """
    让 LLM 列 N 个方向 seed，返回 list of {arxiv_id, title, reason}。

    R14: v3/v4 LIVE 实测 MiniMax-M2.7 偶发在 <think> 块里写超长 reasoning，
    max_tokens 用完时 JSON 还没出。加 retry：第 2/3 次失败用更强 anti-reasoning
    指令让 LLM 直接输出。
    """
    base_human = (
        f"研究方向：{direction}\n\n请列出 {n} 篇该方向核心 seed 论文。"
    )
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        # 第 2/3 次重试时追加更强约束
        human_msg = base_human
        if attempt > 1:
            human_msg += (
                "\n\n【重试 — 上次输出被截断】"
                "上次你写了 <think> 推理块，导致 max_tokens 用尽 JSON 没输出。"
                "**直接以 `{` 开头**，不要任何 reasoning。"
            )
        try:
            response = invoke_with_retry(llm, [
                SystemMessage(content=_LLM_SEED_PROMPT),
                HumanMessage(content=human_msg),
            ])
            raw = parse_llm_json(response.content)
            candidates = raw.get("seed_papers") or []
            valid = [c for c in candidates if isinstance(c, dict) and c.get("arxiv_id")]
            if valid:
                if attempt > 1:
                    print(
                        f"[seed_pool] _llm_list_seed_papers 第 {attempt}/{max_attempts} "
                        f"次重试成功，返回 {len(valid)} 个 seed",
                        file=sys.stderr,
                    )
                return valid
            last_err = "JSON 解析成功但 seed_papers 为空"
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
        print(
            f"[seed_pool] _llm_list_seed_papers 第 {attempt}/{max_attempts} 次失败: "
            f"{last_err}",
            file=sys.stderr,
        )
    return []


def _verify_and_recover_seed(cand: dict) -> dict | None:
    """
    机制 1: arxiv_id verify + title fuzzy fallback。
    LLM 经常把 arxiv_id 末几位记错（2404.16710 → 2404.16701）。
    直接 verify 失败 → 用 title 搜索 S2 回捞。
    """
    arxiv_id = (cand.get("arxiv_id") or "").strip().lstrip("arxiv:").lstrip("ArXiv:")
    title = (cand.get("title") or "").strip()

    # 1. 用 arxiv_id 直查
    if arxiv_id:
        try:
            detail = get_paper_details(f"ArXiv:{arxiv_id}", fields=GRAPH_FIELDS)
            if detail:
                return detail
        except Exception:
            pass

    # 2. fallback: 用 title 搜 S2
    if title:
        try:
            results = search_papers(title, limit=3)
            for r in results:
                if _title_similarity(r.get("title", "") or "", title) > 0.85:
                    return r
        except Exception:
            pass

    return None


def _title_similarity(a: str, b: str) -> float:
    """归一化 token Jaccard，简单但对论文标题足够"""
    import re
    if not a or not b:
        return 0.0
    tokens_a = set(re.findall(r"\w+", a.lower()))
    tokens_b = set(re.findall(r"\w+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _expand_seeds_one_hop(
    seeds: list[dict],
    refs_per_seed: int,
    cits_per_seed: int,
) -> list[dict]:
    """
    对每个 seed 取 references + citations。比 expand_one_hop 简单：
    - 不需要复杂的累加去重（直接 dedup_papers_by_id）
    - 控制每个 seed 的 refs/cits 数量在 8-10 内（防 S2 限流）
    """
    pool = list(seeds)   # seeds 自己也进 pool
    for s in seeds:
        pid = s.get("paperId", "")
        if not pid:
            continue
        try:
            pool.extend(get_references(pid, limit=refs_per_seed) or [])
        except Exception as e:
            print(f"[seed_pool] get_references({pid}) 失败: {type(e).__name__}",
                  file=sys.stderr)
        try:
            pool.extend(get_citations(pid, limit=cits_per_seed) or [])
        except Exception as e:
            print(f"[seed_pool] get_citations({pid}) 失败: {type(e).__name__}",
                  file=sys.stderr)
    return _dedup_papers_by_id(pool)


def _rerank_by_direction_relevance(
    papers: list[dict],
    direction: str,
    seed_ids: set,
) -> list[dict]:
    """
    机制 2: rerank 按 (is_seed, TF-IDF sim to direction, citation) 降序。
    避免 filter_and_rank 的纯 citation 排序把方向论文（cit 低）甩到末尾。
    """
    direction_emb = get_text_embedding(direction)

    def sort_key(p: dict) -> tuple:
        pid = p.get("paperId") or ""
        is_seed = pid in seed_ids
        text = ((p.get("title") or "") + " " + (p.get("abstract") or "")).strip()
        sim = (
            compute_cosine_similarity(direction_emb, get_text_embedding(text))
            if text else 0.0
        )
        cit = p.get("citationCount") or 0
        return (is_seed, sim, cit)

    return sorted(papers, key=sort_key, reverse=True)


# ===========================================================================
# Phase A 主入口
# ===========================================================================


def build_research_material_pack(
    direction: str,
    constraints: ResearchConstraints,
    extractor_llm: BaseChatModel,
    evidence_llm: BaseChatModel,
    *,
    top_k_evidence: int = 12,
    backend: str | None = None,
) -> ResearchMaterialPack:
    """
    Phase A 主入口：研究方向 + 约束 → 完整 ResearchMaterialPack。

    Args:
        direction: 用户输入的研究方向原文
        constraints: 资源/合规约束（已结构化）
        extractor_llm: 用于浅实体抽取（推荐 Haiku 级小模型，便宜）
        evidence_llm:  用于 paper_evidence_extractor 深抽取（推荐 Opus / GPT-4 级）
        top_k_evidence: 深抽取的论文数量上限（默认 12，按 citationCount 取头）
        backend: build_concept_graph 的检索后端 ("s2" / "arxiv" / None=env)

    Returns:
        完整 ResearchMaterialPack，可直接喂给 elaborate_proposal_from_pack
    """
    # ---- Step 0: build_seed_pool (Scheme X) —— 用 LLM 知识列方向 seed ----
    # 替换 S2 keyword search 路径。S2 关键词搜索对 "LLM inference" 这种宽 query
    # 会命中所有 LLM 大论文（GPT-3 / Llama 2 / PyTorch），把方向相关的小众论文
    # （LayerSkip / DEL / QSpec）压在 100+ 名外。LLM 列 seed 直接精确命中方向。
    print(f"[phase_a] Step 0/4: build_seed_pool（LLM 列方向 seed + 一跳扩展 + rerank）",
          file=sys.stderr)
    seed_pool = build_seed_pool(direction, extractor_llm)
    print(f"[phase_a] seed_pool: {len(seed_pool)} 篇候选", file=sys.stderr)

    # ---- Step 1: 实体表 + 结构洞（用 seed_pool 跳过 build_concept_graph 的搜索层）----
    print(f"[phase_a] Step 1/4: build_concept_graph(seed_pool=<{len(seed_pool)} papers>)",
          file=sys.stderr)
    graph = build_concept_graph(
        research_direction=direction,
        core_problem=direction,
        llm=extractor_llm,
        backend=backend,
        seed_pool=seed_pool,
    )
    print(f"[phase_a] graph: {len(graph.papers)} papers, "
          f"{len(graph.entities)} entities, {len(graph.limitations)} limitations, "
          f"{len(graph.novel_pair_hints)} novel pairs", file=sys.stderr)

    # ---- Step 1.5: hook_writer 把 EntityPair 升级为 StructuralHoleHook ----
    # elaborator prompt 里【结构洞 hooks】section 之前永远空（structural_hole_hooks=[]），
    # 现在用 LLM 把 top-K novel pair 升级为带叙事 + relation_type 的 hook
    print(f"[phase_a] Step 1.5/4: hook_writer 升级 top-{min(5, len(graph.novel_pair_hints))} novel pairs",
          file=sys.stderr)
    structural_hole_hooks = write_structural_hole_hooks(
        graph.novel_pair_hints,
        graph.entities,
        graph.papers,
        direction,
        extractor_llm,
        max_hooks=5,
    )
    print(f"[phase_a] hooks: 生成 {len(structural_hole_hooks)} 条 StructuralHoleHook",
          file=sys.stderr)

    # ---- Step 2: 选 top-K paper 做深抽取 ----
    # 排序键：(entity_hits, citation_count) 降序
    # entity_hits = 这篇论文贡献了多少个不同的 entity（method/dataset/metric）
    # 设计动机：纯按 citation 排会让 GPT-3 / Llama 2 / PyTorch 等基础工作压过
    #   方向相关的小众论文（LayerSkip / DEL / QSpec），因为基础论文引用量比方向
    #   论文高一个数量级。entity_hits 反映"实体抽取认为多相关"，是更准的方向
    #   匹配信号。citation_count 作为同分时的 tiebreaker 保留。
    top_papers = _select_top_papers_by_relevance(graph, top_k_evidence)
    print(f"[phase_a] Step 2/4: 选 top-{len(top_papers)} 篇做深抽取 "
          f"（按 entity hits + citation 排序）", file=sys.stderr)

    # ---- Step 3: 解析 arxiv_id（用于全文拉取）----
    arxiv_id_by_paperid = _resolve_arxiv_ids(top_papers)
    n_with_arxiv = sum(1 for v in arxiv_id_by_paperid.values() if v)
    print(f"[phase_a] Step 3/4: {n_with_arxiv}/{len(top_papers)} 篇可拿 arxiv 全文", file=sys.stderr)

    # ---- Step 4: 深抽取 PaperEvidence ----
    full_text_provider = _make_full_text_provider(arxiv_id_by_paperid)
    papers_for_extraction = [
        {
            "paper_id": _format_evidence_id(p, arxiv_id_by_paperid),
            "title": p.title,
            "abstract": p.abstract,
        }
        for p in top_papers
    ]
    print(f"[phase_a] Step 4/4: batch_extract_evidence(n={len(papers_for_extraction)})",
          file=sys.stderr)
    paper_evidence = batch_extract_evidence(
        papers_for_extraction,
        direction=direction,
        llm=evidence_llm,
        full_text_provider=full_text_provider,
    )
    print(f"[phase_a] 深抽取完成: {len(paper_evidence)} 篇成功", file=sys.stderr)

    # ---- Step 4.2: relevance gate (Pri-5) ----
    # 用 paper_evidence_extractor 已抽出的 relation_to_direction 字段过滤无关论文
    # （v9 实测 Mamba / Copy-as-Decode 被 elaborator 当 padding 引用的问题）
    strict_mode = os.environ.get("DARWINIAN_EVIDENCE_STRICT", "0") == "1"
    rel_stats: dict = {}
    paper_evidence = _filter_relevant_evidence(
        paper_evidence, strict=strict_mode, out_stats=rel_stats,
    )
    relevance_warning = _build_relevance_warning(
        rel_stats.get("truly_relevant", 0),
        rel_stats.get("backfilled", 0),
    )
    if relevance_warning:
        print(f"[phase_a] ⚠️ relevance_warning: {relevance_warning}", file=sys.stderr)

    # R12: 真相关 < HARD_ABORT_MIN → 直接 raise，不浪费下游 ~80 LLM call
    # 默认 env var 不设 = 0 = 关闭（保留 v3 旧行为）
    try:
        hard_abort_min = int(os.environ.get("DARWINIAN_PHASE_A_HARD_ABORT_MIN", "0"))
    except ValueError:
        hard_abort_min = 0
    n_truly = rel_stats.get("truly_relevant", 0)
    if hard_abort_min > 0 and n_truly < hard_abort_min:
        raise PhaseAAbortError(
            f"Phase A 真相关论文数 {n_truly} < 阈值 {hard_abort_min}（"
            f"DARWINIAN_PHASE_A_HARD_ABORT_MIN={hard_abort_min}）。"
            f"该 direction 在 S2 上没足够真相关论文，elaborator 会被迫拿 "
            f"orthogonal 论文硬拗 cross-domain 类比。建议换更聚焦的 sub-direction "
            f"（参考 R10 banner 建议）。如要继续跑，set "
            f"DARWINIAN_PHASE_A_HARD_ABORT_MIN=0 关闭硬退出。"
        )

    # ---- Step 4.5: phenomenon_miner 从全文挖"未解释/意外"现象 ----
    # 比 entity 组合更深的 idea seed（13 个 SOTA 系统都没做的差异化能力）
    print(f"[phase_a] Step 4.5/5: phenomenon_miner 挖论文现象", file=sys.stderr)
    phenomena = batch_mine_phenomena(
        papers_for_extraction,
        llm=extractor_llm,
        full_text_provider=full_text_provider,
        max_per_paper=3,
    )
    print(f"[phase_a] phenomena: 抽到 {len(phenomena)} 条现象", file=sys.stderr)

    # ---- Step 4.6: cross_paper_contradiction (R9d) ----
    # 纯规则零 LLM call，扫 quantitative_claims 找跨论文数值矛盾
    contradictions = detect_cross_paper_contradictions(paper_evidence)
    if contradictions:
        print(
            f"[phase_a] contradictions: 检出 {len(contradictions)} 条跨论文矛盾",
            file=sys.stderr,
        )
        phenomena.extend(contradictions)

    # ---- R15: 默认硬熔断 — 0 真相关 + 0 phenomena = 没有任何科学依据可写 ----
    # v3/v4/v5 三次实战证明这种 case 下 elaborator 一定硬拗 orthogonal 论文。
    # 跟 R12（env var opt-in）不同，这是默认开启的"绝对极端 case"防御。
    # env var DARWINIAN_PHASE_A_ALLOW_ZERO_EVIDENCE=1 可关闭（debug / 测试 / 渲染验证）
    allow_zero = os.environ.get("DARWINIAN_PHASE_A_ALLOW_ZERO_EVIDENCE", "0") == "1"
    if not allow_zero and n_truly == 0 and len(phenomena) == 0:
        raise PhaseAAbortError(
            f"Phase A 硬熔断：0 篇真相关论文 + 0 个 phenomena（绝对没有科学依据可写）。"
            f"这种 case 下 elaborator 一定硬拗 orthogonal 兜底论文产出 hand-waved seed。"
            f"建议换更聚焦的 sub-direction，让 Phase A 拉到真相关论文。如要绕过此熔断"
            f"（仅 debug / 渲染测试用），set DARWINIAN_PHASE_A_ALLOW_ZERO_EVIDENCE=1。"
        )

    # ---- Step 5: timeline_signals ----
    timeline = _bucket_by_year(top_papers, arxiv_id_by_paperid)

    return ResearchMaterialPack(
        direction=direction,
        constraints=constraints,
        paper_evidence=paper_evidence,
        concept_graph=graph,
        structural_hole_hooks=structural_hole_hooks,
        phenomena=phenomena,
        timeline_signals=timeline,
        prior_failures=[],
        relevance_warning=relevance_warning,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPAND_QUERIES_SYSTEM_PROMPT = """你是 academic search query 专家。给定一个研究方向描述，
你要拆出 3-5 个**具体且互补的**学术搜索关键词组，覆盖该方向的核心子赛道和方法名。

输出严格 JSON：
{"queries": ["query1", "query2", ...]}

【关键约束】
1. 每条 query 是 3-8 个英文词的关键词组（不是一句话）
2. 优先使用方向里出现的具体方法名 / 数据集 / 现象，而不是宽词
   - 不好: "LLM efficient inference"  → S2 会命中所有 LLM 论文
   - 好:   "self-speculative decoding draft model"
3. 每条 query 应能命中 8-30 篇方向相关论文（不要太宽不要太窄）
4. 多条 query 之间应覆盖不同子赛道，不要互相重复
5. 输出 3-5 条，不超过 5 条

示例（方向='LLM inference acceleration speculative decoding quantization'）：
{"queries": [
  "self-speculative decoding draft model",
  "speculative decoding tree attention LLM",
  "Medusa EAGLE LayerSkip inference",
  "mixed precision quantization LLM",
  "early exit transformer inference"
]}

输出严格 JSON，不要 markdown 包裹。
"""


def _expand_search_queries(
    direction: str,
    llm: BaseChatModel,
    *,
    max_queries: int = 5,
) -> list[str]:
    """
    用便宜 LLM 把宽研究方向拆成 3-5 条具体子查询。

    LLM 失败时返空列表（让 build_concept_graph 退化为单查询行为，0 回归风险）。
    """
    try:
        response = invoke_with_retry(llm, [
            SystemMessage(content=_EXPAND_QUERIES_SYSTEM_PROMPT),
            HumanMessage(content=f"研究方向：{direction}\n\n请输出 3-5 条具体子查询。"),
        ])
        raw = parse_llm_json(response.content)
        queries = raw.get("queries") or []
        # 清洗：strip + 非空 + 截到 max_queries
        cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        return cleaned[:max_queries]
    except Exception as e:
        print(f"[phase_a] _expand_search_queries 失败: {type(e).__name__}: {e}",
              file=sys.stderr)
        return []


def _select_top_papers_by_relevance(
    graph: "ConceptGraph",
    top_k: int,
) -> list[PaperInfo]:
    """
    按 (entity_hits, citation_count) 降序选 top-K paper。

    entity_hits[paper_id] = 这篇 paper 出现在多少个不同 entity 的 paper_ids 列表里。
    数值越大说明 entity 抽取认为它跟方向越相关（贡献了越多方法/数据集/指标实体）。

    与单纯按 citation 排的差异：
    - 基础工作论文（GPT-3/Llama 2/PyTorch）entity 抽取通常只 hit 1-2 个泛词
      （"transformer" / "language model"），entity_hits 低
    - 方向相关论文（LayerSkip / DEL / QSpec）会 hit 多个具体方法/指标
      （"speculative decoding" / "draft model" / "acceptance rate" / "early exit"），
      entity_hits 显著高
    - 同 entity_hits 的用 citation_count tiebreak（保留原有信号但作为次要）
    """
    from collections import Counter
    from darwinian.state import ConceptGraph
    entity_hits: Counter[str] = Counter()
    for e in graph.entities:
        for pid in e.paper_ids:
            entity_hits[pid] += 1

    return sorted(
        graph.papers,
        key=lambda p: (entity_hits.get(p.paper_id, 0), p.citation_count or 0),
        reverse=True,
    )[:top_k]


def _resolve_arxiv_ids(papers: list[PaperInfo]) -> dict[str, str]:
    """
    对每个 PaperInfo.paper_id 查 externalIds.ArXiv。
    返回 dict[s2_paper_id -> arxiv_id]，找不到的 value=""

    Note: build_concept_graph 走 S2 path 时 paper_id 是 S2 paperId（hex 串），
    需要再调一次 get_paper_details 才能拿 arxiv_id。所有调用走 S2 缓存，
    重复跑同一方向时几乎 0 成本。
    """
    result: dict[str, str] = {}
    for p in papers:
        if not p.paper_id:
            continue
        # 已经是 arxiv 格式（如 "2404.16710"）就直接用
        if _looks_like_arxiv_id(p.paper_id):
            result[p.paper_id] = p.paper_id
            continue
        try:
            detail = get_paper_details(p.paper_id, fields="externalIds")
            if detail:
                arxiv_id = (detail.get("externalIds") or {}).get("ArXiv", "") or ""
                result[p.paper_id] = arxiv_id
            else:
                result[p.paper_id] = ""
        except Exception as e:
            print(f"[phase_a] resolve_arxiv_ids 跳过 {p.paper_id}: {type(e).__name__}",
                  file=sys.stderr)
            result[p.paper_id] = ""
    return result


def _looks_like_arxiv_id(s: str) -> bool:
    """形如 '2404.16710' 或 '2404.16710v2' 的视为已经是 arxiv id"""
    import re
    return bool(re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", s))


def _format_evidence_id(p: PaperInfo, arxiv_map: dict[str, str]) -> str:
    """
    构造 PaperEvidence.paper_id 字段：
    - 有 arxiv_id 优先用 'arxiv:<id>'（让下游引用更可读）
    - 没有就回退 's2:<paperId>'
    """
    arxiv_id = arxiv_map.get(p.paper_id, "")
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    return f"s2:{p.paper_id}"


def _make_full_text_provider(
    arxiv_map: dict[str, str],
) -> Callable[[str], str]:
    """
    返回一个 callable(evidence_paper_id) -> rendered_full_text。
    供 batch_extract_evidence 内部调用——按 paper_id 拉全文。

    Note: evidence_paper_id 形如 'arxiv:2404.16710' 或 's2:abc...'。
    s2 形式的我们也尝试回查 arxiv_map 一次（万一 PaperInfo 有别名）。
    arxiv 形式直接拉。失败返空字符串 → batch_extract_evidence 自动降级 abstract-only。
    """
    # 把 arxiv_map（s2_paperid -> arxiv_id）反向也建一份索引
    rev = {f"arxiv:{v}": v for v in arxiv_map.values() if v}
    rev.update({f"s2:{k}": v for k, v in arxiv_map.items() if v})

    def provider(evidence_paper_id: str) -> str:
        arxiv_id = rev.get(evidence_paper_id, "")
        # evidence_paper_id 可能是 'arxiv:2404.16710'，尝试直接解析
        if not arxiv_id and evidence_paper_id.startswith("arxiv:"):
            arxiv_id = evidence_paper_id.split(":", 1)[1]
        if not arxiv_id:
            return ""
        try:
            src = fetch_arxiv_latex(arxiv_id)
            if src and src.has_full_text:
                return render_for_llm(src)
        except Exception as e:
            print(f"[phase_a] full_text_provider 跳过 {arxiv_id}: {type(e).__name__}",
                  file=sys.stderr)
        return ""

    return provider


_DROP_RELATIONS_DEFAULT = {"orthogonal"}
_DROP_RELATIONS_STRICT = {"orthogonal", "inspires", "reproduces"}


def _filter_relevant_evidence(
    evidence_list: list[PaperEvidence],
    *,
    strict: bool = False,
    min_keep: int = 3,
    out_stats: dict | None = None,
) -> list[PaperEvidence]:
    """
    Pri-5: 按 relation_to_direction 过滤无关 PaperEvidence。

    paper_evidence_extractor 给每篇标 5 类关系：
      extends / baseline / inspires / orthogonal / reproduces

    默认模式：丢 orthogonal（明确说"跟方向不相关"的）
    严格模式（DARWINIAN_EVIDENCE_STRICT=1）：仅保留 extends + baseline

    Round 8 fix: 加 min_keep 保底——v10 实测 8/8 全被标 orthogonal/inspires，
    导致只剩 1 篇 evidence → 5 candidates 退化成同一个 idea。保底逻辑：
    过滤后 < min_keep 时按 relation 优先级补回（extends > baseline > inspires
    > reproduces > orthogonal），保证至少 min_keep 篇喂给 elaborator。
    """
    if not evidence_list:
        if out_stats is not None:
            out_stats["truly_relevant"] = 0
            out_stats["backfilled"] = 0
            out_stats["dropped"] = 0
        return []
    drop_set = _DROP_RELATIONS_STRICT if strict else _DROP_RELATIONS_DEFAULT
    kept: list[PaperEvidence] = []
    dropped: list[PaperEvidence] = []
    for ev in evidence_list:
        rel = (ev.relation_to_direction or "").strip().lower()
        if rel in drop_set:
            dropped.append(ev)
        else:
            kept.append(ev)

    n_truly_relevant = len(kept)   # backfill 之前的"真相关"数

    # Round 8 fix: 保底 min_keep
    backfilled: list[str] = []
    if len(kept) < min_keep and dropped:
        # 按 relation 优先级排被丢的
        priority = ["extends", "baseline", "inspires", "reproduces", "orthogonal"]
        dropped_sorted = sorted(
            dropped,
            key=lambda ev: priority.index((ev.relation_to_direction or "").strip().lower())
            if (ev.relation_to_direction or "").strip().lower() in priority
            else 99,
        )
        need = min_keep - len(kept)
        for ev in dropped_sorted[:need]:
            kept.append(ev)
            backfilled.append(f"{ev.short_name or ev.paper_id} "
                              f"({ev.relation_to_direction or '?'})")

    mode_str = "strict" if strict else "default"
    n_dropped = len(dropped) - len(backfilled)
    if n_dropped > 0 or backfilled:
        msg_parts = [f"[phase_a] relevance gate ({mode_str}): "]
        if n_dropped > 0:
            dropped_kept = [ev for ev in dropped
                            if (ev.short_name or ev.paper_id) not in
                            [b.split(" (")[0] for b in backfilled]]
            preview = [f"{ev.short_name or ev.paper_id} "
                       f"({ev.relation_to_direction or '?'})"
                       for ev in dropped_kept[:5]]
            msg_parts.append(f"丢 {n_dropped}/{len(evidence_list)} 篇: "
                             f"{', '.join(preview)}")
            if n_dropped > 5:
                msg_parts.append(f" (+{n_dropped-5} more)")
        if backfilled:
            msg_parts.append(f"; 保底补回 {len(backfilled)} 篇 "
                             f"(min_keep={min_keep}): {', '.join(backfilled)}")
        print("".join(msg_parts), file=sys.stderr)
    else:
        print(f"[phase_a] relevance gate ({mode_str}): 全部 "
              f"{len(evidence_list)} 篇通过", file=sys.stderr)
    if out_stats is not None:
        out_stats["truly_relevant"] = n_truly_relevant
        out_stats["backfilled"] = len(backfilled)
        out_stats["dropped"] = n_dropped
    return kept


def _build_relevance_warning(
    truly_relevant: int,
    backfilled: int,
    *,
    min_truly_relevant: int = 5,
) -> str:
    """
    R10-Pri-2: 真相关论文不足时返回 warning 字符串，否则空。

    阈值 5 是经验值：
    - 加密流量 v2 LIVE: 3 真相关 + 3 兜底 → elaborator 硬拗，应警告
    - LLM inference v11: 8 真相关 + 0 兜底 → 不警告
    - 一般 NeurIPS-tier paper 引用 ≥10 篇相关 prior work，5 篇是 motivation 最低线
    """
    if truly_relevant >= min_truly_relevant:
        return ""
    return (
        f"Phase A 仅找到 {truly_relevant} 篇真相关论文 (extends/baseline/inspires/"
        f"reproduces；阈值 {min_truly_relevant})，已用 {backfilled} 篇 orthogonal "
        f"兜底以避免 evidence 全空。**elaborator 不要把 orthogonal 兜底论文当 "
        f"motivation 主弹药**——它们与本方向不相关，强行类比会产出 hand-waved "
        f"cross-domain 论证。建议换更聚焦的 sub-direction，让 Phase A 拉到 "
        f"≥{min_truly_relevant} 篇真相关论文再跑。"
    )


def _bucket_by_year(
    papers: list[PaperInfo],
    arxiv_map: dict[str, str],
) -> dict[str, list[str]]:
    """
    分时间桶，给 elaborator 在 Why Now section 引用时间感。

    桶定义：
      - foundational_pre_2024: < 2024 年（基础工作）
      - hot_2024_2026:         2024-2026 年（近期热门）
        Note: 2024 是 LLM acceleration 等多个赛道的爆发年（LayerSkip / EAGLE-2 /
        Medusa-2 等），不能像之前 _bucket_by_year v1 那样把 2024 漏掉
    """
    buckets: dict[str, list[str]] = {
        "foundational_pre_2024": [],
        "hot_2024_2026": [],
    }
    for p in papers:
        evid_id = _format_evidence_id(p, arxiv_map)
        year = p.year or 0
        if 0 < year < 2024:
            buckets["foundational_pre_2024"].append(evid_id)
        elif year >= 2024:
            buckets["hot_2024_2026"].append(evid_id)
    # 删空桶
    return {k: v for k, v in buckets.items() if v}

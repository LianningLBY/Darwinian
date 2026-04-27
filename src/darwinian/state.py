"""
ResearchState — 全局强类型状态定义

所有 LangGraph 节点通过此 State 进行数据流转。
采用 Pydantic v2 定义，消除 LLM 输出幻觉与分数作弊。
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# 枚举类型
# ---------------------------------------------------------------------------

class CriticVerdict(str, Enum):
    """理论审查官 (Agent 3) 的审查结论"""
    PASS = "PASS"
    MATH_ERROR = "MATH_ERROR"
    NOT_NOVEL = "NOT_NOVEL"


class ExecutionVerdict(str, Enum):
    """执行结果路由的判定结果"""
    CODE_ERROR = "code_error"       # 语法/运行时错误 → 内层循环修复
    INSUFFICIENT = "insufficient"   # 方法无提升 → 写入 failed_ledger，终止本轮
    SUCCESS = "success"             # 基准测试有提升 → 进入毒药测试


class FinalVerdict(str, Enum):
    """成果验收员 (Agent 7) 的最终裁决"""
    PUBLISH_READY = "publish_ready"     # 四项全绿 → END
    ROBUSTNESS_FAIL = "robustness_fail" # 鲁棒性不足 → 打回阶段一


# ---------------------------------------------------------------------------
# 子结构体
# ---------------------------------------------------------------------------

class BudgetState(BaseModel):
    """预算追踪：Token 消耗与时间窗口"""
    remaining_tokens: int = Field(default=200_000, description="剩余可用 Token 数")
    elapsed_seconds: float = Field(default=0.0, description="已耗时（秒）")
    max_tokens: int = Field(default=200_000)
    max_seconds: float = Field(default=3600.0)

    @property
    def is_exhausted(self) -> bool:
        return (
            self.remaining_tokens <= 0
            or self.elapsed_seconds >= self.max_seconds
        )


class FailedRecord(BaseModel):
    """认知账本中的单条失败记录"""
    feature_vector: list[float] = Field(description="假设的语义嵌入向量，用于余弦相似度去重")
    error_summary: str = Field(description="失败原因摘要")
    failure_type: str = Field(description="失败类型: MATH_ERROR / NOT_NOVEL / INSUFFICIENT / ROBUSTNESS_FAIL")
    iteration: int = Field(description="发生在第几次大循环")
    banned_keywords: list[str] = Field(default_factory=list, description="该方向的禁用关键词，用于 Agent 1 文献过滤")


class PaperInfo(BaseModel):
    """论文元数据 — ConceptGraph 中的节点"""
    paper_id: str = Field(description="Semantic Scholar paperId 或 arxiv id")
    title: str = Field(default="")
    abstract: str = Field(default="")
    year: int = Field(default=0)
    citation_count: int = Field(default=0)
    task_type: str = Field(default="", description="由实体抽取填写：classification/regression/NLP/CV/RL/...")
    source: str = Field(default="semantic_scholar", description="来源：semantic_scholar / arxiv / ...")


class Entity(BaseModel):
    """从论文中抽取的术语实体（方法/数据集/指标）

    task_type 不是 Entity——它是 PaperInfo 的元数据属性，仅用于跨分支
    多样性校验，不参与实体表、结构洞、硬约束引用。
    """
    canonical_name: str = Field(description="规范化名字：全小写、去标点、最短通用英文名")
    aliases: list[str] = Field(default_factory=list, description="合并前的原始写法集合")
    type: Literal["method", "dataset", "metric"] = Field(description="实体类型")
    paper_ids: list[str] = Field(default_factory=list, description="出现在哪些 paper 中")


class LimitationRef(BaseModel):
    """论文承认的某条缺陷 — 可被 AbstractionBranch 作为 solved_limitation_id 引用"""
    id: str = Field(description="稳定哈希 id：hashlib.md5((text+paper_id).encode()).hexdigest()[:8]")
    text: str = Field(description="缺陷的自然语言描述（一句话）")
    source_paper_id: str = Field(description="来源论文 paperId")


class EntityPair(BaseModel):
    """共现矩阵中的一条"结构洞"候选：两端各自成熟但从未共现"""
    entity_a: str = Field(description="Entity.canonical_name")
    entity_b: str = Field(description="Entity.canonical_name")
    score: int = Field(description="min(paper_count_a, paper_count_b)，偏向两端都成熟的组合")


class StructuralHoleHook(BaseModel):
    """
    Phase A → Phase B 的结构洞弹药。
    在 EntityPair 之上加一句话 hook + 关系类型，让 elaborator 不用从零想"为什么这个交叉值得做"。

    生成时机：bottleneck_miner 算完共现矩阵 → 给 top-K novel pair 各调一次小 LLM 写 hook_text。
    供 proposal_elaborator 在 motivation / why_now section 直接引用。
    """
    entity_a: str = Field(description="Entity.canonical_name")
    entity_b: str = Field(description="Entity.canonical_name")
    score: int = Field(default=0, description="EntityPair.score，方便排序")
    hook_text: str = Field(
        description="一句话解释这个交叉为什么值得做。"
                    "如 'A 边 N 篇都用 metric_X 优化，B 边 M 篇都用 metric_Y 测量，"
                    "但没人在同一篇里比较 X 和 Y 的 rank correlation'",
    )
    relation_type: Literal["divergence", "convergence", "transfer"] = Field(
        description="交叉的预期关系类型："
                    "divergence=两端可能发散需对照测；"
                    "convergence=可能等价值得理论分析；"
                    "transfer=一端方法迁移到另一端任务",
    )
    supporting_paper_ids_a: list[str] = Field(
        default_factory=list,
        description="A 边的代表论文 paperId（≤ 5），追溯/查重用",
    )
    supporting_paper_ids_b: list[str] = Field(
        default_factory=list,
        description="B 边的代表论文 paperId（≤ 5）",
    )


class ConceptGraph(BaseModel):
    """
    Phase 1 v2 核心产物：从 60+ 篇论文抽取出的术语 + 缺陷 + 结构洞。
    由 bottleneck_miner 填充，供 hypothesis_generator 作为硬约束源。
    """
    papers: list[PaperInfo] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    limitations: list[LimitationRef] = Field(default_factory=list)
    novel_pair_hints: list[EntityPair] = Field(
        default_factory=list,
        description="共现矩阵 top-10：从未共现但各自高频的术语对（潜在结构洞）",
    )
    is_sufficient: bool = Field(
        default=False,
        description="数据是否足够支撑硬约束。False 时 hypothesis_generator 降级走老 prompt。",
    )

    def entity_by_name(self, name: str) -> Entity | None:
        """按 canonical_name 查实体（未做归一化，调用方负责 normalize）"""
        for e in self.entities:
            if e.canonical_name == name:
                return e
        return None

    def limitation_by_id(self, lid: str) -> LimitationRef | None:
        for l in self.limitations:
            if l.id == lid:
                return l
        return None


class MethodologyPhase(BaseModel):
    """ResearchProposal 里的一个执行阶段"""
    phase_number: int = Field(description="阶段序号 (1, 2, 3, 4)")
    name: str = Field(description="阶段名（如 'Dual-metric profiling'）")
    description: str = Field(description="阶段详细描述（多句）")
    inputs: list[str] = Field(default_factory=list, description="本阶段输入")
    outputs: list[str] = Field(default_factory=list, description="本阶段产物")
    expected_compute_hours: float = Field(default=0.0, description="估计耗时（GPU 小时）")


class QuantitativeClaim(BaseModel):
    """从论文里抽取的单个精确定量结果"""
    metric_name: str = Field(description="指标名，如 'speedup' / 'PPL' / 'top-1 accuracy'")
    metric_value: str = Field(description="精确值，如 '2.16-2.62x' / '5.54' / '85.3%'")
    setting: str = Field(default="", description="模型 + benchmark 上下文，如 'Llama-3.1-8B on MATH'")


class PaperEvidence(BaseModel):
    """
    从单篇论文抽取的"五元组+"，用于 ResearchProposal 的研究现状/Why Now/key references。

    设计目标：让 elaborator 能写出 "DEL (COLM 2025, 2.16-2.62x)" 这种带具体数字 + 出处
    的 grounded 引用，不再泛泛说"已有方法效果不佳"。
    """
    paper_id: str = Field(description="arxiv:2404.16710 / s2:xxxxx 形式")
    title: str = Field(default="")
    short_name: str = Field(
        default="",
        description="论文/方法的口语化简称，如 'LayerSkip' / 'DEL' / 'QSpec'。"
                    "用于 ResearchProposal.key_references_formatted 的 'Name: Full title' 格式",
    )
    venue: str = Field(default="", description="发表会议/期刊，如 'ACL 2024' / 'arxiv preprint'")
    year: int = Field(default=0)
    category: str = Field(
        default="",
        description="本论文所属研究类别，如 'Layer-skipping self-speculative methods'。"
                    "供 elaborator 在研究现状里按类别分组",
    )
    method_names: list[str] = Field(
        default_factory=list,
        description="论文提出/使用的方法名（method 实体）",
    )
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list, description="评估指标名（不带数值）")
    quantitative_claims: list[QuantitativeClaim] = Field(
        default_factory=list,
        description="精确定量结果列表。这是 motivation/why_now 引用的弹药",
    )
    headline_result: str = Field(
        default="",
        description="一句话核心数据，用于 '研究现状' 里的简短引用，如 '2.16-2.62x speedup'",
    )
    limitations: list[str] = Field(default_factory=list, description="论文承认的局限")
    relation_to_direction: str = Field(
        default="",
        description="本论文与当前研究方向的关系。"
                    "枚举值: 'extends' / 'baseline' / 'inspires' / 'orthogonal' / 'reproduces'",
    )
    full_text_used: bool = Field(
        default=False,
        description="True=抽取自全文 LaTeX；False=仅 abstract（精度会低）",
    )


class ResearchConstraints(BaseModel):
    """
    用户对研究方案的约束条件。
    Phase A 收集（用户输入或推断），Phase B elaborator 必须满足这些约束。

    设计目标：把"4×RTX PRO 6000、7天内、不用 RL、benchmark 用现成、≤14B 模型"
    这种约束硬编码成 Pydantic 字段，不再让 LLM 从一段自由文本里"自己理解"。
    """
    # 资源
    gpu_count: int = Field(default=4, description="可用 GPU 数")
    gpu_model: str = Field(default="", description="GPU 型号，如 'RTX PRO 6000 96GB'")
    gpu_hours_budget: float = Field(default=0.0, description="总 GPU 小时预算")
    wall_clock_days: int = Field(default=7, description="项目总周期（天）")

    # 模型/数据
    max_model_params_b: float = Field(default=14.0, description="允许的最大模型参数（B）")
    use_existing_benchmarks_only: bool = Field(default=True, description="必须用现成 benchmark")
    require_human_annotation: bool = Field(default=False, description="是否允许人工标注")

    # 方法学排除
    forbidden_techniques: list[str] = Field(
        default_factory=list,
        description="禁用技术列表，如 ['GRPO', 'PPO', 'DPO', 'RLHF', 'RLVR', 'RLMT']",
    )
    require_no_api_for_main: bool = Field(
        default=True,
        description="主实验是否禁用闭源 API（baseline 比较可以用）",
    )

    # Venue 目标
    target_venues: list[str] = Field(
        default_factory=list,
        description="目标 venue 列表（按优先级），如 ['NeurIPS 2026', 'EMNLP 2026']",
    )

    # 自由扩展
    extra_notes: str = Field(default="", description="未结构化的额外说明，原文保留")


class ResourceEstimate(BaseModel):
    """三种执行模式的资源估算"""
    auto_research: dict = Field(
        default_factory=dict,
        description="纯 AI 自动模式: {'gpu_hours': X, 'usd_cost': Y, 'wall_clock_days': Z}",
    )
    human_in_loop: dict = Field(
        default_factory=dict,
        description="HITL 模式: AI 主导但研究者每天监督 ~1 小时",
    )
    manual: dict = Field(
        default_factory=dict,
        description="人工纯手做模式（用于对比，体现 AI 加速）",
    )


class ExpectedOutcomes(BaseModel):
    """
    QuantSkip 风格的"正反两种结果都可发表"叙事。
    把单一 expected_outcomes 字符串拆成 3 个独立字段，防止 elaborator 把"反向也 publishable"
    这个关键发表保险论证写丢。

    硬约束：positive_finding / negative_finding / why_both_publishable 三者必须都非空。
    """
    positive_finding: str = Field(
        description="正向发现叙事：'如果实验得到 X，则证明 Y'。必须含具体可观测信号",
    )
    negative_finding: str = Field(
        description="反向发现叙事：'如果实验得到 ~X，则证明 Z'。同样必须可发表",
    )
    why_both_publishable: str = Field(
        description="为什么两种结果对社区都有 actionable guidance",
    )


class ResearchProposal(BaseModel):
    """
    Phase 1 v3 产出：从 AbstractionBranch 骨架展开的完整研究 proposal。

    设计目标：对齐 QuantSkip 风格 markdown 模板（标题/状态/种子/描述/核心问题/
    Why Now/方法思路/研究现状/目标 venue/关键参考/资源预估）。

    设计借鉴 HKUDS AI-Researcher 论文（NeurIPS 2025, arxiv:2505.18705）的
    6-section schema 思想 + Darwinian 自有的 ConceptGraph grounding。
    """
    # ---- 链接回骨架 ----
    skeleton: AbstractionBranch = Field(description="生成本提案的骨架 branch")

    # ---- 元数据（用于序列化成 markdown 模板）----
    status: str = Field(default="draft", description="状态: draft/under_review/approved/rejected")
    level: str = Field(default="", description="级别标签，如 'top-tier' / 'workshop'")
    seed: str = Field(default="", description="生成本提案时的输入约束原文，便于追溯")
    created_at: str = Field(default="", description="ISO 时间戳，由 elaborator 填")

    # ---- 标题与电梯演讲 ----
    title: str = Field(description="标题，含子问题，如 'X: Do Y also imply Z?'")
    elevator_pitch: str = Field(description="200 字左右的方案描述")

    # ---- 6-section 内容（顺序借鉴 HKUDS）----
    challenges: str = Field(default="", description="该方向的核心挑战")
    existing_methods: str = Field(
        default="",
        description="现有方法分类组织。Markdown 格式，每类用 '**Category**: paper1, paper2...'，"
                    "末尾必须有 '**The gap**: ...' 段落。供研究现状 section 用",
    )
    motivation: str = Field(
        default="",
        description="为什么现在做。必须引用 ≥3 个具体定量数据 (quantitative_claims)",
    )
    proposed_method: str = Field(default="", description="提出的方法概览")
    technical_details: str = Field(default="", description="技术细节（公式、关键算法）")
    expected_outcomes: str = Field(
        default="",
        description="预期结果（渲染后的自由文本，markdown 模板用）。"
                    "仍保留兼容老 elaborator；新代码请填 expected_outcomes_structured",
    )
    expected_outcomes_structured: ExpectedOutcomes | None = Field(
        default=None,
        description="结构化的正反 publishable 叙事。verifier 校验对象；"
                    "为 None 时退回检查 expected_outcomes 文本",
    )

    # ---- Phased methodology（Darwinian 自创）----
    methodology_phases: list[MethodologyPhase] = Field(
        default_factory=list,
        description="3-4 个执行阶段。phases 的 expected_compute_hours 总和必须 ≤ gpu_hours_budget",
    )
    total_estimated_hours: float = Field(default=0.0)
    fits_resource_budget: bool = Field(default=False, description="是否满足资源约束")

    # ---- Target venue ----
    target_venue: str = Field(default="", description="主要目标 venue（如 'NeurIPS 2026'）")
    target_deadline: str = Field(default="", description="ISO 日期，如 '2026-05-13'")
    fallback_venue: str = Field(default="", description="备选 venue，如时间不够")

    # ---- Key references ----
    key_references: list[str] = Field(
        default_factory=list,
        description="paperId 列表（追溯/查重用）。每个必须存在于 ConceptGraph.papers",
    )
    key_references_formatted: list[str] = Field(
        default_factory=list,
        description="QuantSkip 风格的 markdown 列表项，如 'LayerSkip: Enabling Early Exit "
                    "Inference and Self-Speculative Decoding (ACL 2024)'。直接渲染到模板",
    )

    # ---- 资源预估（三种模式）----
    resource_estimate: ResourceEstimate = Field(
        default_factory=ResourceEstimate,
        description="auto / human_in_loop / manual 三种执行模式的资源估算",
    )


class ResearchMaterialPack(BaseModel):
    """
    Phase A 调研 Agent 的最终产物 / Phase B 强 LLM elaborator 的输入容器。

    设计动机：之前 elaborator 要从 ConceptGraph + list[PaperEvidence] + ResearchConstraints
    + failed_ledger 多个对象拼装上下文，容易漏字段。MaterialPack 把"喂给 Phase B 的所有素材"
    打包成单一对象，elaborator 收到这一份就够。

    生成路径：
      bottleneck_miner → ConceptGraph
      arxiv_latex_fetcher + paper_evidence_extractor → list[PaperEvidence]
      共现矩阵 + LLM hook 写作 → list[StructuralHoleHook]
      用户输入 → ResearchConstraints
      → 组装成 ResearchMaterialPack → 喂给 proposal_elaborator
    """
    direction: str = Field(description="研究方向原文（用户输入）")
    constraints: ResearchConstraints = Field(
        default_factory=ResearchConstraints,
        description="资源/合规/方法学约束。elaborator 必须满足",
    )
    paper_evidence: list[PaperEvidence] = Field(
        default_factory=list,
        description="20-30 篇论文的深抽取五元组，按 category 分组的弹药",
    )
    concept_graph: ConceptGraph | None = Field(
        default=None,
        description="跨论文的实体表 + limitations + 共现矩阵",
    )
    structural_hole_hooks: list[StructuralHoleHook] = Field(
        default_factory=list,
        description="带 hook_text 的结构洞 top-K，elaborator 在 motivation 里直接引用",
    )
    timeline_signals: dict[str, list[str]] = Field(
        default_factory=dict,
        description="按时间分桶的 paperId 列表，给 'Why Now' section 提供时间感。"
                    "key 形如 'foundational_pre_2024' / 'hot_2025_2026' / 'last_30_days'",
    )
    prior_failures: list[FailedRecord] = Field(
        default_factory=list,
        description="failed_ledger 的快照，让 elaborator 主动避开已知失败方向",
    )

    @property
    def evidence_by_category(self) -> dict[str, list[PaperEvidence]]:
        """按 PaperEvidence.category 分组，给 elaborator 的 existing_methods section 直接用"""
        groups: dict[str, list[PaperEvidence]] = {}
        for ev in self.paper_evidence:
            cat = ev.category or "uncategorized"
            groups.setdefault(cat, []).append(ev)
        return groups


# ---------------------------------------------------------------------------
# Phase C: 辩论裁决（Advocate / Challenger / Judge）
# ---------------------------------------------------------------------------

class DebateRound(BaseModel):
    """单轮辩论：Advocate 论证 → Challenger 反驳 → Judge 裁决 + 给概率"""
    round_number: int = Field(description="第几轮（从 1 开始）")
    advocate_argument: str = Field(description="正方论证：为什么这个 idea 能中稿")
    challenger_argument: str = Field(description="反方反驳：弱点 + 已有覆盖 + 风险")
    judge_assessment: str = Field(description="裁决说明：哪些 challenger 论点站得住、哪些不")
    estimated_acceptance_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="本轮裁决后的中稿概率估计 (0-1)",
    )
    revisions_proposed: list[str] = Field(
        default_factory=list,
        description="Judge 建议的 seed 修改点列表，传给下一轮 Advocate 用",
    )


class DebateResult(BaseModel):
    """
    Phase C 完整辩论历史 + 收敛裁决。
    收敛条件：final_acceptance_rate ≥ acceptance_threshold AND
              最近 2 轮 estimated_acceptance_rate 的 |delta| < convergence_delta
    """
    rounds: list[DebateRound] = Field(default_factory=list)
    final_acceptance_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    acceptance_threshold: float = Field(
        default=0.30,
        description="立项的中稿率门槛（默认 30%）",
    )
    convergence_delta: float = Field(
        default=0.05,
        description="判收敛的相邻轮 delta 阈值（默认 5%）",
    )
    converged: bool = Field(
        default=False,
        description="True=已收敛且达到门槛，可立项；False=继续辩论或终止",
    )
    revised_proposal: ResearchProposal | None = Field(
        default=None,
        description="辩论修订后的 seed（最后一轮 Advocate 输出），收敛时填",
    )

    @property
    def is_above_threshold(self) -> bool:
        return self.final_acceptance_rate >= self.acceptance_threshold

    @property
    def delta_last_two(self) -> float:
        """最近两轮 estimated_acceptance_rate 的绝对差；不足两轮返回 inf"""
        if len(self.rounds) < 2:
            return float("inf")
        return abs(self.rounds[-1].estimated_acceptance_rate - self.rounds[-2].estimated_acceptance_rate)


class AbstractionBranch(BaseModel):
    """单个抽象方案分支"""
    name: str = Field(description="方案名称")
    description: str = Field(description="方案描述")
    algorithm_logic: str = Field(description="算法逻辑说明")
    math_formulation: str = Field(description="数学公式映射")
    # Phase 1 v2 新增：硬约束字段
    cited_entity_names: list[str] = Field(
        default_factory=list,
        description="引用的术语（必须在当轮 ConceptGraph.entities 里）",
    )
    solved_limitation_id: str = Field(
        default="",
        description="声明要解决的缺陷 id（必须在当轮 ConceptGraph.limitations 里）",
    )
    existing_combination: bool = Field(
        default=False,
        description="step 7.5 填写：cited_entity_names 组合是否已有人做过",
    )
    existing_combination_refs: list[str] = Field(
        default_factory=list,
        description="若 existing_combination=True，列出命中的 paperId（最多 3 个）",
    )
    # 已弃用字段（保留兼容，后续 commit 清理 prompt 引用）
    source_domain: str = Field(default="", description="[已弃用] v1 用自由文字标领域；v2 用 cited_entities 派生")


class Hypothesis(BaseModel):
    """当前研究假设 — 强类型，Agent 2 必须完整填充所有字段"""
    core_problem: str = Field(description="核心矛盾/待解问题")
    abstraction_tree: list[AbstractionBranch] = Field(
        default_factory=list,
        description="多维解决思路（至少 2 个分支，由 Agent 2 填充）",
    )
    selected_branch: AbstractionBranch | None = Field(
        default=None,
        description="Agent 3 审查通过后选定的执行分支",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Agent 2 对该假设的主观置信度",
    )
    literature_support: list[str] = Field(
        default_factory=list,
        description="支撑该假设的文献引用列表",
    )


class ExperimentCode(BaseModel):
    """Agent 4 生成的实验代码包"""
    baseline_code: str = Field(description="基准模型代码")
    proposed_code: str = Field(description="提出方法代码")
    dataset_loader_code: str = Field(description="数据加载代码")
    ablation_code: str = Field(default="", description="消融实验代码（逐一去掉各组件验证贡献）")
    robustness_code: str = Field(default="", description="Agent 6 生成的鲁棒性测试代码")
    requirements: list[str] = Field(default_factory=list, description="代码依赖的 pip 包")
    retry_count: int = Field(default=0, description="已重试次数，上限 5 次")


class ExperimentResult(BaseModel):
    """代码执行结果"""
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    proposed_metrics: dict[str, float] = Field(default_factory=dict)
    execution_verdict: ExecutionVerdict | None = Field(default=None)
    diagnosis: str = Field(default="", description="Agent 5 的诊断说明")


class RobustnessResult(BaseModel):
    """鲁棒性测试结果"""
    perturbation_strategy: str = Field(description="使用的扰动策略名称")
    perturbed_metrics: dict[str, float] = Field(default_factory=dict)
    degradation_rate: float = Field(default=0.0, description="性能下降比例")


# 向后兼容别名
PoisonTestResult = RobustnessResult


class PublishMatrix(BaseModel):
    """发表指标矩阵 — 四项全绿才能 END"""
    novelty_passed: bool = Field(default=False, description="新颖性达标")
    baseline_improved: bool = Field(default=False, description="超越基准模型")
    robustness_passed: bool = Field(default=False, description="鲁棒性测试通过")
    explainability_generated: bool = Field(default=False, description="可解释性报告已生成")

    @property
    def all_green(self) -> bool:
        return all([
            self.novelty_passed,
            self.baseline_improved,
            self.robustness_passed,
            self.explainability_generated,
        ])


# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------

class DatasetInfo(BaseModel):
    """数据集选择结果"""
    source: str = Field(description="来源: 'huggingface' | 'user_upload' | 'builtin'")
    dataset_id: str = Field(description="HuggingFace dataset ID 或用户上传的文件名")
    description: str = Field(default="", description="数据集描述")
    task_type: str = Field(default="", description="任务类型: classification / regression / etc.")
    load_instruction: str = Field(default="", description="加载代码示例")


class ResearchState(BaseModel):
    """
    Darwinian 系统的全局状态。
    LangGraph 中所有节点共享此对象进行数据流转。
    """

    # 研究方向输入
    research_direction: str = Field(default="", description="用户输入的研究方向描述")
    dataset_schema: dict[str, Any] = Field(default_factory=dict, description="实验数据集 Schema（可选，用户填写）")

    # 数据集选择结果（由 dataset_finder_node 填充）
    selected_dataset: DatasetInfo | None = Field(default=None, description="自动搜索或用户上传的数据集信息")
    user_data_path: str = Field(default="", description="用户上传数据集在宿主机上的路径（Docker 只读挂载）")

    # 预算追踪
    budget_state: BudgetState = Field(default_factory=BudgetState)

    # 认知账本 — 跨循环持久化，每次大循环开始前强制 Agent 1 读取
    failed_ledger: list[FailedRecord] = Field(
        default_factory=list,
        description="历史失败记录，含语义向量，用于余弦相似度去重",
    )

    # Phase 1 v2：从论文网络抽出的术语 + 缺陷 + 结构洞（由 bottleneck_miner 填充）
    concept_graph: ConceptGraph | None = Field(
        default=None,
        description="Phase 1 v2 核心产物，供 hypothesis_generator 作为硬约束源",
    )

    # Phase 1 v3：Phase A 调研产出，供 elaborator 当输入素材
    # 由 phase_a_orchestrator.build_research_material_pack 填充
    material_pack: ResearchMaterialPack | None = Field(
        default=None,
        description="Phase A 完整素材包（PaperEvidence + StructuralHoleHook + "
                    "constraints + timeline）。proposal_elaborator_node_v3 从此读取。",
    )

    # 当前假设
    current_hypothesis: Hypothesis | None = Field(default=None)

    # Phase 1 v3：每个 abstraction_tree branch 对应展开的 ResearchProposal
    # N:N 对齐——research_proposals[i] 对应 current_hypothesis.abstraction_tree[i]
    # 由 proposal_elaborator_node_v3 填充
    research_proposals: list[ResearchProposal] = Field(
        default_factory=list,
        description="Phase 1 v3 产出：每个骨架展开的完整 proposal",
    )

    # 理论审查结论
    critic_verdict: CriticVerdict | None = Field(default=None)
    critic_feedback: str = Field(default="", description="审查官的详细反馈")

    # 实验代码与结果
    experiment_code: ExperimentCode | None = Field(default=None)
    experiment_result: ExperimentResult | None = Field(default=None)

    # 鲁棒性测试
    robustness_result: RobustnessResult | None = Field(default=None)

    # 消融实验结果 {变体名称: 指标字典}，由消融执行节点填充
    ablation_results: dict[str, dict] = Field(default_factory=dict, description="各消融变体的指标结果")

    # 同伴评审分数（模拟三位审稿人）
    peer_review_scores: dict[str, Any] = Field(default_factory=dict, description="模拟审稿人评分")

    # 发表指标矩阵
    publish_matrix: PublishMatrix = Field(default_factory=PublishMatrix)

    # 最终裁决
    final_verdict: FinalVerdict | None = Field(default=None)
    final_report: str = Field(default="", description="最终研究报告 Markdown")

    # 循环计数 — 防止无限大循环
    outer_loop_count: int = Field(default=0, description="外层大循环次数")
    max_outer_loops: int = Field(default=5, description="外层大循环上限")
    # Phase 1 内层假设重试计数（每轮 Phase 1 开始时重置）
    hypothesis_retry_count: int = Field(default=0, description="当前 Phase 1 内层重试次数（NOT_NOVEL 触发）")
    # Phase 1 MATH_ERROR 重试计数（每轮 Phase 1 开始时重置）
    miner_retry_count: int = Field(default=0, description="当前 Phase 1 MATH_ERROR 重试次数（Agent 1 重跑计数）")

    # 跨节点传递的错误关键词（由 theoretical_critic / diagnostician 写入，由 ledger 节点读取）
    last_error_keywords: list[str] = Field(default_factory=list, description="最近一次失败提取的禁用关键词")

    # LangGraph messages（调试追踪，限制最大条数防止无限累积）
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    def trimmed_messages(self, max_count: int = 20) -> list:
        """返回最近 max_count 条消息，避免无限累积"""
        return self.messages[-max_count:] if len(self.messages) > max_count else self.messages

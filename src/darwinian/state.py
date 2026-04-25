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


class ResearchProposal(BaseModel):
    """
    Phase 1 v3 产出：从 AbstractionBranch 骨架展开的完整研究 proposal。

    设计借鉴 HKUDS AI-Researcher 论文（NeurIPS 2025, arxiv:2505.18705）的
    6-section schema 思想 + Darwinian 自有的 ConceptGraph grounding。
    """
    # ---- 链接回骨架 ----
    skeleton: AbstractionBranch = Field(description="生成本提案的骨架 branch")

    # ---- 标题与电梯演讲 ----
    title: str = Field(description="标题，含子问题，如 'X: Do Y also imply Z?'")
    elevator_pitch: str = Field(description="200 字左右的方案描述")

    # ---- 6-section 内容（顺序借鉴 HKUDS）----
    challenges: str = Field(description="该方向的核心挑战")
    existing_methods: str = Field(description="现有方法 + 局限性，按类别组织")
    motivation: str = Field(
        description="为什么现在做。必须引用 ≥3 个 ConceptGraph 里的 quantitative_claims",
    )
    proposed_method: str = Field(description="提出的方法概览")
    technical_details: str = Field(description="技术细节（公式、关键算法）")
    expected_outcomes: str = Field(
        description="预期结果。必须包含'正反两种结果都可发表'的 framing",
    )

    # ---- Phased methodology（Darwinian 自创）----
    methodology_phases: list[MethodologyPhase] = Field(
        default_factory=list,
        description="3-4 个执行阶段。phases 的 expected_compute_hours 总和必须 ≤ seed.gpu_hours_budget",
    )
    total_estimated_hours: float = Field(default=0.0)
    fits_resource_budget: bool = Field(default=False, description="是否满足 seed 资源约束")

    # ---- Target venue ----
    target_venue: str = Field(default="", description="主要目标 venue（如 'NeurIPS 2026'）")
    target_deadline: str = Field(default="", description="ISO 日期，如 '2026-05-13'")
    fallback_venue: str = Field(default="", description="备选 venue，如时间不够")

    # ---- Key references ----
    key_references: list[str] = Field(
        default_factory=list,
        description="关键引用 paperId 列表。每个必须存在于 ConceptGraph.papers",
    )


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

    # 当前假设
    current_hypothesis: Hypothesis | None = Field(default=None)

    # Phase 1 v3：每个 abstraction_tree branch 对应展开的 ResearchProposal
    # N:N 对齐——research_proposals[i] 对应 current_hypothesis.abstraction_tree[i]
    # 由 proposal_elaborator_node 填充
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

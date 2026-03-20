"""
ResearchState — 全局强类型状态定义

所有 LangGraph 节点通过此 State 进行数据流转。
采用 Pydantic v2 定义，消除 LLM 输出幻觉与分数作弊。
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any
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


class AbstractionBranch(BaseModel):
    """单个抽象方案分支"""
    name: str = Field(description="方案名称")
    description: str = Field(description="方案描述")
    algorithm_logic: str = Field(description="算法逻辑说明")
    math_formulation: str = Field(description="数学公式映射")
    source_domain: str = Field(description="灵感来源领域（跨域迁移）")


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

    # 当前假设
    current_hypothesis: Hypothesis | None = Field(default=None)

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

    # 跨节点传递的错误关键词（由 theoretical_critic / diagnostician 写入，由 ledger 节点读取）
    last_error_keywords: list[str] = Field(default_factory=list, description="最近一次失败提取的禁用关键词")

    # LangGraph messages（调试追踪，限制最大条数防止无限累积）
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    def trimmed_messages(self, max_count: int = 20) -> list:
        """返回最近 max_count 条消息，避免无限累积"""
        return self.messages[-max_count:] if len(self.messages) > max_count else self.messages

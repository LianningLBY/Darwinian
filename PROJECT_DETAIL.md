# Darwinian 项目详解

> 面向大模型应用开发岗位的技术文档，涵盖系统架构、技术栈、核心设计决策与关键工程细节。

---

## 一、项目概述

**Darwinian** 是一个基于多智能体协作的端到端自动化科研系统。用户输入一个研究方向，系统自主完成文献挖掘、跨域假设生成、理论审查、实验代码生成与执行、鲁棒性验证，最终产出可发表质量的研究报告。

**核心价值**：将科研闭环中最耗时的"提出假设 → 验证假设 → 迭代"环节自动化，大幅压缩科研试错周期。

---

## 二、技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **编排框架** | LangGraph (StateGraph) | 多 Agent 有状态工作流编排、条件路由、子图嵌套 |
| **LLM 集成** | LangChain Core | 统一 LLM 接口、消息格式、回调机制 |
| **数据验证** | Pydantic v2 | 全局状态强类型定义，消除 LLM 幻觉输出 |
| **LLM 接入** | OpenAI-compatible API | 兼容 MiniMax、DeepSeek 等推理模型 |
| **代码执行** | Docker SDK + subprocess | 沙箱隔离执行 LLM 生成的实验代码 |
| **UI 层** | Streamlit | 研究进度可视化、用户交互界面 |
| **ML 框架** | PyTorch / scikit-learn | 实验代码生成目标框架 |
| **文献检索** | Semantic Scholar API | 自动检索相关论文，提取 Limitations |
| **语义相似度** | 余弦相似度 + 文本嵌入 | 认知账本去重，规避历史失败路径 |

---

## 三、系统架构

### 3.1 整体流程

```
用户输入研究方向
       ↓
   [主图 Main Graph]
       ↓
  ┌─ Phase 1: 假设生成子图 ─────────────────────────────┐
  │  Agent 1: 瓶颈挖掘机 (bottleneck_miner)              │
  │       ↓ 文献检索 + 核心矛盾提炼                       │
  │  Agent 2: 方案合成器 (hypothesis_generator)           │
  │       ↓ 跨域创意方案生成（abstraction_tree）           │
  │  Agent 3: 理论审查官 (theoretical_critic)             │
  │       ↓ 数学可行性 + 新颖性审查                        │
  │  ┌── PASS → Phase 2                                  │
  │  ├── NOT_NOVEL → Agent 2 重试（最多 5 次）            │
  │  └── MATH_ERROR → Agent 1 重试（最多 3 次）           │
  └────────────────────────────────────────────────────┘
       ↓ selected_branch 非空
  ┌─ Phase 2: 实验验证子图 ─────────────────────────────┐
  │  Agent 4: 代码架构师 (code_architect)                │
  │       ↓ 分 4 次 LLM 调用生成：                        │
  │       │  dataset_loader / baseline / proposed / ablation │
  │  [沙箱执行] Docker → subprocess 降级                  │
  │  Agent 5: 诊断分析师 (diagnostician)                  │
  │       ↓ 解析 stdout/stderr，判断 code_error/success   │
  │  ┌── CODE_ERROR → Agent 4 修复重试（最多 5 次）        │
  │  ├── INSUFFICIENT → 写入账本 → Phase 1 重启           │
  │  └── SUCCESS → 消融实验 → 鲁棒性测试                  │
  │  Agent 6: 鲁棒性测试生成器 (robustness_generator)     │
  │       ↓ 选择扰动策略 + 生成测试代码                    │
  │  Agent 7: 成果验收员 (publish_evaluator)              │
  │       ↓ 三审稿人模拟 + 终局裁决                        │
  │  ┌── PUBLISH_READY → 保存报告 → END                  │
  │  └── ROBUSTNESS_FAIL → 写入账本 → Phase 1 重启        │
  └────────────────────────────────────────────────────┘
```

### 3.2 大循环机制

系统支持多轮大循环（默认最多 5 轮）：

- **Phase 2 失败**（方法不足 / 鲁棒性不达标）→ 写入**认知账本** → 重启 Phase 1
- Agent 1 读取账本，规避已知失败路径，生成差异化新假设
- 外层循环计数器 `outer_loop_count` 防止无限运行

### 3.3 子图嵌套结构（LangGraph 特性）

```python
# 主图将两个子图作为节点注册
graph.add_node("phase1", hypothesis_graph)   # 编译后的子图
graph.add_node("phase2", experiment_graph)   # 编译后的子图

# 大循环回路：Phase 2 失败 → 重启 Phase 1
graph.add_conditional_edges("phase2", phase2_result_router, {
    "restart_phase1": "phase1",
    "end_publish_ready": "save_results",
    "end_failed": "log_termination",
})
```

---

## 四、核心设计亮点

### 4.1 认知账本（Cognitive Ledger）

**设计动机**：防止系统在多轮循环中重复尝试已证明无效的方向。

**实现机制**：
- 每次失败（MATH_ERROR / NOT_NOVEL / INSUFFICIENT / ROBUSTNESS_FAIL）都写入 `failed_ledger`
- 每条记录包含：语义嵌入向量、失败原因摘要、失败类型、禁用关键词
- Agent 2 生成新假设后，与账本中所有记录计算余弦相似度，超过阈值 0.85 则拒绝并触发 `DuplicateHypothesisError`
- Agent 1 文献检索时过滤 `banned_keywords`，从源头规避重复方向

```python
class FailedRecord(BaseModel):
    feature_vector: list[float]    # 语义嵌入，用于余弦去重
    error_summary: str             # 失败原因
    failure_type: str              # MATH_ERROR / NOT_NOVEL / INSUFFICIENT / ROBUSTNESS_FAIL
    iteration: int                 # 第几轮大循环
    banned_keywords: list[str]     # 后续禁用的检索关键词
```

### 4.2 分段式代码生成（避免 Token 截断）

**问题**：一次性让 LLM 生成完整实验代码（loader + baseline + proposed + ablation）容易超出 token 上限导致截断。

**方案**：将代码生成拆分为 4 次独立 LLM 调用，每段不超过 150 行：

1. `dataset_loader_code`：数据加载脚本，暴露 `load_data()` 接口
2. `baseline_code`：基准方法，3 seed 统计
3. `proposed_code`：提出方法（**关键**：将 loader 前 600 字符注入 prompt，让 LLM 看到真实 `n_features`，彻底消除维度猜测）
4. `ablation_code`：消融实验

```python
proposed_code = _call_llm(llm, system=_BASE_SYSTEM, user=f"""
以下是本次实验的 dataset_loader.py（必须与此保持维度一致）：
{loader_code[:600]}
任务：只编写「Proposed Method 实验代码」...输入维度必须与 load_data() 返回的 X_train.shape[1] 一致
""")
```

**重试优化**：代码重试时（`retry_count > 0`）直接复用已有的 `loader_code` 和 `baseline_code`，防止维度在重试中再次漂移。

### 4.3 多级容错体系

系统在三个层级分别处理故障：

**网络层（`utils/llm_retry.py`）**：
```python
# 全局 LLM 调用封装，捕获流式连接断开并指数退避重试
def invoke_with_retry(llm, messages, max_retries=3, base_wait=5.0):
    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            if is_retryable(exc) and attempt < max_retries:
                time.sleep(base_wait * (attempt + 1))  # 线性退避：5s/10s/15s
                continue
            raise
```

可重试错误关键词：`RemoteProtocolError` / `incomplete chunked` / `peer closed` / `ConnectionError` / `ReadTimeout`

**解析层（`utils/json_parser.py`）**：

LLM 输出 JSON 的 5 级修复策略（按顺序尝试）：
1. 直接解析
2. 剥离 `<think>...</think>` 推理块（MiniMax / DeepSeek-R1）
3. 处理**未关闭的** `<think>` 块（截断场景）：从后往前扫描，找第一个含 `"key":` 模式的 `{` 行
4. 修复 LaTeX 公式中的非法反斜杠转义（`\mathcal` → `\\mathcal`）
5. 修复字符串内的字面量控制字符（换行 → `\n`）
6. 截断修复：逐字符追踪嵌套栈，补全缺失的引号和括号

**流程层（`graphs/hypothesis_graph.py`）**：

独立计数器防死循环：
```python
# NOT_NOVEL 内层循环：hypothesis_retry_count，上限 5 次
# MATH_ERROR 重跑循环：miner_retry_count，上限 3 次
# 两者在每轮 Phase 1 开始时（preprocess_node）重置为 0
```

### 4.4 强类型状态驱动（Pydantic v2 + LangGraph）

全局状态 `ResearchState` 用 Pydantic v2 定义，所有字段有明确类型和默认值：

```python
class ResearchState(BaseModel):
    current_hypothesis: Hypothesis | None = Field(default=None)
    critic_verdict: CriticVerdict | None = Field(default=None)
    experiment_code: ExperimentCode | None = Field(default=None)
    outer_loop_count: int = Field(default=0)
    hypothesis_retry_count: int = Field(default=0)
    miner_retry_count: int = Field(default=0)
    failed_ledger: list[FailedRecord] = Field(default_factory=list)
    # ...
```

**优势**：
- LangGraph 节点返回 `dict`，框架自动合并到状态，Pydantic 校验类型
- 避免 LLM 幻觉（如将字符串写入数字字段）在运行时静默传播
- 所有枚举值强类型化：`CriticVerdict.PASS / MATH_ERROR / NOT_NOVEL`

### 4.5 代码沙箱执行（Docker → subprocess 降级）

```python
def code_execute(experiment_code, mode, data_dir):
    try:
        client = docker.from_env()
        client.ping()  # 确认 daemon 在线
        return _execute_docker(client, ...)  # Docker 优先
    except DockerException:
        return _execute_subprocess(...)      # 自动降级
```

Docker 模式资源限制：0.5 CPU / 512MB RAM / 300s 超时 / 只读挂载数据目录

---

## 五、关键数据流

```
research_direction (str)
    ↓ Agent 1
current_hypothesis.core_problem (str)
    ↓ Agent 2
current_hypothesis.abstraction_tree (list[AbstractionBranch])
    ↓ Agent 3 (审查通过)
current_hypothesis.selected_branch (AbstractionBranch)
    ↓ Agent 4
experiment_code (ExperimentCode)
    ├── dataset_loader_code
    ├── baseline_code
    ├── proposed_code
    ├── ablation_code
    └── requirements
    ↓ 沙箱执行
experiment_result (ExperimentResult)
    ├── stdout / stderr
    ├── baseline_metrics {"accuracy_mean": 0.85, "accuracy_std": 0.02}
    └── proposed_metrics {"accuracy_mean": 0.91, "accuracy_std": 0.01}
    ↓ Agent 5 (诊断)
execution_verdict (SUCCESS / CODE_ERROR / INSUFFICIENT)
    ↓ Agent 6 + 7
final_verdict (PUBLISH_READY / ROBUSTNESS_FAIL)
    ↓
final_report (Markdown)
```

---

## 六、目录结构

```
Darwinian/
├── app.py                          # Streamlit 入口
├── src/darwinian/
│   ├── state.py                    # 全局状态定义（Pydantic v2）
│   ├── agents/
│   │   ├── bottleneck_miner.py     # Agent 1: 文献挖掘
│   │   ├── hypothesis_generator.py # Agent 2: 方案合成
│   │   ├── theoretical_critic.py   # Agent 3: 理论审查
│   │   ├── code_architect.py       # Agent 4: 代码生成
│   │   ├── diagnostician.py        # Agent 5: 执行诊断
│   │   ├── poison_generator.py     # Agent 6: 鲁棒性测试
│   │   └── publish_evaluator.py    # Agent 7: 成果验收
│   ├── graphs/
│   │   ├── hypothesis_graph.py     # Phase 1 子图
│   │   ├── experiment_graph.py     # Phase 2 子图
│   │   └── main_graph.py           # 主图（大循环编排）
│   ├── tools/
│   │   ├── code_executor.py        # 沙箱执行（Docker/subprocess）
│   │   ├── dataset_finder.py       # 数据集自动搜索
│   │   ├── semantic_scholar.py     # 文献检索 API
│   │   └── perturbation_strategies.py # 鲁棒性扰动策略库
│   └── utils/
│       ├── json_parser.py          # LLM 输出 JSON 容错解析
│       ├── llm_retry.py            # 网络断连自动重试
│       └── similarity.py           # 余弦相似度计算
└── results/                        # 生成的研究报告输出目录
```

---

## 七、量化成果

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| Phase 1 完成率 | 0%（必在 PASS 后立即终止） | 稳定通过，进入 Phase 2 |
| MATH_ERROR 循环 | 无上限，永久死循环 | ≤ 3 次后退出 |
| proposed.py 维度错误 | 100%（每次必崩） | 0%（loader 维度注入后消除） |
| LLM 截断导致崩溃 | 必崩（JSONDecodeError） | 自动修复，5 级解析兜底 |
| 网络断连崩溃 | 必崩（RemoteProtocolError） | 自动重试，最多 3 次，线性退避 |
| 端到端流程跑通 | 从未成功 | 稳定运行至 Phase 2 实验执行 |

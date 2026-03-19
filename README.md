# Darwinian

**状态驱动型多智能体自动化科研系统**

基于 LangGraph 框架，通过 7 个职责隔离的 LLM 智能体节点，实现从文献挖掘到对抗性实验验证的全流程自动化科研辅助。

## 核心设计原则

> **将 LLM 从"决策者"降级为"节点处理器"，将控制权交还给强类型 State 和明确的条件边。**

- `ResearchState`：Pydantic v2 强类型全局状态，消除 LLM 幻觉与分数作弊
- `failed_ledger`：认知账本，持久化失败历史，通过余弦相似度防止"原地打转"
- `publish_matrix`：四项布尔指标，强制客观评判，防止虚高评分

## 系统架构

```
START
  │
  ▼
Phase 1: 研究方案生成子图
  │  Agent 1: 瓶颈挖掘机 (bottleneck_miner_node)
  │  Agent 2: 方案合成器 (hypothesis_generator_node)
  │  Agent 3: 理论审查官 (theoretical_critic_node)
  │  条件路由: PASS → Phase 2 | MATH_ERROR → 写 ledger → Agent 1 | NOT_NOVEL → Agent 2
  │
  ▼
Phase 2: 自动实验与对抗验证子图
  │  Agent 4: 代码架构师 (code_architect_node)
  │  Tool:    代码执行沙箱 (Docker 隔离)
  │  Agent 5: 诊断分析师 (diagnostician_node)
  │  条件路由: code_error → Agent 4 (max 5次) | insufficient → 写 ledger → END
  │  Agent 6: 毒药数据生成器 (poison_generator_node) [固定策略库]
  │  Agent 7: 成果验收员 (publish_evaluator_node)
  │  条件路由: publish_ready → 保存报告 | robustness_fail → 写 ledger → Phase 1
  │
  ▼
END (生成 results/report_*.md)
```

## 快速开始

### 安装依赖

```bash
pip install -e ".[dev]"
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_API_KEY
```

### 运行测试

```bash
pytest tests/ -v
```

### 启动前端界面（推荐）

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`，在侧边栏填入研究方向和 API Key 后点击"开始研究"。

### 命令行运行

```bash
python examples/run_research.py
```

## 项目结构

```
src/darwinian/
├── state.py                    # 全局强类型状态 (ResearchState)
├── agents/
│   ├── bottleneck_miner.py     # Agent 1: 文献挖掘
│   ├── hypothesis_generator.py # Agent 2: 方案合成 + 去重检查
│   ├── theoretical_critic.py   # Agent 3: 理论审查
│   ├── code_architect.py       # Agent 4: 实验代码生成
│   ├── diagnostician.py        # Agent 5: 执行结果诊断
│   ├── poison_generator.py     # Agent 6: 对抗数据生成（策略库选择）
│   └── publish_evaluator.py    # Agent 7: 成果验收
├── tools/
│   ├── semantic_scholar.py     # Semantic Scholar API 封装
│   ├── code_executor.py        # Docker 沙箱执行
│   └── perturbation_strategies.py  # 7 种固定扰动策略
├── graphs/
│   ├── hypothesis_graph.py     # Phase 1 子图
│   ├── experiment_graph.py     # Phase 2 子图
│   └── main_graph.py           # 主图（大循环控制）
└── utils/
    └── similarity.py           # 余弦相似度（TF-IDF，离线可用）
```

## 关键防护机制

| 风险 | 防护机制 |
|------|----------|
| 重复假设 | 余弦相似度 > 0.85 → DuplicateHypothesisError |
| 代码死循环 | Docker 资源限制 + 300s 超时 |
| 无限大循环 | `max_outer_loops=5` 硬上限 |
| 分数虚高 | `publish_matrix` 四项全绿才能 END |
| 扰动测试无效 | Agent 6 只能从固定策略库中选择 |

## 环境要求

- Python 3.11+
- Docker Desktop（代码执行沙箱必需）
- Anthropic API Key（或 OpenAI API Key）

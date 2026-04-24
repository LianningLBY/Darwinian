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
├── state.py                    # 全局强类型状态 (ResearchState + ConceptGraph v2)
├── agents/
│   ├── bottleneck_miner.py     # Agent 1: 文献挖掘 + ConceptGraph 构建
│   ├── hypothesis_generator.py # Agent 2: 方案合成 + 硬约束校验 + step 7.5 组合查重
│   ├── theoretical_critic.py   # Agent 3: 理论审查
│   ├── code_architect.py       # Agent 4: 实验代码生成
│   ├── diagnostician.py        # Agent 5: 执行结果诊断
│   ├── poison_generator.py     # Agent 6: 对抗数据生成（策略库选择）
│   └── publish_evaluator.py    # Agent 7: 成果验收
├── tools/
│   ├── semantic_scholar.py     # S2 API: 搜索 + 引用图遍历 + pickle 缓存 (7d TTL)
│   ├── code_executor.py        # Docker 沙箱执行
│   └── perturbation_strategies.py  # 7 种固定扰动策略
├── graphs/
│   ├── hypothesis_graph.py     # Phase 1 子图
│   ├── experiment_graph.py     # Phase 2 子图
│   └── main_graph.py           # 主图（大循环控制）
└── utils/
    ├── similarity.py           # 余弦相似度（TF-IDF，离线可用）
    ├── json_parser.py          # LLM JSON 解析（含截断修复）
    ├── llm_retry.py            # 瞬时网络错误重试
    └── knowledge_graph.py      # Phase 1 v2: ConceptGraph 构建管道
```

## Phase 1 v2 — 从"LLM 自由联想"到"在真实论文网络上填空"

Agent 1 + Agent 2 不再让 LLM 凭空发挥：

1. **分两档检索** — 经典论文 + 近三年新作
2. **一跳引用图扩展** — refs + citations，去重剪枝到 top 60 篇
3. **批量实体抽取** — Haiku 级小模型从 title+abstract 抽 {method, dataset, metric, task_type, limitations}
4. **别名合并** — word-boundary substring containment（`adam` ⊂ `adam optimizer` 合并；`bert` ⊂ `bertopic` **不**合并）
5. **相关性裁剪** — TF-IDF top 60 + 论文数 top 20 兜底（防跨域冷门被误杀）
6. **结构洞发现** — 共现矩阵找"高频但从未共现"的术语对，score = min(papers_a, papers_b)
7. **硬约束** — Agent 2 每分支必须：cited ≥ 2 entities from ≥ 2 papers、绑 1 个 limitation_id、跨分支覆盖 ≥ 2 task_types
8. **反馈重试** — 校验失败时带具体候选（同类型 + 按论文数降序 top 3）重试最多 3 次
9. **组合查重（step 7.5）** — cited entities 组合去 S2 查重，标题+摘要同时含所有术语才算命中
10. **优雅降级** — concept_graph 数据不足时 Agent 2 走老 prompt 自由发挥（不强制硬约束）

完整设计见 `PHASE1_V2_DESIGN.md`。

环境变量：
- `SEMANTIC_SCHOLAR_API_KEY` — 可选，提速到 10 req/s
- `DARWINIAN_S2_CACHE_DIR` — S2 缓存目录，默认 `.cache/s2/`
- `DARWINIAN_S2_CACHE_TTL` — 缓存 TTL（秒），默认 7 天

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

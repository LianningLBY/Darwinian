# 经验教训沉淀

## 2026-03-19 | 初始架构搭建

**Commit**: (见本次提交 ID)

### 关键决策记录

1. **Agent 6 策略收窄**
   - 问题：完全开放的扰动代码生成无法保证测试有效性，可能产生"毒药太弱 → 鲁棒性虚高"的反效果
   - 决策：将 Agent 6 职责缩窄为从 `perturbation_strategies.py` 固定策略库（7 种）中选择组合
   - 效果：鲁棒性测试结果可量化、可复现

2. **去重使用本地 TF-IDF 而非外部 Embedding API**
   - 问题：外部 embedding 服务增加依赖、成本和延迟
   - 决策：使用哈希技巧实现 512 维 TF-IDF 向量，离线可用
   - 权衡：精度略低于语义 embedding，但对去重任务已足够

3. **代码执行沙箱使用 Docker**
   - 问题：LLM 生成的任意代码存在安全风险（文件系统访问、网络请求、无限循环）
   - 决策：Docker 容器 + `--network none` + CPU/内存限制 + 300s 超时
   - 注意：需要 Docker Desktop 运行，CI 环境需要特殊配置

4. **failed_ledger 的写入时机**
   - 设计：`write_math_error_to_ledger`、`write_insufficient_to_ledger`、`write_robustness_fail_to_ledger` 作为独立节点，而非在路由函数中直接写
   - 原因：LangGraph 的路由函数只能返回路由字符串，不能修改状态

### 已知限制（待后续迭代解决）

- [ ] 代码执行沙箱的 Docker 镜像 (`darwinian-sandbox:latest`) 需要手动 build
- [ ] Agent 间没有实现 token 消耗追踪，`budget_state.remaining_tokens` 暂不自动更新
- [x] Phase 2 的 `poison_code` 已在 `ExperimentCode` 中增加专用字段，不再借用 `dataset_loader_code`

---

## 2026-04-23 | Phase 1 v2 改造（ConceptGraph + 硬约束）

**Commits**: f4ab5de (doc) → b95f67d (schema) → 931a2c0 (S2) → 67f3c0a (kg base)
→ 2bd2328 (extract+canonical) → 47916a4 (novel_pairs+relevance) → 15268cf (miner)
→ 33a683f (hypothesis v2)

### 背景与动机

用户反馈："流程能走完但效果差"。原因：Agent 2 的 `abstraction_tree` 每个字段全靠
LLM 自律，没有硬约束——`source_domain` 可以写"控制论"、`math_formulation` 可以是 σ/∇
堆砌，下游没人验证。本次改造把 Agent 2 从"自由作家"降级为"填空题选手"。

### 关键决策

1. **不做 SciMON 多轮循环，做"差异化陈述"的压缩版**
   - 权衡：SciMON 3 轮循环每 branch 要 6-9 次额外 LLM 调用，与硬约束校验梯度冲突
     （"差异化" vs "只能用实体表"两头拉扯 Agent 2）
   - 决策：step 7.5 做一次性组合查重，命中后标记 `existing_combination=True`，
     传给 Agent 3 (`theoretical_critic`) 作为判 NOT_NOVEL 的输入信号
   - 成本：每 branch 多 1 次 S2 查询，无额外 LLM 调用

2. **别名合并不用 Levenshtein**
   - 权衡：Levenshtein ≤ 2 会合并 `BERT`/`BART`、`GPT-2`/`GPT-4` 这些真正不同的模型
   - 决策：只用 "lower + 去标点精确匹配" + "词边界 substring containment"
     （regex `\b`），配合 LLM prompt 里 "canonical_name 最短通用英文名全小写" 的约束
   - 反例测试锁死：`bert` ⊄ `bertopic`、`gan` ⊄ `organ`

3. **source_domain 字段 optional 而非删除**
   - 权衡：设计文档要求删，但 code_architect / poison_generator 都在读此字段，
     级联改动超出 schema commit 的原子范围
   - 决策：本次改 `source_domain: str = Field(default="")`，v2 prompt 不再要求填写，
     下游读到空字符串不崩；后续 commit 清理 prompt 引用
   - 已在 state.py 加 `[已弃用]` 注释标记

4. **带候选结构化反馈而非简单"重试"**
   - 观察：LLM 校验失败后如果只告诉 "你错了" 它会继续瞎猜
   - 决策：`_build_validation_feedback` 对每条 error 生成具体候选（MISSING_ENTITY 用
     word-boundary substring 匹配，按 paper_ids 数降序取 top 3；INVALID_LIMITATION
     列 graph.limitations 里前 10 个 id）
   - 代价：反馈文本变长，但内层重试收敛率显著提升

5. **降级路径：is_sufficient=False 时走 v1 老 prompt**
   - 权衡：硬约束必须要求最低数据量（entities ≥ 20, papers ≥ 10），否则卡死
   - 决策：`ConceptGraph.is_sufficient` 字段显式标记，Agent 2 据此切换 prompt；
     state.messages 会打印警告便于追查
   - 防止 S2 限流/空 abstract 等"罕见但必然发生"场景直接让 Phase 1 死循环

### 踩的坑

1. **pip install -e 的包路径不跟 worktree 走**
   - 症状：在 worktree 里 `pytest` 报 `ImportError: cannot import name 'Entity'`——
     Python 仍然从主项目路径加载 `darwinian.state`
   - 解法：所有 pytest 命令前加 `PYTHONPATH=$(pwd)/src`
   - 已记入 TEST.md 的 "Worktree 下运行测试" 章节

2. **基线测试 2 项预挂**
   - 发现：`test_loop_limit_forces_end` 和 `test_success_routes_to_poison_generator` 在
     Phase 1 v2 改造**之前**就已挂起，routing 逻辑和测试期望不一致
   - 决定：不在本次 task 范围内修复（需独立 task 对 routing 行为达成共识），
     但记入 TEST.md 的"预挂测试"小节避免混淆

### 测试增量

commit f4ab5de → 33a683f 期间：
- 新增测试文件 5 个：test_semantic_scholar.py / test_knowledge_graph.py /
  test_bottleneck_miner.py / test_hypothesis_generator.py
- test_state.py 扩展：新增 TestConceptGraph + TestAbstractionBranchV2
- 总测试数：50 → 156 pass (2 pre-existing fail)

### 设计取舍留给 v2 Phase 2

- **Exa API 接入**：step 7.5 目前仅用 S2 查重，设计文档建议 Exa + S2 双路，
  但 exa-py 依赖未加、token 额度需用户自己申请，留作后续 commit
- **arxiv 补充源**：同理，能扩大种子论文覆盖面但需新依赖
- **Daytona/Modal 替换 Docker**：解决 `code_executor.py` 的 GPU 短板，属 Phase 2 改造

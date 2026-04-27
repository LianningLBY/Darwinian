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

---

## 2026-04-26 | Phase 1 v3 seed-schema gap 补齐

**Commit**: `9219fb0` (feat(state): Phase 1 v3 seed-schema 新增 6 个 schema)

### 关键决策

1. **ResearchMaterialPack 作为 Phase A→B 边界容器**
   - 问题：之前 elaborator 要从 ConceptGraph + list[PaperEvidence] + ResearchConstraints + failed_ledger 多个对象自己拼上下文，容易漏字段
   - 决策：把"喂给 Phase B 的所有素材"打包成单一 Pydantic 对象，elaborator 收一份就够
   - 配套 `evidence_by_category` property，给 existing_methods section 直接用，避免 elaborator 二次分组算

2. **ExpectedOutcomes 拆字段而非保留单 str**
   - 问题：QuantSkip 的关键发表保险是"正反两种结果都 publishable"——这个叙事元素如果埋在自由文本里，elaborator 极易漏掉
   - 决策：拆成 positive_finding / negative_finding / why_both_publishable 三个 required 字段
   - 兼容：保留 expected_outcomes: str 字段，新增 expected_outcomes_structured: ExpectedOutcomes | None，老 elaborator 不破坏

3. **DebateRound/Result 提前 schema 化但不写 prompt**
   - 权衡：完整 Phase C 应包含 Advocate/Challenger/Judge prompt 模板
   - 决策：只先写 schema (DebateRound + DebateResult + delta_last_two/is_above_threshold property)，prompt 模板待 seed 质量稳定后再实现
   - 理由：辩论 Judge 给的"中稿率"只在 seed 质量到 QuantSkip 档次才有意义，否则 Judge 会对一坨平庸 seed 反复打 15%，浪费 token

### 踩的坑

1. **stash 期间路径模式不通过**
   - 症状：`git stash push -- 'src/**/*.pyc'` 在 zsh 下 glob 行为不一致，stash 内容空
   - 解法：stash 用显式路径，或先 `git rm --cached` 处理 .pyc
   - 预防：以后 stash pyc 噪声直接 `git checkout -- 'src/**/*.pyc'` 更直接

2. **HTTPS PAT 过期 + SSH key 跨账户失效**
   - 症状：origin remote 用嵌入 PAT 的 HTTPS URL，token 过期；本机 SSH key 是另一个 GitHub 账户，对 LianningLBY/Darwinian 无 push 权限
   - 解法：用户提供新 PAT 后，**一次性** push 而不写入 git config（`git push https://<TOKEN>@github.com/...`）
   - 预防：CLAUDE.md 标的 SSH remote `git@github.com:LianningLBY/Darwinian.git` 仅文档说法，实际 origin 是 HTTPS+token 形式；token 过期需用户介入，agent 别自己改 git config

3. **worktree 删不掉因为 .pyc 噪声**
   - 症状：`git worktree remove` 报 "contains modified or untracked files" — pytest 跑完留一堆 .pyc
   - 解法：`git worktree remove --force`（.pyc 是构建产物，丢弃安全）
   - 预防：pytest 跑前 `export PYTHONDONTWRITEBYTECODE=1` 或 worktree 加 .gitignore（暂未做）

### 测试增量

- test_state.py: +6 个 TestClass / +16 个 test (TestStructuralHoleHook×3, TestResearchConstraints×2, TestExpectedOutcomes×2, TestResearchProposalExpectedOutcomesField×2, TestResearchMaterialPack×3, TestDebate×4)
- 总测试: 308 → 324 pass (2 pre-existing fail 未变)

---

## 2026-04-26 | seed_renderer + diff demo

**Commit**: `4fbe487` (feat(seed_renderer): ResearchProposal → QuantSkip 风格 seed.md)

### 关键决策

1. **先做 render-only diff，不接 LLM**
   - 问题：直接跑 elaborator + LLM 看输出，成本高且无法分离"schema 不全"和"prompt 不够好"两类问题
   - 决策：手工构造一份"理想 ResearchProposal"作 ground truth，渲染后对照 QuantSkip 原文
   - 收益：先确认 schema + 渲染层覆盖度 100%，再去优化 LLM 端；定位问题精度高

2. **缺字段渲染 '(待补)' 而非 raise**
   - 问题：LangGraph 节点中如果 elaborator 漏字段就让渲染崩溃，会让整条链路在 Phase 1 末尾失败
   - 决策：renderer 是 best-effort，缺字段画占位符，让 markdown 仍能完整渲出
   - 配套：占位符 '(待补)' 显眼，HITL/QA 看一眼就知道哪里要补

3. **key_references_formatted 不前缀 short_name**
   - 问题：demo 里写成 `f"{ev.short_name}: {ev.title} ({ev.venue})"` 渲出来变成
     "LayerSkip: LayerSkip: Enabling..."（重复）
   - 原因：论文 title 通常已以方法名开头（"LayerSkip: ..."、"DEL: ..."、"RAMP: ..."）
   - 决策：直接用 `f"{ev.title} ({ev.venue})"`，不重复前缀
   - 配套：未来 elaborator 输出时也按此格式

### diff vs QuantSkip 原文的发现

渲染输出基本对齐，**信息密度足够**。仅 3 处 cosmetic 已修：
- 标题与 metadata 块之间空行
- metadata 块四行用 markdown 硬换行 `  \n` 连续显示
- references 不前缀 short_name 防重复

**schema 完全覆盖** QuantSkip 所有 section，无需新增字段。

### 留给后续的事

- elaborator 当前接 ConceptGraph，未接 ResearchMaterialPack — 下一个 task 升级
- diff demo 只是手工构造的 "ideal proposal"，没验证 LLM 能产到这个质量 —
  接好 elaborator 后跑实测才能验证
- StructuralHoleHook 已构造但 elaborator 没读 — 接口升级时一起接

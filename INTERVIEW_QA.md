# 面试问答整理

> 针对大模型应用开发岗位，基于 Darwinian 项目的面试题及参考答案。

---

## 一、系统设计类

### Q1：你的多 Agent 系统是如何编排的？为什么选 LangGraph 而不是直接写代码调用？

**答**：

系统用 LangGraph 的 `StateGraph` 编排 7 个 Agent。选 LangGraph 而非手写调用链，核心原因有三：

1. **有状态**：LangGraph 将全局状态（`ResearchState`）贯穿所有节点，每个节点只返回需要更新的字段，框架自动合并，避免手写时的状态传递混乱。

2. **条件路由**：`add_conditional_edges` 让路由逻辑（PASS/NOT_NOVEL/MATH_ERROR）与执行逻辑完全解耦，修改路由规则不需要改 Agent 代码。

3. **子图嵌套**：Phase1 和 Phase2 分别是独立编译的子图，注册为主图的节点。这样每个阶段可以独立测试，也方便大循环（Phase2 失败 → 重启 Phase1）的实现。

---

### Q2：Phase1 和 Phase2 子图之间如何共享状态？大循环是怎么实现的？

**答**：

LangGraph 子图和主图共享同一个 `ResearchState` 对象，子图节点返回的 dict 会直接更新全局状态，所以 Phase1 写入的 `current_hypothesis.selected_branch` 在 Phase2 中直接可读。

大循环通过主图的条件路由实现：

```python
graph.add_conditional_edges("phase2", phase2_result_router, {
    "restart_phase1": "phase1",   # 失败则回到 Phase1 节点
    "end_publish_ready": "save_results",
    "end_failed": "log_termination",
})
```

`phase2` 节点执行完后调用 `phase2_result_router`，若失败且未超出最大循环次数，直接路由回 `phase1` 节点，形成环路。`outer_loop_count` 在每次 `preprocess_node` 中 +1，超过 `max_outer_loops`（默认 5）则终止。

---

### Q3：你是如何防止无限循环的？为什么需要多个计数器？

**答**：

系统有三个独立计数器，分别针对不同层级的循环：

| 计数器 | 触发场景 | 上限 | 重置时机 |
|--------|----------|------|----------|
| `outer_loop_count` | 每轮大循环（Phase1 + Phase2） | 5 | 从不重置（全局单调递增） |
| `hypothesis_retry_count` | NOT_NOVEL，Agent 2 内层重试 | 5 | 每次 Phase1 开始时 |
| `miner_retry_count` | MATH_ERROR，Agent 1 重跑 | 3 | 每次 Phase1 开始时 |

为什么不能用一个计数器？因为 `outer_loop_count` 是跨 Phase1+Phase2 的全局计数，如果用它来限制 Phase1 内层重试，在 `max_outer_loops=1` 时，`preprocess_node` 把 `outer_loop_count` 设为 1，`1 >= 1` 立即触发退出，第一轮连 Agent 2 都没机会跑。必须用独立计数器在 Phase1 开始时清零，才能让每轮 Phase1 有完整的重试配额。

---

### Q4：认知账本（Cognitive Ledger）的设计思路是什么？

**答**：

认知账本是系统跨轮次学习的核心机制，解决的核心问题是"不重复踩坑"。

**写入时机**：每次失败（MATH_ERROR / NOT_NOVEL / INSUFFICIENT / ROBUSTNESS_FAIL）都在对应的 ledger 节点写入一条记录，包含：语义嵌入向量、失败原因、失败类型、禁用关键词。

**使用时机**：
- Agent 2 生成新假设后，计算与账本中每条记录的余弦相似度，超过 0.85 则抛出 `DuplicateHypothesisError`，强制重新生成
- Agent 1 检索文献时，账本中的 `banned_keywords` 作为过滤词，从源头规避重复方向

**为什么用余弦相似度而不是精确匹配**：假设的表达形式每次都不同（LLM 生成的自然语言），精确匹配必然失效。余弦相似度基于语义嵌入，能识别"换了个说法但本质相同"的假设。

---

## 二、LLM 工程类

### Q5：你遇到了哪些 LLM 输出不稳定的问题？怎么解决的？

**答**：遇到了 4 类典型问题：

**① 输出截断（超 token 上限）**：LLM 在 JSON 中途停止，如 `{"key": "val` 后面没了。解决：`json_parser.py` 中实现截断修复——逐字符追踪嵌套栈（`{}` / `[]` / 字符串引号），在末尾自动补全缺失的引号和括号。

**② `<think>` 块未关闭**（推理模型特有）：MiniMax / DeepSeek-R1 等推理模型会先输出 `<think>` 推理过程，网络断连时 `</think>` 还没发出就截断了，导致真正的 JSON 被包裹在 `<think>` 内。解决：从后往前扫描，找第一个以 `{` 开头且包含 `"key":` 模式的行，跳过推理块直接提取 JSON。判断标准需排除 LaTeX 数学公式中的花括号（它们不含 `"key":` 模式）。

**③ LaTeX 公式中的非法转义**：LLM 在 JSON 字符串里写 `\mathcal`、`\sum` 等 LaTeX 命令，而 JSON 规范不允许 `\m` 这样的转义序列。解决：正则替换 `\\(?!["\\/bfnrtu])` → `\\\\`，将非法反斜杠全部转义。

**④ 字符串内的字面换行**：LLM 生成代码时直接在 JSON 字符串值中换行，而非用 `\n`。解决：逐字符扫描，在字符串上下文内将 `\n`、`\r`、`\t` 替换为转义形式。

---

### Q6：推理模型（如 MiniMax M2.7）和普通 chat 模型在工程上有什么不同？

**答**：主要有两点工程差异：

**① 输出结构不同**：推理模型会先输出 `<think>` 推理链，再输出实际回答。如果直接 `json.loads(response.content)`，拿到的是 `<think>` 块而非 JSON。需要在解析前先剥离推理块：`re.sub(r"<think>[\s\S]*?</think>", "", text)`。

**② 响应更长、截断风险更高**：推理模型因为要先"想"再"说"，token 消耗更大，截断概率比普通模型高得多。需要将生成任务拆小（我们把代码生成拆成 4 次独立调用），同时解析层要有截断修复能力。

---

### Q7：你的 LLM 调用重试策略是怎么设计的？

**答**：

分两层重试，针对不同失败类型：

**网络层**（`llm_retry.py`）：捕获 `RemoteProtocolError`、`incomplete chunked read`、`peer closed` 等流式连接断开错误，线性退避重试（5s / 10s / 15s，最多 3 次）。这类错误是服务端瞬时关闭连接导致的，等待后重试成功率高。

**解析层**（`hypothesis_generator.py`）：如果 LLM 响应返回了但 JSON 解析失败（空 `abstraction_tree`），在 Agent 内部直接重试 LLM 调用（最多 3 次，间隔 5/10/15s）。这样避免每次解析失败都走 MATH_ERROR 路径重跑整个 Agent 1，节省大量时间。

**区分重试与放弃**：业务错误（如 `CriticVerdict.NOT_NOVEL`）不重试网络，只走流程重试。只有明确的网络/解析异常才触发底层重试。

---

### Q8：如何让 LLM 稳定生成结构化 JSON？

**答**：我用了三个手段：

1. **Prompt 约束**：System Prompt 末尾加 `"禁止输出 JSON 以外的任何内容"`，并提供完整的 JSON 示例模板，字段名和格式都写清楚。

2. **Pydantic 强类型校验**：LLM 输出的 JSON 会被 `Hypothesis(**raw)` 等 Pydantic 模型解析，字段类型不对或缺失会立即抛出异常，而不是静默传播错误。

3. **解析兜底 + 重试**：就算 Prompt 写得再好，LLM 偶尔还是会格式错误（尤其是推理模型的 `<think>` 块）。`json_parser.py` 提供 5 级修复，修复失败则触发内层重试，而非直接崩溃。

---

## 三、代码生成类

### Q9：让 LLM 生成可运行实验代码，最大的挑战是什么？你是怎么解决维度一致性问题的？

**答**：

最大的挑战是**维度一致性**。系统分 4 次独立调用生成 `loader / baseline / proposed / ablation` 代码。如果 `proposed.py` 对输入特征维度 `n_features` 的猜测与 `loader.py` 实际返回值不一致，运行时必然报 shape mismatch。

这个问题之前每次都触发：proposed 猜 `n_features=64`，实际是 `20`，导致 5 次重试全部失败。

解决方案是**在生成 proposed 代码时，将 `loader_code` 的前 600 字符注入 prompt**：

```python
user = f"""以下是本次实验的 dataset_loader.py（必须与此保持维度一致）：
{loader_code[:600]}
任务：编写 proposed.py，输入维度必须与 load_data() 返回的 X_train.shape[1] 一致"""
```

LLM 看到真实的 loader 代码后，可以直接读取 `n_features` 而不是猜测。另外，代码重试时（`retry_count > 0`）直接复用上一轮的 `loader_code`，防止重试时再次生成不同维度的 loader。

---

### Q10：实验代码执行失败时，系统如何自愈？

**答**：

系统有一个内层修复循环（最多 5 次）：

1. `code_execute_node` 执行代码，捕获 stdout / stderr
2. `diagnostician_node`（Agent 5）分析输出，判断是 `code_error`（程序崩溃）还是 `insufficient`（方法无效）
3. 若是 `code_error`：将 `stderr` 和诊断结论注入 `retry_context`，传给 `code_architect_node`（Agent 4）重新生成 proposed 代码，`retry_count + 1`
4. 若 `retry_count >= 5`：放弃，写入账本，类型标记为 `CODE_BUG`，触发大循环换假设

关键设计：重试时 Agent 4 能看到具体的报错信息和维度不匹配的数字（从 stderr 中 regex 提取），生成的修复代码针对性强。

---

## 四、算法与 ML 类

### Q11：NOT_NOVEL 的判定逻辑是什么？如何避免误判？

**答**：

NOT_NOVEL 由 Agent 3（理论审查官）判定，System Prompt 中明确了判断标准：
- 改变数据集/超参数/损失函数系数**不构成**新颖性
- 简单组合两个已有方法不构成新颖性（除非组合方式本身有创新）
- 必须指出与哪篇已有工作本质相同

为防止 Agent 3 过于保守（倾向 NOT_NOVEL），Prompt 中加了 `"如果不确定是否新颖，倾向于判 NOT_NOVEL 而非 PASS"`——这看起来是加强了保守，但实际上通过账本机制，NOT_NOVEL 记录会让 Agent 2 下次避开相似方向，倒逼其生成真正有差异的假设。

另外，Agent 3 对空 `abstraction_tree` 有前置拦截：若 Agent 2 返回空树，直接返回 MATH_ERROR，不进入 LLM 判断逻辑，避免把系统故障误判为方案本身的问题。

---

### Q12：鲁棒性测试是如何自动化的？

**答**：

系统维护一个**扰动策略库**（`perturbation_strategies.py`），包含标签噪声、特征缺失、分布偏移、对抗扰动等多种策略。

Agent 6（鲁棒性测试生成器）的流程：
1. 分析当前方法的核心假设（从 `selected_branch.algorithm_logic` 提取）
2. 从策略库中选择 3-5 种最能"打破该假设"的策略（LLM 判断）
3. 生成鲁棒性测试代码，在扰动数据上运行 proposed 方法
4. 计算性能下降比例 `degradation_rate`

Agent 7 的验收标准：性能下降 < 30% 且覆盖至少 2 种扰动维度，才算鲁棒性通过。

---

## 五、工程素养类

### Q13：你在这个项目中遇到的最难的 Bug 是什么？

**答**：

最难的是 `outer_loop_count` 被同时用于外层循环限制和内层 NOT_NOVEL 重试判断，在 `max_loops=1` 时触发逻辑矛盾：

- `preprocess_node` 在 Phase1 开始时将 `outer_loop_count` 从 0 加到 1
- `critic_router` 中原逻辑是：`if outer_loop_count >= max_outer_loops: return "__end__"`
- 设 `max_loops=1` 时，Phase1 第一次 NOT_NOVEL，`1 >= 1` 立即退出，连重试机会都没有

这个 Bug 的难点在于：外层循环和内层重试共用了同一个计数器，逻辑上混乱，但在 `max_loops > 1` 时偶然可以工作（因为内层耗尽时外层计数也确实更大了），掩盖了问题。

解决方案：新增独立的 `hypothesis_retry_count` 和 `miner_retry_count`，分别追踪内层 NOT_NOVEL 重试和 MATH_ERROR 重试，与外层 `outer_loop_count` 完全解耦，每轮 Phase1 开始时清零。

---

### Q14：如果让你重新设计这个系统，你会改什么？

**答**：

1. **异步 LLM 调用**：目前 Agent 1-4 是串行的，但 `baseline_code` 和 `proposed_code` 生成实际上可以并行（都依赖 `loader_code`）。改成 `asyncio` 并发调用可以减少约 30-40% 的等待时间。

2. **更细粒度的状态快照**：目前失败后整个 Phase 重跑，理想情况下应该能从失败点续跑（如 `code_error` 只重跑 Agent 4，不重跑 Agent 1-3）。LangGraph 的 Checkpoint 机制可以支持这一点。

3. **账本的向量数据库**：随着失败记录积累，全量余弦相似度计算复杂度是 O(n)，记录多了会有性能问题。改用 FAISS 或 ChromaDB 做 ANN 近似检索更合理。

4. **推理模型的流式输出感知**：目前等待完整响应后再解析，对于 5000+ token 的推理输出等待时间很长。改成流式解析，`<think>` 块完成时立即切换到 JSON 解析模式，可以提前失败检测。

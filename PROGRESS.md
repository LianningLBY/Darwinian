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

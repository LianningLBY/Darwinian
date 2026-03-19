# 测试指南

## 运行测试

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行全部测试
pytest tests/ -v

# 运行并生成覆盖率报告
pytest tests/ --cov=src/darwinian --cov-report=term-missing

# 运行单个测试文件
pytest tests/test_state.py -v
pytest tests/test_tools/test_similarity.py -v
pytest tests/test_tools/test_perturbation_strategies.py -v
pytest tests/test_graphs/test_routing.py -v
```

## 测试分层

### 单元测试（无 LLM 调用）

| 测试文件 | 覆盖范围 |
|----------|----------|
| `tests/test_state.py` | ResearchState 所有字段验证、枚举类型、PublishMatrix 逻辑 |
| `tests/test_tools/test_similarity.py` | TF-IDF 嵌入、余弦相似度计算 |
| `tests/test_tools/test_perturbation_strategies.py` | 7 种扰动策略代码模板执行正确性 |
| `tests/test_graphs/test_routing.py` | critic_router / execution_router / final_router 路由逻辑 |

### 集成测试（需要 LLM API Key）

> 集成测试需要真实 LLM，默认跳过（使用 `pytest -m integration` 运行）。

| 测试文件 | 覆盖范围 |
|----------|----------|
| `tests/test_agents/` (待补充) | 各 Agent 的 prompt → JSON 解析链路 |
| `tests/test_graphs/test_phase1_e2e.py` (待补充) | Phase 1 子图端到端 |

## 新增功能的测试规范

每新增一个 Agent 或工具，必须同步：

1. **状态相关**：在 `test_state.py` 中添加对应字段的验证测试
2. **工具函数**：在 `tests/test_tools/` 下添加纯函数单元测试
3. **路由逻辑**：在 `tests/test_graphs/test_routing.py` 中添加路由条件测试
4. **更新本文件**：在上方表格中登记新的测试覆盖

## 测试原则

- **路由测试不调用 LLM**：通过构造特定 State 直接测试路由函数
- **扰动策略测试用 exec**：直接执行 code_template，用 numpy 验证输出
- **相似度测试用语义常识**：相关文本相似度 > 0.3，不相关文本 < 0.5

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
| `tests/test_state.py` | ResearchState 所有字段、枚举类型、PublishMatrix、ConceptGraph v2 schema、AbstractionBranch v2 新字段、v3 seed-schema (StructuralHoleHook / ResearchConstraints / ExpectedOutcomes / ResearchMaterialPack / DebateRound / DebateResult) |
| `tests/test_tools/test_seed_renderer.py` | ResearchProposal → markdown 渲染：metadata 块、各 section 占位符、phases / outcomes structured 优先、references formatted vs raw、resource estimate 三档 |
| `tests/test_tools/test_similarity.py` | TF-IDF 嵌入、余弦相似度计算 |
| `tests/test_tools/test_perturbation_strategies.py` | 7 种扰动策略代码模板执行正确性 |
| `tests/test_tools/test_json_parser.py` | LLM JSON 输出解析、代码块剥离、截断修复 |
| `tests/test_tools/test_semantic_scholar.py` | 缓存键稳定性、references/citations 包装解包、分两档检索去重、api_key 环境变量 |
| `tests/test_tools/test_knowledge_graph.py` | expand_one_hop、filter_and_rank、normalize、word-boundary (bert⊄bertopic)、批抽取、canonicalize、相关性裁剪、结构洞、充分性、build_concept_graph 编排 |
| `tests/test_agents/test_bottleneck_miner.py` | ConceptGraph 管道接入、banned_keywords 在 entity 层过滤、formatters、降级 prompt |
| `tests/test_agents/test_hypothesis_generator.py` | 硬约束 5 种 error code、候选建议 (word-boundary+兜底)、结构化反馈、step 7.5 组合查重、v2/降级双路径 |
| `tests/test_agents/test_proposal_elaborator.py` | v2 ConceptGraph 路径（_build_proposal / _validate 6 种 error code / 重试 / node wrapper）+ v3 ResearchMaterialPack 路径（按 category 分组 prompt / forbidden_techniques 校验 / structured outcomes / resource estimate 兜底） |
| `tests/test_agents/test_phase_a_orchestrator.py` | helper（_looks_like_arxiv_id / _format_evidence_id / _bucket_by_year）/ _resolve_arxiv_ids 走 S2（含异常吞掉）/ _make_full_text_provider 按 evidence_paper_id 反查 / build_research_material_pack 端到端串接 / Scheme X 完整路径（_llm_list_seed_papers / _verify_and_recover_seed 含 title 回捞 / _title_similarity / _expand_seeds_one_hop / _rerank_by_direction_relevance / build_seed_pool 端到端） |
| `tests/test_graphs/test_routing.py` | critic_router / execution_router / final_router 路由逻辑 |

### 预挂测试（基线，非 Phase 1 v2 引入）

| 测试 | 说明 |
|------|------|
| `test_loop_limit_forces_end` | critic_router 的 outer_loop_count 上限逻辑，预先就挂 |
| `test_success_routes_to_poison_generator` | execution_router 成功路径，预先就挂（routing 改到 ablation_execute 优先） |

> 这 2 条在 Phase 1 v2 改造**之前**就已存在，本改造与之无关。修复需独立 task。

### 集成测试（需要 LLM API Key）

> 集成测试需要真实 LLM，默认跳过（使用 `pytest -m integration` 运行）。

| 测试文件 | 覆盖范围 |
|----------|----------|
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
- **Agent/工具测试 mock 外部 API**：`unittest.mock.patch` 掉 `httpx` / LLM `invoke`，
  不打真实 Semantic Scholar、不烧 LLM token
- **关键反例必测**：例如 `bert` ⊄ `bertopic`、`GPT-2` ≠ `GPT-4` 这类 fuzzy match 陷阱
  必须有测试用例兜底

## Worktree 下运行测试

在 worktree 里跑测试时，需要覆盖 PYTHONPATH 指向 worktree 的 src（因为 `pip install -e .`
装的包指向主项目路径）：

```bash
PYTHONPATH=$(pwd)/src python -m pytest tests/ -v
```

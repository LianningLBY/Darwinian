# Phase 1 v2 改造设计文档

> **目标：** 把 Phase 1（Agent 1 + Agent 2）从"LLM 自由联想"改造成"基于真实论文网络的填空 + 结构洞启发 + 组合查重"管道。
> **范围：** 仅改 Phase 1，Phase 2（代码生成/Docker/验收）一字不动。
> **定位：** 不是"自动提取 idea"，是"让 LLM 编的 idea 每一部分都可追溯、每一种组合都自动查重"。

---

## 一、改前 vs 改后

```
改前（v0）:
  search 20 papers → 拼文字塞给 Agent 2 → LLM 自由编方案 → Agent 3 审

改后（v2）:
  search 40 papers（分两档）
    → 一跳引用图扩展到 ~60 篇
    → 小模型 batch 抽结构化四元组 + limitations
    → 合并别名，汇总成 ConceptGraph
    → TF-IDF 裁剪 + 频次兜底到 ~70 实体
    → 共现矩阵找"高频未共现"结构洞 top-10
    → Agent 2 在术语表 + 结构洞启发下填空生成
    → 机械校验硬约束 + S2 组合查重
    → 不通过带候选反馈重试（复用 3 次内层重试）
    → Agent 3 审（existing_combination 作为输入信号）
```

---

## 二、完整 11 步流水线

### 1. 分两档检索种子

- **输入：** `state.research_direction`
- **做法：**
  ```python
  classics = s2.search(query=research_direction, limit=20)
  recent   = s2.search(query=research_direction, year="2023-2026", limit=20)
  # 去重，优先保留经典档
  seeds = {p["paperId"]: p for p in classics + recent}.values()
  ```
- **输出：** `~40` 篇种子论文 `List[PaperInfo]`

### 2. 一跳引用图扩展

- **做法：** 对每篇种子论文各调一次 `get_references` 和 `get_citations`
- **细节：**
  - S2 返回 `{"citedPaper": {...}}` / `{"citingPaper": {...}}` 结构，需解包
  - `fields=paperId,title,abstract,year,citationCount` 一次性指定
  - 每次 API 请求前先查文件缓存，7 天 TTL
- **输出：** 候选池 `~几百篇 Paper`

### 3. 清洗、剪枝、筛到 60 篇

```python
# 去重
unique = {p["paperId"]: p for p in candidates}.values()
# 过滤空 abstract（含扩展来的 paper）
filtered = [p for p in unique if len(p.get("abstract") or "") >= 100]
# 按引用数剪到 top 60
top_papers = sorted(filtered, key=lambda p: p.get("citationCount", 0), reverse=True)[:60]
```

### 4. 小模型 batch 抽结构化五元组

- **模型：** Haiku（便宜，质量够用）
- **批大小：** 每批 8 篇，共 8 次 LLM 调用
- **prompt 要求每篇产出：**
  ```json
  {
    "paper_id": "...",
    "method": ["canonical_name1", ...],       // 最短通用英文名，全小写
    "dataset": [...],
    "metric": [...],
    "task_type": "classification|regression|...",
    "limitations": ["文本1", "文本2"]          // 每条一句话
  }
  ```
- **预算：** 每次调用 ~7k token × 8 = 56k，显式 `budget_state.remaining_tokens -= used`

### 5. 合并统一（别名归并）

**规则（不用 Levenshtein）：**
1. 精确匹配：lower + 去标点 + 去连字符/下划线后相同 → 合并
2. 词边界子串包含：`re.search(rf"\b{short}\b", long)` → 合并
   - 例：`adam` 在 `adam optimizer` 里是完整词 → 合并
   - 反例：`bert` 在 `bertopic` 里不是完整词 → 不合并
3. 8 次 batch 跑完后做一次全局 post-merge，收集所有 canonical_name 再跑规则

### 6. 实体相关性裁剪到 ~70

```python
core_vec = similarity.get_text_embedding(state.current_hypothesis.core_problem)
scored = [(cosine(core_vec, emb_of(e)), e) for e in entities]
top_60_relevant = sorted(scored, reverse=True)[:60]
top_20_popular  = sorted(entities, key=lambda e: len(e.paper_ids), reverse=True)[:20]
final = dedupe(top_60_relevant + top_20_popular)   # ≈ 70
```

**为什么兜底 top-20 by paper count：** TF-IDF 会误杀跨域冷门词（它们本来就和 core_problem 文字不相似）。

### 6.5. 共现矩阵找结构洞 ⭐️新增

```python
from itertools import combinations

novel_pairs = []
for a, b in combinations(final_entities, 2):
    shared = set(a.paper_ids) & set(b.paper_ids)
    if len(shared) == 0 and len(a.paper_ids) >= 3 and len(b.paper_ids) >= 3:
        # 用 min 而不是 sum，偏向两端都成熟的组合
        score = min(len(a.paper_ids), len(b.paper_ids))
        novel_pairs.append((a, b, score))

novel_pair_hints = sorted(novel_pairs, key=lambda x: x[2], reverse=True)[:10]
```

**产物写入 `ConceptGraph.novel_pair_hints`，作为 Agent 2 prompt 的启发区（非强制）。**

### 7. Agent 2 生成方案

**prompt 结构：**
```
核心问题：{core_problem}

[术语表 — 70 个]
- self_attention (method, 出现在 5 篇)
- flash_attention (method, 出现在 2 篇)
- mamba (method, 出现在 3 篇)
- [BANNED] linear_attention  ← banned_keywords normalize 后在 entity 层标记
...

[待解缺陷 — LimitationRef]
- L_a3f2: "transformer O(n²) 复杂度" — 来自 P3
- L_b8d1: "长序列推理内存爆" — 来自 P5
...

[启发提示 — 结构洞 top 10]
这些术语组合在现有文献中从未共现，可能是潜在创新点：
- (mamba, flash_attention)
- (state_space_model, retrieval_augmented)
...

请生成 2 个分支。每个必须填：
- cited_entity_names: list[str]（来自术语表）
- solved_limitation_id: str（来自待解缺陷）
- 两个分支合起来，cited 的 entities 对应的 paper 必须覆盖 ≥ 2 个不同 task_type
```

### 7.5. 组合新颖性查重 ⭐️新增

```python
for branch in branches:
    query = " ".join(branch.cited_entity_names)  # 例 "mamba flash_attention"
    hits = s2.search(query, limit=10)
    # 标题/摘要里同时包含所有引用术语才算命中
    matched = [h for h in hits if all(
        term.lower() in (h.get("title", "") + h.get("abstract", "")).lower()
        for term in branch.cited_entity_names
    )]
    branch.existing_combination = len(matched) > 0
    branch.existing_combination_refs = [h["paperId"] for h in matched[:3]]
```

**不否决 branch，只打标记。** 传给 Agent 3（`theoretical_critic`）作为判 NOT_NOVEL 的输入信号——架构不变。

### 8. 硬约束校验

```python
def validate_branch(branch, graph) -> list[str]:
    errors = []
    # 每个引用 entity 必须在表里（normalize 后）
    for name in branch.cited_entity_names:
        if normalize(name) not in graph.entity_index:
            errors.append(("MISSING_ENTITY", name))
    # ≥ 2 entities from ≥ 2 papers
    papers = {p for name in branch.cited_entity_names
                for p in graph.entity_index[normalize(name)].paper_ids}
    if len(branch.cited_entity_names) < 2 or len(papers) < 2:
        errors.append(("INSUFFICIENT_COVERAGE", None))
    # ≥ 1 valid limitation
    if branch.solved_limitation_id not in graph.limitation_index:
        errors.append(("MISSING_LIMITATION", branch.solved_limitation_id))
    return errors

# 跨 branch 校验：合并所有 cited paper 的 task_type 集合 >= 2
```

### 9. 带候选结构化反馈重试

**失败反馈格式：**
```
❌ 你引用的实体 "transformer_attention" 不在术语表。
   最接近的候选（同类型、按出现论文数降序）：
     - self_attention (5 篇)
     - multi_head_attention (3 篇)

❌ solved_limitation_id "L_xxx" 不存在。
   可选：L_a3f2, L_b8d1, L_c5e9

请严格从候选中挑选，不要发明新名字。
```

**候选生成：**
- MISSING_ENTITY：同 type 内 word-boundary substring 匹配 → 按 paper_ids 数降序 → 取 top 3
- MISSING_LIMITATION：列出全部现存 `L_*` id

**复用现有 `hypothesis_generator.py:86-99` 3 次重试循环**，不新增计数器。

---

## 三、数据结构（state.py 新增/修改）

```python
class Entity(BaseModel):
    canonical_name: str
    aliases: list[str] = []
    type: Literal["method", "dataset", "metric", "task_type"]
    paper_ids: list[str]

class LimitationRef(BaseModel):
    id: str                           # hashlib.md5((text+paper_id).encode()).hexdigest()[:8]
    text: str
    source_paper_id: str

class PaperInfo(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    year: int = 0
    citation_count: int = 0
    task_type: str = ""               # 从抽取结果填

class EntityPair(BaseModel):
    entity_a: str
    entity_b: str
    score: int                        # min(paper_count_a, paper_count_b)

class ConceptGraph(BaseModel):
    papers: list[PaperInfo] = []
    entities: list[Entity] = []
    limitations: list[LimitationRef] = []
    novel_pair_hints: list[EntityPair] = []
    is_sufficient: bool = False       # 不足则 Agent 2 降级跳硬约束

# 修改：
class AbstractionBranch(BaseModel):
    name: str
    description: str
    algorithm_logic: str
    math_formulation: str
    # source_domain: str              ← 删除
    cited_entity_names: list[str] = []       # 新增
    solved_limitation_id: str = ""           # 新增
    existing_combination: bool = False       # 新增（step 7.5 填）
    existing_combination_refs: list[str] = []  # 新增

# ResearchState 加：
concept_graph: ConceptGraph | None = None
```

---

## 四、文件变更清单

| 文件 | 类型 | 改动 |
|------|------|------|
| `src/darwinian/state.py` | 修改 | 加 `Entity` / `LimitationRef` / `PaperInfo` / `EntityPair` / `ConceptGraph`；改 `AbstractionBranch`；`ResearchState` 加 `concept_graph` |
| `src/darwinian/tools/semantic_scholar.py` | 修改 | 加 `get_references()` / `get_citations()` / `get_paper_details()`；加 pickle 文件缓存（7 天 TTL，key=paperId+endpoint+args）；`SEMANTIC_SCHOLAR_API_KEY` env var 提速 |
| `src/darwinian/utils/knowledge_graph.py` | 新建 | `expand_one_hop()` / `filter_and_rank()` / `batch_extract_entities()` / `canonicalize_merge()` / `rank_relevance_top_k()` / `find_novel_pairs()` |
| `src/darwinian/agents/bottleneck_miner.py` | 修改 | 改造流程：分两档 → 扩展 → 抽取 → 合并 → 裁剪 → 结构洞；写入 `state.concept_graph`；若 `is_sufficient=False` 打警告 |
| `src/darwinian/agents/hypothesis_generator.py` | 修改 | prompt 改造（塞术语表 + 缺陷 + 结构洞）；`validate_branch()`；校验失败构造带候选反馈；复用现有 3 次重试；数据不足时降级走老 prompt |
| `src/darwinian/agents/theoretical_critic.py` | 小修 | prompt 增加 `existing_combination` 信号输入，作为判 NOT_NOVEL 的参考（不改边/路由） |
| `tests/test_knowledge_graph.py` | 新建 | 单元测试：别名合并、结构洞、硬约束、候选生成；S2 用 fixture 不打真实 API |

---

## 五、硬约束 & 降级策略汇总

| 条件 | 动作 |
|------|------|
| `len(entities) < 20` 或 `len(papers) < 10` | `is_sufficient=False`，Agent 2 跳硬约束，落老 prompt，`state.messages` 记警告 |
| Agent 2 校验失败（3 次内层重试） | 走原 MATH_ERROR 路径写 `failed_ledger` |
| S2 限流/网络断 | 返回缓存值，缓存失效时重试 3 次，全失败则该次扩展跳过 |
| banned_keywords 匹配 entity | entity 打 `[BANNED]` 标签，不物理删除（保留统计），prompt 里明示禁止引用 |

---

## 六、预算与成本估算

| 项目 | v0 每轮 | v2 每轮 | 备注 |
|------|---------|---------|------|
| S2 API 调用 | 1 | ~85 | 1 搜索 + 40 refs + 40 citations + N 查重；有缓存后开发期 ≈ 0 |
| LLM 调用数 | 3（Agent 1/2/3） | ~12（+8 抽取 + Agent 2 重试平均 0.5 次） | |
| LLM token 花费 | ~30k | ~90k | 新增主要是 Haiku 侧；Opus 侧只多结构化反馈那点 |
| Phase 1 耗时 | ~60s | ~180s | 首轮无缓存；后续轮缓存命中率 > 80% |

---

## 七、实现顺序（建议 7 个 commit）

1. **schema** — `state.py` 加新类型，跑 `pytest` 验证 Pydantic 通过
2. **semantic_scholar 接口 + 缓存** — 加 `get_references` / `get_citations`，单独脚本打一次真 API 确认 response 结构
3. **knowledge_graph 基建** — `expand_one_hop` + `filter_and_rank`，纯数据处理 + 单测
4. **batch 抽取** — `batch_extract_entities` + `canonicalize_merge`，mock LLM 单测
5. **结构洞 + 相关性裁剪** — `find_novel_pairs` + `rank_relevance_top_k`
6. **bottleneck_miner 接入** — 把整条管线串起来，run 一次端到端冒烟
7. **hypothesis_generator 校验 + 反馈 + 7.5 查重** — 最后上 Agent 2 改造

每个 commit 后确保现有 `pytest tests/` 全绿。

---

## 八、v1 不做的（推 v2）

- GoAI 风格的语义边标签（extension/comparison/baseline）
- TrustResearcher 的 Graph-of-Thought 在图上做 beam search
- AI-Researcher 的多源资源收集（GitHub/HuggingFace）
- ReviewingAgents 真实评审偏好库
- Challenger 对 low_confidence_entity 的针对性攻击
- Phase 2 算力升级（GPU/nvidia-docker）
- 代码生成的 AST 校验 + 维度检查

---

## 九、验收标准（如何知道 v2 好了）

一次人工测试，给定 `research_direction="transformer 长序列效率问题"`：

- [ ] `state.concept_graph.papers` 数量 ≥ 40
- [ ] `state.concept_graph.entities` 数量 ≥ 50，type 分布均衡（method ≥ 15）
- [ ] `state.concept_graph.limitations` 数量 ≥ 10，每条能追到 `source_paper_id`
- [ ] `state.concept_graph.novel_pair_hints` 数量 ≥ 5
- [ ] Agent 2 输出的 2 个分支全部校验通过（或内层重试后通过）
- [ ] 每个分支 `cited_entity_names` 可追溯到 entity 表
- [ ] 每个分支 `solved_limitation_id` 可追溯到 limitation
- [ ] 至少 1 个分支的 `existing_combination=false`（真的没人做过的组合）
- [ ] 缓存命中后第二次跑 Phase 1 耗时 < 30s

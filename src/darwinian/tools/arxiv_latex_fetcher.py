"""
arxiv 论文 LaTeX 源码获取 + 章节切分

为什么不用 PDF：arxiv 大半论文有 LaTeX 源码可下载，比 PDF 准 100 倍：
- 公式 0 损失（直接是 LaTeX）
- 表格保留 \\begin{tabular} 结构
- bibliography 完整 .bbl 文件
- 章节边界明确（\\section{}）
- ~50KB / 论文，纯文本即可处理

接口：
- fetch_arxiv_latex(arxiv_id) -> LatexSource | None
- split_sections(latex_text) -> dict[str, str]   # 内部函数，也可单独用

下游：paper_evidence_extractor 用 split_sections 的输出抽取五元组，
只把 method + experiments + conclusion 三段喂 LLM 节省 token。
"""

from __future__ import annotations

import io
import os
import re
import tarfile
import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass

import httpx

from darwinian.tools.semantic_scholar import _cache_get, _cache_key, _cache_set


# arxiv 不希望被工具持续连发请求；3s 间隔是官方建议
_LAST_REQUEST_TIME: float = 0.0
_MIN_INTERVAL = float(os.environ.get("DARWINIAN_ARXIV_MIN_INTERVAL", "3.0"))

# arxiv e-print 端点：返回 .tar.gz / .tex.gz / 偶尔 .pdf
ARXIV_E_PRINT_BASE = "https://arxiv.org/e-print"

# arxiv 要求带 User-Agent，不带可能 403
_USER_AGENT = "Darwinian/0.1 (research automation; mailto:noreply@example.com)"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class LatexSource:
    """从 arxiv 拉到的论文 LaTeX 源码"""
    arxiv_id: str                    # 不带前缀，如 "2510.23766" 或 "cs/0607055"
    main_tex: str = ""               # 主 .tex 文件原始内容
    sections: dict[str, str] = None  # 切分后的章节: abstract/introduction/method/...
    has_full_text: bool = False      # False = LaTeX 不可用（PDF-only / 老论文）

    def __post_init__(self):
        if self.sections is None:
            self.sections = {}

    def section(self, name: str) -> str:
        """安全取章节文本（不存在返空）"""
        return self.sections.get(name, "")


# ---------------------------------------------------------------------------
# 限流
# ---------------------------------------------------------------------------

def _respect_rate_limit() -> None:
    """阻塞到与上次 arxiv 请求间隔 ≥ _MIN_INTERVAL 秒"""
    global _LAST_REQUEST_TIME
    elapsed = time.time() - _LAST_REQUEST_TIME
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_REQUEST_TIME = time.time()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def fetch_arxiv_latex(arxiv_id: str) -> LatexSource | None:
    """
    从 arxiv 拉论文 LaTeX 源码并切分章节。

    Args:
        arxiv_id: 不带前缀的 arxiv id，如 "2510.23766"。也接受带 "arxiv:" 前缀。

    Returns:
        LatexSource 或 None（论文不存在 / 网络失败 / 返回的不是 LaTeX）
        论文存在但无 LaTeX 源码（如商业转载）时，返 has_full_text=False 的对象
    """
    if not arxiv_id:
        return None

    # 去前缀容错
    aid = arxiv_id.removeprefix("arxiv:").strip()
    if not aid:
        return None

    # 缓存（pickle 整个 LatexSource 对象）
    cache_k = _cache_key("/arxiv/e-print", {"id": aid})
    cached = _cache_get(cache_k)
    if cached is not None:
        return cached

    _respect_rate_limit()

    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(
                f"{ARXIV_E_PRINT_BASE}/{aid}",
                headers={"User-Agent": _USER_AGENT},
            )
            resp.raise_for_status()
            content = resp.content
            content_type = resp.headers.get("content-type", "").lower()
    except Exception:
        return None

    # 解析返回内容
    main_tex = _extract_main_tex_from_archive(content, content_type)

    if not main_tex:
        # 拉到了但不是有效 LaTeX（可能是 PDF / 空 / 损坏）
        result = LatexSource(arxiv_id=aid, has_full_text=False)
    else:
        sections = split_sections(main_tex)
        result = LatexSource(
            arxiv_id=aid,
            main_tex=main_tex,
            sections=sections,
            has_full_text=True,
        )

    _cache_set(cache_k, result)
    return result


# ---------------------------------------------------------------------------
# 解压 + 主 tex 文件检测
# ---------------------------------------------------------------------------

def _extract_main_tex_from_archive(content: bytes, content_type: str) -> str:
    """
    从 arxiv e-print 返回的字节流里抽出主 .tex 文件，**自动展开 \\input{} 引用**。

    arxiv 返回类型多变：
    - 多文件: .tar.gz （最常见，多 .tex + 图，主文件 \\input 子文件）
    - 单文件: .tex.gz （直接是压缩的单 tex）
    - PDF-only: %PDF- 开头（没 LaTeX 源码的论文）

    返回：主 .tex 文件的字符串内容（含 \\input 子文件已内联）；未找到返空。
    """
    if not content:
        return ""

    # 检测 PDF（arxiv 有时只给 PDF）
    if content[:5] == b"%PDF-":
        return ""

    # 尝试 tar.gz
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            tex_files = _collect_tex_files(tar)
            return _find_main_and_expand_inputs(tex_files)
    except (tarfile.ReadError, EOFError):
        pass

    # 尝试 tar（不压缩）
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as tar:
            tex_files = _collect_tex_files(tar)
            return _find_main_and_expand_inputs(tex_files)
    except (tarfile.ReadError, EOFError):
        pass

    # 尝试 gzip 单文件 (.tex.gz)
    try:
        import gzip
        decompressed = gzip.decompress(content)
        text = decompressed.decode("utf-8", errors="ignore")
        if "\\documentclass" in text or "\\begin{document}" in text:
            return text
    except Exception:
        pass

    # 尝试纯文本
    try:
        text = content.decode("utf-8", errors="ignore")
        if "\\documentclass" in text or "\\begin{document}" in text:
            return text
    except Exception:
        pass

    return ""


def _collect_tex_files(tar: tarfile.TarFile) -> dict[str, str]:
    """读 tar 里所有 .tex 文件成 {basename: content}（兼容多种 \\input 路径写法）"""
    files: dict[str, str] = {}
    for member in tar.getmembers():
        if not member.isfile():
            continue
        name = member.name
        if not name.lower().endswith(".tex"):
            continue
        try:
            f = tar.extractfile(member)
            if f is None:
                continue
            text = f.read().decode("utf-8", errors="ignore")
        except Exception:
            continue
        # 同时存全路径和 basename 两种 key，方便 \input 查找
        files[name] = text
        basename = name.rsplit("/", 1)[-1]
        if basename != name:
            files[basename] = text
        # 去 .tex 后缀的 key 也存（\input{abstract} 不带后缀的常见写法）
        if basename.lower().endswith(".tex"):
            files[basename[:-4]] = text
        if name.lower().endswith(".tex"):
            files[name[:-4]] = text
    return files


def _find_main_tex_in_tar(tar: tarfile.TarFile) -> str:
    """
    [兼容旧测试] 在 tar 包里找主 .tex 文件——优先含 \\documentclass 或 \\begin{document}。
    新代码请用 _collect_tex_files + _find_main_and_expand_inputs。
    """
    files = _collect_tex_files(tar)
    if not files:
        return ""

    # 优先含 documentclass 的
    for name, text in files.items():
        if "\\documentclass" in text:
            return text
    # 没有 documentclass 但有 \begin{document}
    for name, text in files.items():
        if "\\begin{document}" in text:
            return text
    # 兜底：第一个
    return next(iter(files.values()))


_INPUT_PATTERN = re.compile(r"\\(?:input|include)\{([^}]+)\}")


def _find_main_and_expand_inputs(tex_files: dict[str, str], max_depth: int = 5) -> str:
    """
    从 .tex 文件集合里找主文件，递归展开 \\input{} / \\include{} 引用。

    Args:
        tex_files: _collect_tex_files 的输出（含原路径、basename、去后缀三种 key）
        max_depth: 防 \\input 循环引用的递归深度上限

    Returns:
        主文件内容（子文件已内联），未找到返空
    """
    if not tex_files:
        return ""

    # 找主文件（优先含 \documentclass）
    main_text = ""
    for text in tex_files.values():
        if "\\documentclass" in text:
            main_text = text
            break
    if not main_text:
        for text in tex_files.values():
            if "\\begin{document}" in text:
                main_text = text
                break
    if not main_text:
        # 兜底：第一个 .tex 文件
        main_text = next(iter(tex_files.values()))

    # 递归展开 \input / \include
    return _expand_inputs(main_text, tex_files, max_depth)


def _expand_inputs(text: str, tex_files: dict[str, str], depth: int) -> str:
    """递归把 \\input{name} / \\include{name} 替换成对应文件的内容"""
    if depth <= 0:
        return text

    def _replace(match: re.Match) -> str:
        target = match.group(1).strip()
        # 试多种 key 形式
        candidates = [target, f"{target}.tex"]
        # 也试 basename
        if "/" in target:
            base = target.rsplit("/", 1)[-1]
            candidates.append(base)
            candidates.append(f"{base}.tex")
        for cand in candidates:
            if cand in tex_files:
                # 递归展开嵌套 \input
                return _expand_inputs(tex_files[cand], tex_files, depth - 1)
        # 找不到：保留原样，让下游忽略
        return match.group(0)

    return _INPUT_PATTERN.sub(_replace, text)


# ---------------------------------------------------------------------------
# 章节切分
# ---------------------------------------------------------------------------

# 常见章节名 → 规范 key 的映射
# 设计原则：覆盖 ML/NLP 顶会（NeurIPS/ICML/ACL/ICLR）论文的常见章节命名习惯
# 历史 gap：实跑 LayerSkip(arxiv:2404.16710) 时发现 "Proposed Solution"、
# "Motivation"、"Ablation Studies" 等常见章节没被识别 → 全部加入
_SECTION_NAME_MAP = {
    # canonical: [aliases (lowercased, 去标点后)]
    "abstract": ["abstract"],

    "introduction": [
        "introduction", "intro", "background", "preliminaries",
        "motivation", "background and motivation", "problem statement",
    ],

    "related_work": [
        "related work", "related works", "prior work", "literature review",
        "previous work",
    ],

    "method": [
        "method", "methods", "methodology", "approach", "our method",
        "proposed method", "approach overview", "model",
        "proposed solution", "proposed approach", "framework",
        "design", "architecture", "our approach", "our framework",
        "system design", "model architecture", "training",
    ],

    "experiments": [
        "experiments", "experiment", "evaluation", "results",
        "experimental results", "experimental setup", "experimental setting",
        "ablation", "ablation study", "ablation studies",
        "empirical evaluation", "empirical results", "empirical study",
        "benchmarks", "benchmark results",
    ],

    "conclusion": [
        "conclusion", "conclusions", "discussion", "summary",
        "discussion and conclusion", "limitations", "limitation",
        "future work", "broader impact",
    ],
}


def _extract_abstract(latex_text: str) -> str:
    """
    依次尝试 3 种 LaTeX 论文常见的 abstract 写法：

    1. \\begin{abstract}...\\end{abstract}             (NeurIPS/ICML/ACL 标准)
    2. \\begin{abstract*}...\\end{abstract*}           (少数会议)
    3. \\abstract{...}                                  (自定义宏，少见)

    Returns: 抽到的 abstract 文本（去掉前后空白），未匹配返空字符串
    """
    # 模式 1+2: \begin{abstract[*]}...\end{abstract[*]}
    m = re.search(
        r"\\begin\{abstract\*?\}(.*?)\\end\{abstract\*?\}",
        latex_text,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()

    # 模式 3: \abstract{...}（balanced braces 比 .* 更稳）
    abstract_macro = re.search(r"\\abstract\s*\{", latex_text)
    if abstract_macro:
        start = abstract_macro.end()
        depth = 1
        i = start
        while i < len(latex_text) and depth > 0:
            ch = latex_text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return latex_text[start:i - 1].strip()

    return ""


def split_sections(latex_text: str) -> dict[str, str]:
    """
    把 LaTeX 全文按 \\section / \\section* 切成章节字典。

    返回字典 keys 是规范化名（abstract/introduction/method/experiments/conclusion）。
    未识别的章节归入 "other_<原标题>"。

    特殊处理 abstract：依次尝试 3 种常见写法。
    """
    sections: dict[str, str] = {}

    # ① 抽 abstract——按优先级尝试 3 种 LaTeX 写法
    abstract_text = _extract_abstract(latex_text)
    if abstract_text:
        sections["abstract"] = abstract_text

    # ② 用 \section{...} / \section*{...} 切分正文
    # 找出所有 section 标记的位置 + 标题
    section_pattern = re.compile(
        r"\\section\*?\{([^}]+)\}",
        re.IGNORECASE,
    )

    matches = list(section_pattern.finditer(latex_text))
    if not matches:
        # 无章节标记：把全文塞进 "method"（兜底）
        # 仅在没抽到 abstract 时这样做，避免与 abstract 重叠
        if "abstract" not in sections and latex_text.strip():
            sections["method"] = latex_text.strip()
        return sections

    # 切分：每个 section 范围从当前 \section 到下一个 \section 之前
    for i, match in enumerate(matches):
        title_raw = match.group(1).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(latex_text)
        body = latex_text[body_start:body_end].strip()

        # 规范化 key
        canonical_key = _canonicalize_section_name(title_raw)

        # 重复 key 拼接（如多个章节都映射到 "method"）
        if canonical_key in sections:
            sections[canonical_key] = sections[canonical_key] + "\n\n" + body
        else:
            sections[canonical_key] = body

    return sections


def _canonicalize_section_name(title: str) -> str:
    """
    把 \\section{...} 的标题映射到规范 key。

    映射规则：
    - 大小写无关 + 去标点 + 子串匹配
    - 比如 "Experimental Results" → "experiments"
    - 未匹配返 "other_<原标题大写>"
    """
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", "", title).strip().lower()

    for canonical, aliases in _SECTION_NAME_MAP.items():
        for alias in aliases:
            if alias in cleaned or cleaned in alias:
                return canonical

    # 没匹配上：保留原标题作为 key（去空格）
    return f"other_{title.strip()}"


# ---------------------------------------------------------------------------
# 工具：拼"给 LLM 看的精简全文"
# ---------------------------------------------------------------------------

def render_for_llm(
    source: LatexSource,
    sections_to_include: list[str] | None = None,
    max_chars_per_section: int = 8000,
) -> str:
    """
    把 LatexSource 渲染为给 LLM 抽取的紧凑文本。

    Args:
        source: fetch_arxiv_latex 的输出
        sections_to_include: 默认只取 method + experiments + conclusion
                             这是抽五元组最有用的三段
        max_chars_per_section: 每段截断字符数（防 token 超限）

    Returns:
        markdown-style 字符串，按 section 标题组织
    """
    if not source.has_full_text:
        return ""

    if sections_to_include is None:
        sections_to_include = ["method", "experiments", "conclusion"]

    parts = []
    for sec_name in sections_to_include:
        text = source.section(sec_name)
        if not text:
            continue
        if len(text) > max_chars_per_section:
            text = text[:max_chars_per_section] + "\n\n[...truncated...]"
        parts.append(f"## {sec_name.upper()}\n\n{text}")

    return "\n\n".join(parts)

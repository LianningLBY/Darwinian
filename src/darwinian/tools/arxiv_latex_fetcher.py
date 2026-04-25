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
    从 arxiv e-print 返回的字节流里抽出主 .tex 文件。

    arxiv 返回类型多变：
    - 多文件: .tar.gz （最常见，多 .tex + 图）
    - 单文件: .tex.gz （直接是压缩的单 tex）
    - PDF-only: %PDF- 开头（没 LaTeX 源码的论文）

    返回：主 .tex 文件的字符串内容；未找到返空。
    """
    if not content:
        return ""

    # 检测 PDF（arxiv 有时只给 PDF）
    if content[:5] == b"%PDF-":
        return ""

    # 尝试 tar.gz
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
            return _find_main_tex_in_tar(tar)
    except (tarfile.ReadError, EOFError):
        pass

    # 尝试 tar（不压缩）
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as tar:
            return _find_main_tex_in_tar(tar)
    except (tarfile.ReadError, EOFError):
        pass

    # 尝试 gzip 单文件 (.tex.gz)
    try:
        import gzip
        decompressed = gzip.decompress(content)
        # 看是不是 LaTeX
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


def _find_main_tex_in_tar(tar: tarfile.TarFile) -> str:
    """
    在 tar 包里找主 .tex 文件——优先含 \\documentclass 或 \\begin{document}。

    策略：
    1. 收集所有 .tex 文件
    2. 优先返回含 \\documentclass 的（顶级文档）
    3. 如果只有一个 .tex 文件，直接返
    4. 如果多个但都不含 \\documentclass，返第一个（兜底）
    """
    candidates: list[tuple[str, str]] = []   # (name, content)

    for member in tar.getmembers():
        if not member.isfile():
            continue
        name = member.name.lower()
        if not name.endswith(".tex"):
            continue
        try:
            f = tar.extractfile(member)
            if f is None:
                continue
            text = f.read().decode("utf-8", errors="ignore")
            candidates.append((member.name, text))
        except Exception:
            continue

    if not candidates:
        return ""

    # 优先含 documentclass 的
    for _, text in candidates:
        if "\\documentclass" in text:
            return text

    # 没有 documentclass 但有 \begin{document}
    for _, text in candidates:
        if "\\begin{document}" in text:
            return text

    # 兜底：第一个 .tex
    return candidates[0][1]


# ---------------------------------------------------------------------------
# 章节切分
# ---------------------------------------------------------------------------

# 常见章节名 → 规范 key 的映射
_SECTION_NAME_MAP = {
    # canonical: [aliases (lowercased)]
    "abstract": ["abstract"],
    "introduction": ["introduction", "intro", "background"],
    "related_work": ["related work", "related works", "prior work", "literature review"],
    "method": ["method", "methods", "methodology", "approach", "our method",
               "proposed method", "approach overview", "model"],
    "experiments": ["experiments", "experiment", "evaluation", "results",
                    "experimental results", "experimental setup", "experimental setting"],
    "conclusion": ["conclusion", "conclusions", "discussion", "summary",
                   "discussion and conclusion"],
}


def split_sections(latex_text: str) -> dict[str, str]:
    """
    把 LaTeX 全文按 \\section / \\section* 切成章节字典。

    返回字典 keys 是规范化名（abstract/introduction/method/experiments/conclusion）。
    未识别的章节归入 "other_<原标题>"。

    特殊处理 abstract：抽 \\begin{abstract}...\\end{abstract} 块。
    """
    sections: dict[str, str] = {}

    # ① 抽 abstract（\begin{abstract}...\end{abstract}）
    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        latex_text,
        re.DOTALL,
    )
    if abstract_match:
        sections["abstract"] = abstract_match.group(1).strip()

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

"""
arxiv_latex_fetcher 测试

不打真实 arxiv API：mock httpx 和 tar 文件构造，验证：
- LaTeX 源码下载 + 解压
- 主 .tex 文件识别（\\documentclass / \\begin{document}）
- PDF-only 论文降级
- 章节切分（\\section / \\section* / \\begin{abstract}）
- 章节名规范化映射
- render_for_llm 截断
"""

from __future__ import annotations

import gzip
import io
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from darwinian.tools import arxiv_latex_fetcher as alf
from darwinian.tools.arxiv_latex_fetcher import (
    LatexSource,
    fetch_arxiv_latex,
    render_for_llm,
    split_sections,
    _canonicalize_section_name,
    _extract_main_tex_from_archive,
    _find_main_tex_in_tar,
)


# ---------------------------------------------------------------------------
# 工具：构造假 tar 包
# ---------------------------------------------------------------------------

def _build_targz(files: dict[str, str]) -> bytes:
    """构造一个 tar.gz 字节流，包含若干 .tex 文件"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _canonicalize_section_name
# ---------------------------------------------------------------------------

class TestCanonicalizeSectionName:
    def test_abstract(self):
        assert _canonicalize_section_name("Abstract") == "abstract"

    def test_introduction_alias(self):
        assert _canonicalize_section_name("Introduction") == "introduction"
        assert _canonicalize_section_name("Background") == "introduction"

    def test_method_alias(self):
        assert _canonicalize_section_name("Methodology") == "method"
        assert _canonicalize_section_name("Our Approach") == "method"
        assert _canonicalize_section_name("Proposed Method") == "method"

    def test_experiments_alias(self):
        assert _canonicalize_section_name("Experiments") == "experiments"
        assert _canonicalize_section_name("Experimental Results") == "experiments"
        assert _canonicalize_section_name("Evaluation") == "experiments"

    def test_conclusion_alias(self):
        assert _canonicalize_section_name("Conclusion") == "conclusion"
        assert _canonicalize_section_name("Discussion") == "conclusion"

    def test_unknown_section_keeps_prefix(self):
        result = _canonicalize_section_name("Custom Section")
        assert result.startswith("other_")
        assert "Custom Section" in result


# ---------------------------------------------------------------------------
# split_sections
# ---------------------------------------------------------------------------

class TestSplitSections:
    def test_extracts_abstract(self):
        latex = r"""
        \begin{abstract}
        We present a novel method for X.
        \end{abstract}
        """
        out = split_sections(latex)
        assert "abstract" in out
        assert "novel method for X" in out["abstract"]

    def test_splits_top_level_sections(self):
        latex = r"""
        \section{Introduction}
        Intro body here.

        \section{Method}
        Our method is...

        \section{Experiments}
        We evaluate on...
        """
        out = split_sections(latex)
        assert "introduction" in out and "Intro body" in out["introduction"]
        assert "method" in out and "Our method" in out["method"]
        assert "experiments" in out and "We evaluate" in out["experiments"]

    def test_handles_section_star(self):
        latex = r"""
        \section*{Method}
        body
        """
        out = split_sections(latex)
        assert "method" in out

    def test_combines_repeat_canonical(self):
        """多个章节都映射到 method 时应拼接"""
        latex = r"""
        \section{Method}
        Method A description

        \section{Approach}
        Approach B description
        """
        out = split_sections(latex)
        assert "method" in out
        assert "Method A" in out["method"]
        assert "Approach B" in out["method"]

    def test_no_sections_falls_back_to_method(self):
        """无 \section 时全文进 method（兜底，但不与 abstract 重叠）"""
        latex = "Some plain text without sections."
        out = split_sections(latex)
        assert out["method"].startswith("Some plain")

    def test_unknown_section_kept_as_other(self):
        latex = r"""
        \section{Custom Heading}
        custom body
        """
        out = split_sections(latex)
        keys_starting_other = [k for k in out if k.startswith("other_")]
        assert len(keys_starting_other) == 1


# ---------------------------------------------------------------------------
# _find_main_tex_in_tar
# ---------------------------------------------------------------------------

class TestFindMainTexInTar:
    def test_single_tex_returned(self):
        targz = _build_targz({"paper.tex": r"\documentclass{article}\begin{document}body\end{document}"})
        with tarfile.open(fileobj=io.BytesIO(targz), mode="r:gz") as tar:
            result = _find_main_tex_in_tar(tar)
        assert "documentclass" in result

    def test_prefers_file_with_documentclass(self):
        """多个 .tex 时优先含 \\documentclass 的"""
        targz = _build_targz({
            "section1.tex": "subsection content without preamble",
            "main.tex": r"\documentclass{article}\begin{document}main\end{document}",
        })
        with tarfile.open(fileobj=io.BytesIO(targz), mode="r:gz") as tar:
            result = _find_main_tex_in_tar(tar)
        assert "documentclass" in result

    def test_no_tex_files_returns_empty(self):
        targz = _build_targz({"figure.png": "binary garbage"})
        with tarfile.open(fileobj=io.BytesIO(targz), mode="r:gz") as tar:
            result = _find_main_tex_in_tar(tar)
        assert result == ""


# ---------------------------------------------------------------------------
# _extract_main_tex_from_archive
# ---------------------------------------------------------------------------

class TestExtractMainTexFromArchive:
    def test_pdf_returns_empty(self):
        """arxiv 偶尔只给 PDF（无 LaTeX 源），应返空"""
        assert _extract_main_tex_from_archive(b"%PDF-1.4\n...", "application/pdf") == ""

    def test_targz_extracted(self):
        targz = _build_targz({"main.tex": r"\documentclass{article}\begin{document}hi\end{document}"})
        text = _extract_main_tex_from_archive(targz, "application/gzip")
        assert "documentclass" in text

    def test_single_gzip_tex(self):
        """有些 arxiv 论文返回单文件 .tex.gz（不是 tar.gz）"""
        raw_tex = r"\documentclass{article}\begin{document}content\end{document}"
        gz = gzip.compress(raw_tex.encode("utf-8"))
        text = _extract_main_tex_from_archive(gz, "application/gzip")
        assert "documentclass" in text

    def test_empty_returns_empty(self):
        assert _extract_main_tex_from_archive(b"", "") == ""

    def test_random_bytes_returns_empty(self):
        """不可识别字节流应返空，不抛"""
        assert _extract_main_tex_from_archive(b"\x00\x01\x02\x03 garbage", "") == ""


# ---------------------------------------------------------------------------
# fetch_arxiv_latex 端到端
# ---------------------------------------------------------------------------

class TestFetchArxivLatex:
    def _mock_client_with_response(self, status, content, content_type="application/gzip"):
        resp = MagicMock()
        resp.status_code = status
        resp.content = content
        resp.headers = {"content-type": content_type}
        if status >= 400:
            resp.raise_for_status = MagicMock(side_effect=Exception(f"HTTP {status}"))
        else:
            resp.raise_for_status = MagicMock()

        c = MagicMock()
        c.__enter__ = MagicMock(return_value=c)
        c.__exit__ = MagicMock(return_value=False)
        c.get = MagicMock(return_value=resp)
        return c

    def test_happy_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("darwinian.tools.semantic_scholar.CACHE_DIR", tmp_path)
        monkeypatch.setattr(alf, "_MIN_INTERVAL", 0.0)

        targz = _build_targz({
            "paper.tex": r"""
\documentclass{article}
\begin{document}
\begin{abstract}
We present X.
\end{abstract}
\section{Method}
Our method.
\section{Experiments}
We test on Y.
\end{document}
            """.strip(),
        })
        client = self._mock_client_with_response(200, targz)
        with patch("httpx.Client", return_value=client):
            src = fetch_arxiv_latex("2510.23766")

        assert src is not None
        assert src.has_full_text is True
        assert src.arxiv_id == "2510.23766"
        assert "X" in src.section("abstract")
        assert "Our method" in src.section("method")
        assert "test on Y" in src.section("experiments")

    def test_strips_arxiv_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr("darwinian.tools.semantic_scholar.CACHE_DIR", tmp_path)
        monkeypatch.setattr(alf, "_MIN_INTERVAL", 0.0)
        targz = _build_targz({"x.tex": r"\documentclass{article}\begin{document}x\end{document}"})
        client = self._mock_client_with_response(200, targz)
        with patch("httpx.Client", return_value=client):
            src = fetch_arxiv_latex("arxiv:2510.23766")
        assert src is not None
        assert src.arxiv_id == "2510.23766"

    def test_pdf_only_marks_no_full_text(self, tmp_path, monkeypatch):
        """arxiv 返 PDF 时应返 has_full_text=False，不抛"""
        monkeypatch.setattr("darwinian.tools.semantic_scholar.CACHE_DIR", tmp_path)
        monkeypatch.setattr(alf, "_MIN_INTERVAL", 0.0)
        client = self._mock_client_with_response(200, b"%PDF-1.4\nfake")
        with patch("httpx.Client", return_value=client):
            src = fetch_arxiv_latex("ancient.paper")
        assert src is not None
        assert src.has_full_text is False
        assert src.main_tex == ""

    def test_http_404_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("darwinian.tools.semantic_scholar.CACHE_DIR", tmp_path)
        monkeypatch.setattr(alf, "_MIN_INTERVAL", 0.0)
        client = self._mock_client_with_response(404, b"Not found")
        with patch("httpx.Client", return_value=client):
            src = fetch_arxiv_latex("nonexistent.id")
        assert src is None

    def test_empty_id_returns_none(self):
        assert fetch_arxiv_latex("") is None
        assert fetch_arxiv_latex("   ") is None
        assert fetch_arxiv_latex("arxiv:") is None

    def test_cache_hit_skips_http(self, tmp_path, monkeypatch):
        from darwinian.tools.semantic_scholar import _cache_set, _cache_key
        monkeypatch.setattr("darwinian.tools.semantic_scholar.CACHE_DIR", tmp_path)
        monkeypatch.setattr(alf, "_MIN_INTERVAL", 0.0)

        # 预存
        cached_obj = LatexSource(arxiv_id="abc", main_tex="cached", has_full_text=True)
        _cache_set(_cache_key("/arxiv/e-print", {"id": "abc"}), cached_obj)
        with patch("httpx.Client") as mc:
            src = fetch_arxiv_latex("abc")
        assert src.main_tex == "cached"
        assert mc.call_count == 0


# ---------------------------------------------------------------------------
# render_for_llm
# ---------------------------------------------------------------------------

class TestRenderForLlm:
    def _src_with_sections(self):
        return LatexSource(
            arxiv_id="x",
            main_tex="...",
            sections={
                "abstract": "abs",
                "introduction": "intro",
                "method": "meth body",
                "experiments": "exp body",
                "conclusion": "concl body",
            },
            has_full_text=True,
        )

    def test_default_includes_method_exp_conclusion(self):
        out = render_for_llm(self._src_with_sections())
        assert "## METHOD" in out and "meth body" in out
        assert "## EXPERIMENTS" in out and "exp body" in out
        assert "## CONCLUSION" in out and "concl body" in out
        # introduction 默认不在
        assert "intro" not in out

    def test_custom_section_list(self):
        out = render_for_llm(self._src_with_sections(),
                              sections_to_include=["abstract", "method"])
        assert "abs" in out
        assert "meth body" in out
        assert "exp body" not in out

    def test_truncation(self):
        big_text = "x" * 20000
        src = LatexSource(arxiv_id="x", main_tex="...",
                          sections={"method": big_text}, has_full_text=True)
        out = render_for_llm(src, sections_to_include=["method"], max_chars_per_section=5000)
        assert "[...truncated...]" in out

    def test_no_full_text_returns_empty(self):
        src = LatexSource(arxiv_id="x", has_full_text=False)
        assert render_for_llm(src) == ""

    def test_missing_section_skipped(self):
        src = LatexSource(arxiv_id="x", sections={"method": "m"}, has_full_text=True)
        out = render_for_llm(src, sections_to_include=["method", "experiments"])
        assert "## METHOD" in out
        assert "## EXPERIMENTS" not in out

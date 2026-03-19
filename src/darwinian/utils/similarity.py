"""
语义相似度工具

用于 failed_ledger 的余弦相似度去重。
使用轻量级 TF-IDF 向量化（无需外部 embedding API），确保离线可用。
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter


def get_text_embedding(text: str) -> list[float]:
    """
    将文本转换为 TF-IDF 风格的稀疏向量（哈希技巧，固定 512 维）。

    设计说明：
    - 不依赖外部 embedding 服务，保证离线可用
    - 哈希技巧将词映射到固定维度，避免词表构建
    - 对于去重目的，此精度已足够
    """
    tokens = _tokenize(text)
    if not tokens:
        return [0.0] * 512

    term_freq = Counter(tokens)
    total = sum(term_freq.values())

    vector = [0.0] * 512
    for term, count in term_freq.items():
        # 哈希到 [0, 512) 范围
        idx = int(hashlib.md5(term.encode()).hexdigest(), 16) % 512
        vector[idx] += count / total  # 归一化频率

    # L2 归一化
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]

    return vector


def compute_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    计算两个向量的余弦相似度。

    Args:
        vec_a: 向量 A
        vec_b: 向量 B（长度可与 A 不同，取较短者）

    Returns:
        相似度值，范围 [0, 1]
    """
    if not vec_a or not vec_b:
        return 0.0

    min_len = min(len(vec_a), len(vec_b))
    dot = sum(a * b for a, b in zip(vec_a[:min_len], vec_b[:min_len]))

    norm_a = math.sqrt(sum(v * v for v in vec_a))
    norm_b = math.sqrt(sum(v * v for v in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _tokenize(text: str) -> list[str]:
    """简单的词元化：转小写、按非字母数字字符分割"""
    import re
    text = text.lower()
    tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text)
    # 过滤停用词（英文）
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "to", "of", "in", "for", "on", "with", "that", "this", "it",
        "we", "our", "and", "or", "but", "not", "from", "by", "as",
    }
    return [t for t in tokens if t not in stopwords and len(t) > 1]

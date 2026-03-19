"""
相似度工具测试

验证余弦相似度计算和 TF-IDF 嵌入向量的正确性。
"""

import pytest
from darwinian.utils.similarity import get_text_embedding, compute_cosine_similarity


class TestGetTextEmbedding:
    def test_returns_fixed_length(self):
        vec = get_text_embedding("test sentence")
        assert len(vec) == 512

    def test_empty_text_returns_zeros(self):
        vec = get_text_embedding("")
        assert all(v == 0.0 for v in vec)

    def test_normalized(self):
        import math
        vec = get_text_embedding("some text about machine learning")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_different_texts_different_vectors(self):
        vec1 = get_text_embedding("neural network optimization")
        vec2 = get_text_embedding("cooking recipe pasta")
        similarity = compute_cosine_similarity(vec1, vec2)
        # 完全不相关的文本相似度应远低于 0.5
        assert similarity < 0.5


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = [1.0, 0.0, 0.0, 0.5]
        sim = compute_cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        sim = compute_cosine_similarity(vec_a, vec_b)
        assert abs(sim) < 1e-6

    def test_empty_vector(self):
        sim = compute_cosine_similarity([], [1.0, 2.0])
        assert sim == 0.0

    def test_similar_texts(self):
        text1 = "graph neural network node classification"
        text2 = "graph neural network link prediction"
        vec1 = get_text_embedding(text1)
        vec2 = get_text_embedding(text2)
        sim = compute_cosine_similarity(vec1, vec2)
        # 相关文本相似度应较高
        assert sim > 0.3

    def test_result_in_range(self):
        import random
        random.seed(42)
        vec_a = [random.random() for _ in range(512)]
        vec_b = [random.random() for _ in range(512)]
        sim = compute_cosine_similarity(vec_a, vec_b)
        assert 0.0 <= sim <= 1.0

"""
扰动策略库测试

验证所有固定扰动策略的代码生成和执行正确性。
"""

import pytest
import numpy as np
from darwinian.tools.perturbation_strategies import (
    STRATEGY_REGISTRY,
    generate_perturbation_code,
)


class TestStrategyRegistry:
    def test_all_strategies_have_required_fields(self):
        for name, strategy in STRATEGY_REGISTRY.items():
            assert strategy.name, f"{name} 缺少 name"
            assert strategy.description, f"{name} 缺少 description"
            assert strategy.code_template, f"{name} 缺少 code_template"

    def test_minimum_strategy_count(self):
        assert len(STRATEGY_REGISTRY) >= 5, "策略库至少需要 5 个策略"

    def test_expected_strategies_exist(self):
        expected = ["gaussian_noise", "label_flip", "feature_mask", "ood_distribution_shift"]
        for name in expected:
            assert name in STRATEGY_REGISTRY, f"缺少策略: {name}"


class TestStrategyExecution:
    """通过 exec 执行每个策略的代码模板，验证其正确性"""

    def _run_strategy(self, strategy_name: str, X, y):
        strategy = STRATEGY_REGISTRY[strategy_name]
        namespace = {"np": np}
        exec(strategy.code_template, namespace)
        # 获取函数名
        func_name = [k for k in namespace if k.startswith("apply_")][0]
        func = namespace[func_name]
        return func(X.copy(), y.copy())

    def setup_method(self):
        rng = np.random.RandomState(42)
        self.X = rng.randn(100, 10).astype(np.float64)
        self.y = rng.randint(0, 2, size=100)

    def test_gaussian_noise_preserves_shape(self):
        X_out, y_out = self._run_strategy("gaussian_noise", self.X, self.y)
        assert X_out.shape == self.X.shape
        assert y_out.shape == self.y.shape

    def test_gaussian_noise_changes_values(self):
        X_out, _ = self._run_strategy("gaussian_noise", self.X, self.y)
        assert not np.allclose(X_out, self.X)

    def test_label_flip_preserves_shape(self):
        _, y_out = self._run_strategy("label_flip", self.X, self.y)
        assert y_out.shape == self.y.shape

    def test_label_flip_changes_some_labels(self):
        _, y_out = self._run_strategy("label_flip", self.X, self.y)
        n_changed = np.sum(y_out != self.y)
        assert n_changed > 0, "label_flip 应该改变一些标签"
        assert n_changed < len(self.y), "label_flip 不应该改变所有标签"

    def test_feature_mask_zeros_some_columns(self):
        X_out, _ = self._run_strategy("feature_mask", self.X, self.y)
        # 某些列应该全为 0
        zero_cols = np.where(np.all(X_out == 0, axis=0))[0]
        assert len(zero_cols) > 0

    def test_ood_shift_changes_distribution(self):
        X_out, _ = self._run_strategy("ood_distribution_shift", self.X, self.y)
        assert not np.allclose(X_out, self.X)
        # 均值应该偏移
        assert abs(X_out.mean() - self.X.mean()) > 0.1


class TestGeneratePerturbationCode:
    def test_generated_code_contains_strategy_names(self):
        strategies = [STRATEGY_REGISTRY["gaussian_noise"], STRATEGY_REGISTRY["label_flip"]]
        code = generate_perturbation_code(strategies, dataset_schema={})
        assert "gaussian_noise" in code
        assert "label_flip" in code

    def test_generated_code_is_string(self):
        strategies = [STRATEGY_REGISTRY["feature_mask"]]
        code = generate_perturbation_code(strategies, dataset_schema={})
        assert isinstance(code, str)
        assert len(code) > 100

    def test_generated_code_contains_apply_calls(self):
        strategies = [STRATEGY_REGISTRY["gaussian_noise"]]
        code = generate_perturbation_code(strategies, dataset_schema={})
        assert "apply_gaussian_noise" in code

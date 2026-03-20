"""
鲁棒性测试固定扰动策略库

Agent 6 从此固定库中「选择+组合」，
覆盖数据质量、分布偏移、对抗扰动、稀疏场景等多个维度，
确保鲁棒性测试的可量化性和可复现性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerturbationStrategy:
    name: str
    description: str
    code_template: str  # Python 代码模板，包含 {X} 和 {y} 占位符


# ---------------------------------------------------------------------------
# 策略定义
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, PerturbationStrategy] = {

    "gaussian_noise": PerturbationStrategy(
        name="gaussian_noise",
        description="向特征矩阵添加高斯噪声（σ=0.1 * 特征标准差），模拟传感器噪声",
        code_template="""\
import numpy as np
def apply_gaussian_noise(X, y, noise_level=0.1, seed=42):
    rng = np.random.RandomState(seed)
    std = np.std(X, axis=0, keepdims=True) + 1e-8
    X_noisy = X + rng.normal(0, noise_level * std, X.shape)
    return X_noisy, y
""",
    ),

    "label_flip": PerturbationStrategy(
        name="label_flip",
        description="随机翻转 20% 的标签，模拟标注错误场景",
        code_template="""\
import numpy as np
def apply_label_flip(X, y, flip_rate=0.2, seed=42):
    rng = np.random.RandomState(seed)
    y_noisy = y.copy()
    n_flip = int(len(y) * flip_rate)
    flip_indices = rng.choice(len(y), n_flip, replace=False)
    unique_labels = np.unique(y)
    for idx in flip_indices:
        other_labels = unique_labels[unique_labels != y[idx]]
        if len(other_labels) > 0:
            y_noisy[idx] = rng.choice(other_labels)
    return X, y_noisy
""",
    ),

    "feature_mask": PerturbationStrategy(
        name="feature_mask",
        description="随机遮掩 30% 的特征维度（置零），模拟缺失数据",
        code_template="""\
import numpy as np
def apply_feature_mask(X, y, mask_rate=0.3, seed=42):
    rng = np.random.RandomState(seed)
    X_masked = X.copy()
    n_features = X.shape[1]
    n_mask = int(n_features * mask_rate)
    mask_cols = rng.choice(n_features, n_mask, replace=False)
    X_masked[:, mask_cols] = 0.0
    return X_masked, y
""",
    ),

    "ood_distribution_shift": PerturbationStrategy(
        name="ood_distribution_shift",
        description="对测试集进行分布偏移（均值偏移 + 方差缩放），模拟域迁移",
        code_template="""\
import numpy as np
def apply_ood_shift(X, y, mean_shift=2.0, scale=0.5, seed=42):
    rng = np.random.RandomState(seed)
    X_shifted = X * scale + mean_shift
    return X_shifted, y
""",
    ),

    "adversarial_perturbation": PerturbationStrategy(
        name="adversarial_perturbation",
        description="基于梯度方向的对抗性扰动（FGSM 风格，ε=0.05），模拟对抗攻击",
        code_template="""\
import numpy as np
def apply_adversarial_perturbation(X, y, epsilon=0.05, seed=42):
    # 简化版：沿最大方差方向扰动
    rng = np.random.RandomState(seed)
    cov = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 最大特征向量方向
    adv_dir = eigenvectors[:, -1]
    signs = rng.choice([-1, 1], size=len(X))
    X_adv = X + epsilon * np.outer(signs, adv_dir)
    return X_adv, y
""",
    ),

    "temporal_reversal": PerturbationStrategy(
        name="temporal_reversal",
        description="将时序数据逆序排列，测试方法对时序依赖的鲁棒性",
        code_template="""\
import numpy as np
def apply_temporal_reversal(X, y):
    return X[::-1].copy(), y[::-1].copy()
""",
    ),

    "class_imbalance": PerturbationStrategy(
        name="class_imbalance",
        description="对少数类进行欠采样，将类别比例调整为 10:1，模拟极度不平衡场景",
        code_template="""\
import numpy as np
def apply_class_imbalance(X, y, majority_ratio=10, seed=42):
    rng = np.random.RandomState(seed)
    unique, counts = np.unique(y, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]
    minority_idx = np.where(y == minority_class)[0]
    majority_idx = np.where(y == majority_class)[0]
    n_keep = min(len(minority_idx) * majority_ratio, len(majority_idx))
    keep_majority = rng.choice(majority_idx, n_keep, replace=False)
    keep_idx = np.concatenate([minority_idx, keep_majority])
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]
""",
    ),
}


def generate_perturbation_code(
    strategies: list[PerturbationStrategy],
    dataset_schema: dict[str, Any],
) -> str:
    """
    组合多个扰动策略，生成完整的扰动+测试代码。

    Args:
        strategies: 选中的扰动策略列表
        dataset_schema: 数据集 Schema，用于生成正确的数据加载代码

    Returns:
        完整可执行的 Python 代码字符串
    """
    strategy_functions = "\n\n".join(s.code_template for s in strategies)
    strategy_calls = "\n".join(
        f"    X_test, y_test = {_get_function_name(s)}(X_test, y_test)"
        for s in strategies
    )
    strategy_names = ", ".join(s.name for s in strategies)

    return f"""\
# ============================================================
# Darwinian 鲁棒性测试 — 扰动策略: {strategy_names}
# ============================================================
import numpy as np
import json
import sys

# 扰动函数定义
{strategy_functions}

# 数据加载（复用 dataset_loader.py 的逻辑）
try:
    from dataset_loader import load_data
    X_train, X_test, y_train, y_test = load_data()
except ImportError:
    print(json.dumps({{"error": "dataset_loader.py 中未定义 load_data() 函数"}}))
    sys.exit(1)

# 应用扰动
print(f"原始测试集 shape: {{X_test.shape}}")
{strategy_calls}
print(f"扰动后测试集 shape: {{X_test.shape}}")

# 加载 proposed 模型并在扰动数据上测试
try:
    from proposed import train_model, evaluate_model
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    print(json.dumps({{"model": "proposed_robustness", "metrics": metrics, "strategies": "{strategy_names}"}}))
except Exception as e:
    print(json.dumps({{"model": "proposed_robustness", "error": str(e)}}))
    sys.exit(1)
"""


def _get_function_name(strategy: PerturbationStrategy) -> str:
    """从代码模板中提取函数名"""
    for line in strategy.code_template.splitlines():
        if line.startswith("def apply_"):
            return line.split("(")[0].replace("def ", "")
    return f"apply_{strategy.name}"

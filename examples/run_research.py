"""
使用示例：启动一次完整的自动化科研流程

用法：
    # 设置环境变量
    export MINIMAX_API_KEY=your_key_here

    # 运行
    python examples/run_research.py
"""

import os
from dotenv import load_dotenv

from darwinian.state import ResearchState
from darwinian.graphs.main_graph import build_main_graph
from darwinian.llms import ChatMiniMax

load_dotenv()


def main():
    # 初始化 LLM - 使用 MiniMax M2.7
    llm = ChatMiniMax(
        model="MiniMax-M2.7",
        api_key=os.environ["MINIMAX_API_KEY"],
        max_tokens=8192,
    )

    # 构建图
    graph = build_main_graph(llm)

    # 定义初始状态
    initial_state = ResearchState(
        research_direction="图神经网络在分子属性预测中的应用",
        dataset_schema={
            "type": "molecular_graph",
            "task": "binary_classification",
            "n_samples": 10000,
            "features": "atom_features (133-dim), bond_features (14-dim)",
            "dataset_name": "BBBP",
            "metric": "ROC-AUC",
        },
    )

    print("🚀 启动 Darwinian 自动化科研系统...")
    print(f"研究方向：{initial_state.research_direction}")
    print("-" * 60)

    # 执行（支持 LangGraph checkpointing）
    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("📊 最终状态摘要：")
    print(f"  - 外层循环次数：{final_state['outer_loop_count']}")
    print(f"  - 失败记录数：{len(final_state['failed_ledger'])}")
    print(f"  - 最终裁决：{final_state['final_verdict']}")
    if final_state.get('publish_matrix'):
        matrix = final_state['publish_matrix']
        print(f"  - 发表矩阵: 新颖性={matrix.novelty_passed}, "
              f"基准提升={matrix.baseline_improved}, "
              f"鲁棒性={matrix.robustness_passed}, "
              f"可解释性={matrix.explainability_generated}")


if __name__ == "__main__":
    main()

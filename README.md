# Decision Tree to Neural Network

这个项目用纯 Python 实现截图里的 **Recursive Tree-to-NN Parsing** 思路：把二叉决策树转换成一个稀疏神经网络，并评估生成的 neural network 与原始 decision tree 的预测是否一致。

## 功能

- 用 `DecisionNode` / `Leaf` 定义输入 decision tree。
- 为每个内部节点创建 true / false 条件神经元。
- 用 ReLU 路径神经元表达 root-to-leaf 路径上的逻辑 AND。
- 用 clipped ReLU 输出层表达同类别路径的逻辑 OR。
- 提供随机样本、阈值边界样本、离散网格评估，输出一致率和 mismatch 明细。

## 一个重要实现细节

截图中的条件神经元写作：

```text
g_t(v) = sigma(x_j - tau_v)
g_f(v) = sigma(tau_v - x_j)
```

如果同一个 step 函数在 `0` 处同时让两边为真，样本刚好等于阈值时 true / false 分支会同时激活。为了和常见 decision tree 语义完全一致，本项目显式采用：

```text
true branch:  x_j >= threshold
false branch: x_j <  threshold
```

因此在阈值相等的边界上也可以做到精确一致。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m dt_to_nn.demo
```

也可以运行：

```bash
dt-to-nn-demo
```

测试：

```bash
python3 -m unittest discover -s tests
```

论文式 evaluation：

```bash
.venv/bin/python -m examples.paper_style_evaluation
```

严格按照 Editable XAI Eq. (1)-(5) 的 smooth-sigmoid 版本及一致率：

```bash
.venv/bin/python -m examples.evaluate_paper_exact
```

这个版本与原来的硬阈值版本并存：

- `convert_tree_to_network()` 使用硬 branch indicator，目标是与 DT 严格预测等价。
- `convert_tree_to_paper_exact_network()` 第一层严格使用论文的
  `sigmoid(+x_j-tau)` / `sigmoid(-x_j+tau)`，权重不额外乘 temperature。

论文公式版在参数和激活函数上忠实于论文，但 smooth sigmoid 会产生连续值，
所以不保证预测标签 100% 一致，也不保证输出是精确 one-hot。专用 evaluation
会分别报告随机样本、threshold probes 和合并样本上的预测一致率与输出误差。

Direct-path parser 对比：

```bash
.venv/bin/python -m examples.compare_direct_path
```

生成 decision tree 和 neural network 的 SVG 可视化：

```bash
.venv/bin/python -m examples.visualize_demo
```

Performance feasibility baselines:

```bash
.venv/bin/python -m examples.run_performance_experiment \
  --datasets diabetes breast_cancer \
  --settings single_tree \
  --models source_tree coexplain_soft same_arch_random djinn path_expansion mlp \
  --alphas 5 20 \
  --split-seeds 0
```

结果会写到 `results/performance_feasibility/`，包含 per-epoch loss、
per-run summary、model size、alpha sensitivity 和 validation-selected alpha 表。

Full grid entry point:

```bash
.venv/bin/python -m examples.run_all_performance
```

默认 full grid 包含 4 个数据集、single-tree / 10-tree ensemble、5 个 split 和
alpha grid `[0.5, 1, 2, 5, 10, 20, 50, 100]`，并使用 `--stream-results`
持续刷新 CSV。California Housing 默认抽样到 2500 条以便本地 feasibility run；
若要使用 sklearn 全量数据，可运行 `examples.run_performance_experiment` 并加
`--no-dataset-cap`。

## 使用示例

```python
from dt_to_nn import (
    DecisionNode,
    Leaf,
    convert_tree_to_network,
    evaluate_equivalence,
    random_samples,
    threshold_probe_samples,
)

tree = DecisionNode(
    feature_index=0,
    threshold=0.5,
    true_child=DecisionNode(
        feature_index=1,
        threshold=1.0,
        true_child=Leaf("A"),
        false_child=Leaf("B"),
    ),
    false_child=Leaf("C"),
)

network = convert_tree_to_network(tree)

samples = random_samples(2, 1000, low=-2.0, high=2.0, seed=42)
samples.extend(threshold_probe_samples(tree))

result = evaluate_equivalence(tree, network, samples)
print(result.to_dict())
```

返回结果里的重点字段：

- `prediction_agreement`: 决策树预测标签与神经网络预测标签的一致率。
- `output_vector_agreement`: 神经网络输出是否等于 tree label 的 one-hot 向量。
- `max_output_error`: 输出向量最大误差。
- `mismatches`: 前几个不一致样本，便于定位问题。

## 等价性说明

在这个实现中，条件神经元输出二值 branch indicator；路径神经元 `ReLU(a + b - 1)` 对两个二值输入实现逻辑 AND；输出神经元 `clipped_relu(sum(paths))` 对同类路径实现逻辑 OR。所以在相同阈值分支语义下，转换出的网络与输入树在结构和预测上等价。

有限样本 evaluation 不能单独证明连续空间上的所有点都一致，但它可以：

- 检查实际数据集上的一致程度。
- 检查阈值相等和阈值两侧的边界行为。
- 对离散特征空间做完整网格枚举。

## 参考论文的 Evaluation

项目额外提供 [docs/evaluation_protocol.md](/Users/amywang/Documents/DT to NN/docs/evaluation_protocol.md)，把两篇论文里的要求拆成两层：

1. **Editable XAI 算法正确性**：检查 parsed neural network 是否严格复现输入 decision tree。
2. **DJINN-style 训练表现**：参考 `2.pdf` 的 fixed five-fold 80/20 split、accuracy/precision/recall、training loss、random dense/sparse baseline，并加入 Editable XAI 里的 threshold-only enhancement 和 zero padding 后 topology enhancement。
3. **Direct path 结构改进**：见 [docs/direct_path_comparison.md](/Users/amywang/Documents/DT to NN/docs/direct_path_comparison.md)，每条 root-to-leaf path 直接建一个 path neuron，比较它和原递归 AND parser 的结构、严格等价性和训练表现。

当前 enhanced evaluation 优先使用 sklearn 做 split/metrics，使用 PyTorch 做 cross-entropy backpropagation。如果没有安装这些可选依赖，会自动退回到纯 NumPy evaluation。

注意：zero padding 后继续训练的目标是提高对数据标签的预测表现；训练后网络参数被更新，因此不再保证和原始 tree 完全一致。这和 Editable XAI 的 Enhance 阶段一致：先由 tree 得到可训练的 NN，再通过额外容量和梯度更新提升性能。

## 项目结构

```text
dt_to_nn/
  tree.py        # 决策树数据结构和预测
  network.py     # 稀疏神经网络图和前向计算
  converter.py   # Tree-to-NN 转换算法
  paper_exact_converter.py  # 严格按照论文 smooth-sigmoid 公式
  paper_exact_evaluation.py # 论文公式版 DT/NN 一致率
  performance_baselines.py  # soft parser / DJINN / EntropyNet-like performance baselines
  direct_path_converter.py
  evaluation.py  # 一致性评估工具
  visualization.py
  trainable.py   # 可微 parsed NN 和 zero-padding 训练
  torch_trainable.py
  paper_evaluation.py
  demo.py        # 示例
tests/
  test_equivalence.py
```

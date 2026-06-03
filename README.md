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

当前 enhanced evaluation 优先使用 sklearn 做 split/metrics，使用 PyTorch 做 cross-entropy backpropagation。如果没有安装这些可选依赖，会自动退回到纯 NumPy evaluation。

注意：zero padding 后继续训练的目标是提高对数据标签的预测表现；训练后网络参数被更新，因此不再保证和原始 tree 完全一致。这和 Editable XAI 的 Enhance 阶段一致：先由 tree 得到可训练的 NN，再通过额外容量和梯度更新提升性能。

## 项目结构

```text
dt_to_nn/
  tree.py        # 决策树数据结构和预测
  network.py     # 稀疏神经网络图和前向计算
  converter.py   # Tree-to-NN 转换算法
  evaluation.py  # 一致性评估工具
  trainable.py   # 可微 parsed NN 和 zero-padding 训练
  torch_trainable.py
  paper_evaluation.py
  demo.py        # 示例
tests/
  test_equivalence.py
```

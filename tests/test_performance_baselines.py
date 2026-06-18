import unittest

import numpy as np

try:
    import torch
    from sklearn.tree import DecisionTreeClassifier

    from dt_to_nn.performance_baselines import (
        DJINNLikeNetwork,
        SoftTreeNetwork,
        choose_mlp_width,
        count_layers,
        count_neurons,
        extract_sklearn_paths,
    )

    BASELINES_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency guard
    BASELINES_IMPORT_ERROR = exc


@unittest.skipIf(BASELINES_IMPORT_ERROR is not None, f"optional baselines unavailable: {BASELINES_IMPORT_ERROR}")
class PerformanceBaselineTest(unittest.TestCase):
    def _tiny_classifier(self):
        x = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [0.8, 0.1],
                [0.9, 0.9],
                [0.2, 0.8],
                [0.7, 0.7],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
        tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(x, y)
        return x, tree

    def test_soft_tree_forward_shapes(self):
        x, tree = self._tiny_classifier()
        paths = extract_sklearn_paths(tree, task="classification", output_dim=2)

        editable = SoftTreeNetwork(paths, alpha=5.0, mode="editable", seed=0)
        path_expansion = SoftTreeNetwork(paths, alpha=5.0, mode="path_expansion", seed=0)

        with torch.no_grad():
            editable_output = editable(torch.as_tensor(x))
            path_output = path_expansion(torch.as_tensor(x))

        self.assertEqual(tuple(editable_output.shape), (len(x), 2))
        self.assertEqual(tuple(path_output.shape), (len(x), 2))
        self.assertEqual(count_layers(editable), 3)
        self.assertEqual(count_neurons(editable), len(paths.node_features) + paths.leaf_count + 2)

    def test_djinn_like_forward_shape(self):
        x, tree = self._tiny_classifier()
        model = DJINNLikeNetwork(tree, task="classification", output_dim=2, seed=0)

        with torch.no_grad():
            output = model(torch.as_tensor(x))

        self.assertEqual(tuple(output.shape), (len(x), 2))
        self.assertGreaterEqual(count_layers(model), 2)

    def test_parameter_matched_width_is_close_to_target(self):
        width = choose_mlp_width(n_features=4, output_dim=2, target_params=120)
        params = (4 + 1) * width + (width + 1) * width + (width + 1) * 2

        neighbor_width = width + 1
        neighbor_params = (
            (4 + 1) * neighbor_width
            + (neighbor_width + 1) * neighbor_width
            + (neighbor_width + 1) * 2
        )

        self.assertLessEqual(abs(params - 120), abs(neighbor_params - 120))


if __name__ == "__main__":
    unittest.main()

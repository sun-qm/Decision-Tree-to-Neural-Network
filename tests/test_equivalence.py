import unittest

from dt_to_nn import (
    DecisionNode,
    Leaf,
    TrainableParsedNetwork,
    convert_tree_to_direct_path_network,
    convert_tree_to_network,
    evaluate_equivalence,
    random_samples,
    threshold_probe_samples,
)
from dt_to_nn.trainable import one_hot
from dt_to_nn.visualization import render_network_svg, render_tree_svg

import numpy as np


def demo_tree():
    return DecisionNode(
        feature_index=0,
        threshold=0.5,
        true_child=DecisionNode(
            feature_index=1,
            threshold=1.0,
            true_child=Leaf("A"),
            false_child=Leaf("B"),
        ),
        false_child=DecisionNode(
            feature_index=2,
            threshold=-0.2,
            true_child=Leaf("B"),
            false_child=Leaf("C"),
        ),
    )


class TreeToNNEquivalenceTest(unittest.TestCase):
    def test_random_and_threshold_samples_match_exactly(self):
        tree = demo_tree()
        network = convert_tree_to_network(tree)
        samples = random_samples(3, 500, low=-3.0, high=3.0, seed=7)
        samples.extend(threshold_probe_samples(tree))

        result = evaluate_equivalence(tree, network, samples)

        self.assertTrue(result.is_fully_consistent, result.to_dict())
        self.assertEqual(result.max_output_error, 0.0)

    def test_threshold_equality_uses_true_branch_only(self):
        tree = DecisionNode(
            feature_index=0,
            threshold=1.5,
            true_child=Leaf("true"),
            false_child=Leaf("false"),
        )
        network = convert_tree_to_network(tree)

        self.assertEqual(network.outputs([1.5]), {"false": 0.0, "true": 1.0})
        self.assertEqual(network.predict([1.5]), "true")

    def test_single_leaf_tree_is_supported(self):
        tree = Leaf("only")
        network = convert_tree_to_network(tree)
        result = evaluate_equivalence(tree, network, [[], [1.0, 2.0]])

        self.assertTrue(result.is_fully_consistent, result.to_dict())
        self.assertEqual(network.outputs([99.0]), {"only": 1.0})

    def test_direct_path_network_matches_tree_exactly(self):
        tree = demo_tree()
        network = convert_tree_to_direct_path_network(tree)
        samples = random_samples(3, 300, low=-3.0, high=3.0, seed=13)
        samples.extend(threshold_probe_samples(tree))

        result = evaluate_equivalence(tree, network, samples)

        self.assertTrue(result.is_fully_consistent, result.to_dict())
        self.assertEqual(result.max_output_error, 0.0)

    def test_summary_contains_expected_network_shape(self):
        network = convert_tree_to_network(demo_tree())
        summary = network.summary()

        self.assertEqual(summary["max_depth"], 2)
        self.assertEqual(summary["condition_neurons"], 6)
        self.assertEqual(summary["output_neurons"], 3)

    def test_zero_padded_trainable_network_can_train(self):
        tree = demo_tree()
        model = TrainableParsedNetwork.from_tree(
            tree,
            classes=("A", "B", "C"),
            n_features=3,
            zero_padding_width=2,
            zero_padding_layers=1,
        )
        x = np.array(
            [
                [0.6, 1.2, 0.0],
                [0.6, 0.4, 0.0],
                [0.1, 0.0, -0.5],
                [0.1, 0.0, 0.0],
            ],
            dtype=float,
        )
        y = one_hot(["A", "B", "C", "B"], ("A", "B", "C"))
        before = model.loss(x, y)
        history = model.fit(x, y, epochs=5, learning_rate=0.01, batch_size=2, seed=0)
        after = model.loss(x, y)

        self.assertTrue(np.isfinite(before))
        self.assertTrue(np.isfinite(after))
        self.assertEqual(len(history.losses), 5)

    def test_visualization_renders_svg(self):
        tree = demo_tree()
        network = convert_tree_to_network(tree)

        tree_svg = render_tree_svg(tree, sample=[0.6, 1.2, 0.0])
        network_svg = render_network_svg(network, n_features=3, sample=[0.6, 1.2, 0.0])

        self.assertIn("<svg", tree_svg)
        self.assertIn("Decision Tree Structure", tree_svg)
        self.assertIn("Neural Network Architecture", network_svg)
        self.assertIn("line", network_svg)


if __name__ == "__main__":
    unittest.main()

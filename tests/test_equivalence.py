import unittest

from dt_to_nn import (
    DecisionNode,
    Leaf,
    convert_tree_to_network,
    evaluate_equivalence,
    random_samples,
    threshold_probe_samples,
)


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

    def test_summary_contains_expected_network_shape(self):
        network = convert_tree_to_network(demo_tree())
        summary = network.summary()

        self.assertEqual(summary["max_depth"], 2)
        self.assertEqual(summary["condition_neurons"], 6)
        self.assertEqual(summary["output_neurons"], 3)


if __name__ == "__main__":
    unittest.main()

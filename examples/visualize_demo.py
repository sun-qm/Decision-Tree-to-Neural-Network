"""Generate SVG visualizations for a demo tree and generated networks."""

from pathlib import Path

from dt_to_nn.converter import convert_tree_to_network
from dt_to_nn.demo import build_demo_tree
from dt_to_nn.direct_path_converter import convert_tree_to_direct_path_network
from dt_to_nn.visualization import save_tree_and_network_svg


def main() -> None:
    tree = build_demo_tree()
    sample = [0.7, 1.2, -0.4]
    out_dir = Path("artifacts")
    original = convert_tree_to_network(tree)
    direct = convert_tree_to_direct_path_network(tree)

    save_tree_and_network_svg(
        tree,
        original,
        out_dir / "demo_original_tree_to_nn.svg",
        sample=sample,
        title="Original Recursive Tree-to-NN",
    )
    save_tree_and_network_svg(
        tree,
        direct,
        out_dir / "demo_direct_path_tree_to_nn.svg",
        sample=sample,
        title="Direct Path Tree-to-NN",
    )
    print(out_dir / "demo_original_tree_to_nn.svg")
    print(out_dir / "demo_direct_path_tree_to_nn.svg")


if __name__ == "__main__":
    main()

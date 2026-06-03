"""Compare original recursive parser with direct root-to-leaf path parser."""

from pprint import pprint

from dt_to_nn.direct_path_evaluation import run_direct_path_comparison


if __name__ == "__main__":
    pprint(run_direct_path_comparison(), sort_dicts=False)

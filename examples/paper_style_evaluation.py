"""Run the paper-style evaluation."""

from __future__ import annotations

from pprint import pprint

from dt_to_nn.paper_evaluation import run_paper_style_evaluation


if __name__ == "__main__":
    pprint(run_paper_style_evaluation(), sort_dicts=False)

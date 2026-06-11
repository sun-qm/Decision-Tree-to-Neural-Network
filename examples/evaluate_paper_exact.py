"""Evaluate the exact smooth-sigmoid equations from Editable XAI."""

from __future__ import annotations

from pprint import pprint

from dt_to_nn.paper_exact_evaluation import run_paper_exact_evaluation


if __name__ == "__main__":
    pprint(run_paper_exact_evaluation(), sort_dicts=False)

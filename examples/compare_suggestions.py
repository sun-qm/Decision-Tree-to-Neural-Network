"""Run experimental suggestion comparison without changing the baseline parser."""

from pprint import pprint

from dt_to_nn.suggestion_evaluation import run_suggestion_evaluation


if __name__ == "__main__":
    pprint(run_suggestion_evaluation(), sort_dicts=False)

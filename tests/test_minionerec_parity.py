import math

import numpy as np

from verl_gr.recipes.minionerec.minionerec_format import build_sid_prompt, parse_maybe_list
from verl_gr.recipes.minionerec.minionerec_reward import compute_score, ndcg_penalties, normalize_sid


def test_sid_dataset_prompt_matches_minionerec_template():
    prompt, history_key = build_sid_prompt(["<a_1><b_2><c_3>", "<a_4><b_5><c_6>"])
    assert history_key == "<a_1><b_2><c_3>::<a_4><b_5><c_6>"
    assert prompt == (
        "### User Input: \n"
        "The user has interacted with items <a_1><b_2><c_3>, <a_4><b_5><c_6> in chronological order. "
        "Can you predict the next possible item that the user may expect?\n\n"
        "### Response:\n"
    )


def test_parse_maybe_list_accepts_minionerec_csv_lists():
    assert parse_maybe_list("['<a_1><b_2><c_3>']") == ["<a_1><b_2><c_3>"]
    assert parse_maybe_list(np.array(["x", "y"])) == ["x", "y"]


def test_sample_level_reward_normalization():
    assert normalize_sid('" <a_1><b_2><c_3>\n" ') == "<a_1><b_2><c_3>"
    score = compute_score("minionerec", "<a_1><b_2><c_3>\n", "<a_1><b_2><c_3>\n", {})
    assert score["score"] == 1.0
    assert score["valid_sid"] == 1.0


def test_ndcg_penalties_match_minionerec_formula():
    penalties = ndcg_penalties(4)
    raw = [-1.0 / math.log2(i + 2) for i in range(4)]
    expected = [(-value / sum(raw)) for value in raw]
    assert penalties == expected
    assert all(value < 0 for value in penalties)

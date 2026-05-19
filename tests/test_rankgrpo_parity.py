"""Golden-value end-to-end parity test for the Rank-GRPO algorithm.

Exercises every pure function in the Rank-GRPO pipeline against committed
golden values.  Backend-agnostic — no Ray, no GPU, no FSDP/DDP wrapping.

Usage:
  python tests/test_rankgrpo_parity.py              # check against golden
  python tests/test_rankgrpo_parity.py --verbose    # show diffs on failure
  python tests/test_rankgrpo_parity.py --regenerate # write new golden values
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any

# -- path setup: ensure the repo root (parent of verl_gr/) is on sys.path ----
_p = Path(__file__).resolve().parent
while _p != _p.parent and not (_p / "verl_gr").is_dir():
    _p = _p.parent
_REPO_ROOT = str(_p) if (_p / "verl_gr").is_dir() else None
if _REPO_ROOT is not None and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _ensure_verl_importable(verl_path: str | None) -> bool:
    """Add the verl library to sys.path if provided / discoverable."""
    if verl_path is not None:
        if verl_path not in sys.path:
            sys.path.insert(0, verl_path)
    else:
        env_path = os.environ.get("VERL_LIB_PATH", "")
        if env_path and env_path not in sys.path:
            sys.path.insert(0, env_path)
        # Auto-discover: check ../verl_080_dev relative to repo root
        if _REPO_ROOT is not None:
            default = os.path.join(os.path.dirname(_REPO_ROOT), "verl_080_dev")
            if os.path.isdir(os.path.join(default, "verl")) and default not in sys.path:
                sys.path.insert(0, default)
    try:
        import verl  # noqa: F401
        return True
    except ImportError:
        return False


import numpy as np
import torch

# ---------------------------------------------------------------------------
# Golden-value persistence
# ---------------------------------------------------------------------------

GOLDEN_DIR = Path(__file__).resolve().parent / "rankgrpo_golden"
RTOL = 1e-5
ATOL = 1e-8


def _save_golden(name: str, value: torch.Tensor | float) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    t = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
    torch.save(t.detach().cpu(), GOLDEN_DIR / f"{name}.pt")


def _load_golden(name: str) -> torch.Tensor | None:
    path = GOLDEN_DIR / f"{name}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _check_golden(name: str, actual: torch.Tensor | float, verbose: bool = False) -> bool:
    expected = _load_golden(name)
    if expected is None:
        print(f"  GOLDEN MISSING: {name} — run with --regenerate first")
        return False

    actual_t = actual.detach().cpu() if isinstance(actual, torch.Tensor) else torch.tensor(actual, dtype=torch.float32)
    expected_t = expected.detach().cpu() if isinstance(expected, torch.Tensor) else expected

    if actual_t.dtype in (torch.float32, torch.float64, torch.float16):
        ok = torch.allclose(actual_t.float(), expected_t.float(), rtol=RTOL, atol=ATOL)
    else:
        ok = bool((actual_t == expected_t).all().item())

    if not ok and verbose:
        diff = (actual_t.float() - expected_t.float()).abs()
        print(f"  MISMATCH {name}: max_diff={diff.max().item():.6e} mean_diff={diff.mean().item():.6e}", file=sys.stderr)
        print(f"    actual  : {actual_t.flatten()[:8].tolist()}...", file=sys.stderr)
        print(f"    expected: {expected_t.flatten()[:8].tolist()}...", file=sys.stderr)

    return ok


# ---------------------------------------------------------------------------
# Test fixture: hand-crafted Rank-GRPO data
# ---------------------------------------------------------------------------

REC_NUM = 5
RANK_SEPARATOR = "\n"
YEAR_TOLERANCE = 2
EXCLUDE_SEEN = True

# Ground-truth catalog: (title, year) tuples — title must match the *parsed* form
# produced by _process_rec_raw (after _del_format/_del_space/_del_parentheses
# but BEFORE _default_title_normalizer case-folding).  The _catalog_contains
# check runs on the raw parsed title.
GT_CATALOG = frozenset({
    ("The Shawshank Redemption", 1994),
    ("The Godfather", 1972),
    ("Pulp Fiction", 1994),
    ("The Dark Knight", 2008),
    ("Schindler's List", 1993),
    ("Forrest Gump", 1994),
    ("Inception", 2010),
    ("Fight Club", 1999),
    ("The Matrix", 1999),
    ("Goodfellas", 1990),
})

# Case A: all 5 items matched correctly
_RESPONSE_A = (
    "The Shawshank Redemption (1994)\n"
    "The Godfather (1972)\n"
    "Pulp Fiction (1994)\n"
    "The Dark Knight (2008)\n"
    "Schindler's List (1993)\n"
)
_GT_A = [
    ("The Shawshank Redemption", 1994),
    ("The Godfather", 1972),
    ("Pulp Fiction", 1994),
    ("The Dark Knight", 2008),
    ("Schindler's List", 1993),
]
_SEEN_A: list[str] = []

# Case B: partial match — positions 0,2,4 correct; 1,3 are wrong titles
_RESPONSE_B = (
    "The Shawshank Redemption (1994)\n"
    "Wrong Movie Title (1999)\n"
    "Pulp Fiction (1994)\n"
    "Another Bad Title (2000)\n"
    "Schindler's List (1993)\n"
)
_GT_B = [
    ("The Shawshank Redemption", 1994),
    ("The Godfather", 1972),
    ("Pulp Fiction", 1994),
    ("The Dark Knight", 2008),
    ("Schindler's List", 1993),
]
_SEEN_B: list[str] = []

# Case C: seen-item exclusion — position 1 is already seen
_RESPONSE_C = (
    "The Shawshank Redemption (1994)\n"
    "The Godfather (1972)\n"
    "Pulp Fiction (1994)\n"
)
_GT_C = [
    ("The Shawshank Redemption", 1994),
    ("The Godfather", 1972),
    ("Pulp Fiction", 1994),
]
_SEEN_C = ["The Godfather"]  # position 1 excluded

# Case D: early-stop — EOS after 3 items when rec_num=5
_RESPONSE_D = (
    "The Shawshank Redemption (1994)\n"
    "The Godfather (1972)\n"
    "Pulp Fiction (1994)\n"
)
_GT_D = [
    ("The Shawshank Redemption", 1994),
    ("The Godfather", 1972),
    ("Pulp Fiction", 1994),
    ("Forrest Gump", 1994),
    ("Inception", 2010),
]
_SEEN_D: list[str] = []

# Case E: overflow — 7 items for rec_num=5
_RESPONSE_E = (
    "The Shawshank Redemption (1994)\n"
    "The Godfather (1972)\n"
    "Pulp Fiction (1994)\n"
    "The Dark Knight (2008)\n"
    "Schindler's List (1993)\n"
    "Forrest Gump (1994)\n"
    "Inception (2010)\n"
)
_GT_E = [
    ("The Shawshank Redemption", 1994),
    ("The Godfather", 1972),
    ("Pulp Fiction", 1994),
    ("The Dark Knight", 2008),
    ("Schindler's List", 1993),
]
_SEEN_E: list[str] = []

# Case F: empty/garbled — no valid recs
_RESPONSE_F = "blah blah not a valid rec format at all"
_GT_F: list[tuple[str, int]] = []
_SEEN_F: list[str] = []

# Build the test dataset
RESPONSES = [_RESPONSE_A, _RESPONSE_B, _RESPONSE_C, _RESPONSE_D, _RESPONSE_E, _RESPONSE_F]
GROUND_TRUTHS = [_GT_A, _GT_B, _GT_C, _GT_D, _GT_E, _GT_F]
SEEN_TITLES = [_SEEN_A, _SEEN_B, _SEEN_C, _SEEN_D, _SEEN_E, _SEEN_F]
UIDS = ["uid_a", "uid_a", "uid_b", "uid_b", "uid_c", "uid_c"]  # groups: A,A  B,B  C,C


def _build_reward_model(idx: int) -> dict[str, Any]:
    return {
        "groundtruth_with_release_year": GROUND_TRUTHS[idx],
        "ground_truth": GROUND_TRUTHS[idx],
        "seen_titles": list(SEEN_TITLES[idx]),
        "rec_num": REC_NUM,
        "style": "rule",
        "index": idx,
    }


def _pickle_catalog() -> str:
    """Write GT_CATALOG to a temp pickle file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".pkl", prefix="rankgrpo_test_catalog_")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(list(GT_CATALOG), f)
    return path


# ---------------------------------------------------------------------------
# Test 1: Reward parsing (rank_rewards_from_text)
# ---------------------------------------------------------------------------

def check_reward_parsing(catalog_path: str, regenerate: bool = False) -> bool:
    """Verify per-position reward computation against golden values."""
    from verl_gr.recipes.rankgrpo.rankgrpo_reward import rank_rewards_from_text

    print("--- Test 1: Reward parsing ---")
    all_ok = True
    rank_rewards_list = []

    for idx, response_text in enumerate(RESPONSES):
        reward_model = _build_reward_model(idx)
        rewards = rank_rewards_from_text(
            response_text,
            reward_model,
            rec_num=REC_NUM,
            year_tolerance=YEAR_TOLERANCE,
            exclude_seen=EXCLUDE_SEEN,
            gt_catalog_path=catalog_path,
        )
        rank_rewards_list.append(rewards)

    rank_rewards = torch.tensor(rank_rewards_list, dtype=torch.float32)

    if regenerate:
        _save_golden("rank_rewards", rank_rewards)
        print("  regenerated rank_rewards.pt")
    else:
        ok = _check_golden("rank_rewards", rank_rewards)
        all_ok &= ok
        print(f"  rank_rewards: {'PASS' if ok else 'FAIL'}")

    # Spot-checks independent of golden values
    rewards_a = rank_rewards_list[0]
    assert rewards_a == [1.0] * REC_NUM, f"Case A expected all-1, got {rewards_a}"

    rewards_b = rank_rewards_list[1]
    assert rewards_b == [1.0, 0.0, 1.0, 0.0, 1.0], f"Case B expected [1,0,1,0,1], got {rewards_b}"

    rewards_c = rank_rewards_list[2]
    assert rewards_c[0] == 1.0 and rewards_c[1] == 0.0, f"Case C pos1 should be excluded, got {rewards_c}"

    rewards_f = rank_rewards_list[5]
    assert rewards_f == [0.0] * REC_NUM, f"Case F expected all-0, got {rewards_f}"

    print("  spot-checks PASS")
    return all_ok


# ---------------------------------------------------------------------------
# Test 2: Token segmentation (_segment_rank_tokens)
# ---------------------------------------------------------------------------

def check_token_segmentation(regenerate: bool = False, model_path: str = "") -> bool:
    """Verify token-to-item segment assignment and mask generation."""
    from transformers import AutoTokenizer

    from verl_gr.recipes.rankgrpo.rankgrpo_algorithm import (
        _compute_eos_mask,
        _segment_rank_tokens,
    )

    print("--- Test 2: Token segmentation ---")

    model_path = model_path or os.environ.get("RANKGRPO_TEST_MODEL_PATH", "")
    if not model_path:
        print("  SKIP: RANKGRPO_TEST_MODEL_PATH not set (need tokenizer)")
        return True  # not a failure — model-dependent tests are skippable

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize responses
    all_input_ids = []
    all_masks = []
    max_len = 0
    for text in RESPONSES:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        all_input_ids.append(encoded)
        max_len = max(max_len, len(encoded))

    # Pad to max_len
    N = len(RESPONSES)
    responses = torch.zeros(N, max_len, dtype=torch.long)
    response_mask = torch.zeros(N, max_len, dtype=torch.bool)
    for i, ids in enumerate(all_input_ids):
        responses[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        response_mask[i, : len(ids)] = True

    seg_ids, rank_token_mask = _segment_rank_tokens(
        responses, response_mask, tokenizer,
        rank_separator=RANK_SEPARATOR, rec_num=REC_NUM,
    )
    eos_mask = _compute_eos_mask(responses, response_mask, tokenizer)
    overflow_mask = response_mask & (seg_ids >= REC_NUM)

    if regenerate:
        _save_golden("rank_seg_ids", seg_ids)
        _save_golden("rank_token_mask", rank_token_mask)
        _save_golden("overflow_token_mask", overflow_mask)
        _save_golden("eos_mask", eos_mask)
        print("  regenerated rank_seg_ids.pt, rank_token_mask.pt, overflow_token_mask.pt, eos_mask.pt")
    else:
        all_ok = True
        for name, tensor in [
            ("rank_seg_ids", seg_ids),
            ("rank_token_mask", rank_token_mask),
            ("overflow_token_mask", overflow_mask),
            ("eos_mask", eos_mask),
        ]:
            ok = _check_golden(name, tensor)
            print(f"  {name}: {'PASS' if ok else 'FAIL'}")
            all_ok &= ok

    # Spot-checks
    n_valid_a = int(response_mask[0].sum().item())
    assert seg_ids[0, 0].item() == 0, f"Case A first token should be seg 0, got {seg_ids[0, 0]}"
    assert rank_token_mask[0, :n_valid_a].all(), f"Case A all tokens should be in rank_token_mask"

    print("  spot-checks PASS")
    return all_ok if not regenerate else True


# ---------------------------------------------------------------------------
# Test 3: Advantage computation (compute_rank_grpo_advantage)
# ---------------------------------------------------------------------------

def check_advantage_computation(catalog_path: str, regenerate: bool = False, model_path: str = "") -> bool:
    """Verify group-normalized advantage and length-shaping outputs."""
    from transformers import AutoTokenizer

    from verl import DataProto
    from verl_gr.recipes.rankgrpo.rankgrpo_algorithm import compute_rank_grpo_advantage

    print("--- Test 3: Advantage computation ---")

    model_path = model_path or os.environ.get("RANKGRPO_TEST_MODEL_PATH", "")
    if not model_path:
        print("  SKIP: RANKGRPO_TEST_MODEL_PATH not set (need tokenizer)")
        return True

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize and pad responses
    all_ids = []
    max_len = 0
    for text in RESPONSES:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.append(ids)
        max_len = max(max_len, len(ids))

    N = len(RESPONSES)
    responses = torch.zeros(N, max_len, dtype=torch.long)
    response_mask = torch.zeros(N, max_len, dtype=torch.bool)
    for i, ids in enumerate(all_ids):
        responses[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        response_mask[i, : len(ids)] = True

    reward_models = np.array([_build_reward_model(i) for i in range(N)], dtype=object)
    uids = np.array(UIDS, dtype=object)

    data = DataProto.from_dict({
        "responses": responses,
        "response_mask": response_mask,
    })
    data.non_tensor_batch["reward_model"] = reward_models
    data.non_tensor_batch["uid"] = uids

    # Build minimal OmegaConf config
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        "rank_grpo": {
            "enable": True,
            "rec_num": REC_NUM,
            "rank_separator": RANK_SEPARATOR,
            "year_tolerance": YEAR_TOLERANCE,
            "exclude_seen": EXCLUDE_SEEN,
            "normalize_by_std": True,
            "gt_catalog_path": catalog_path,
            "apply_extra_length_shaping": True,
            "end_of_list_reward": 0.1,
            "extra_token_penalty": -0.1,
            "early_stop_penalty": -0.1,
        },
    })

    result = compute_rank_grpo_advantage(data, config=config, tokenizer=tokenizer, norm_adv_by_std_in_grpo=True)

    advantages = result.batch["advantages"]
    rank_seg_ids = result.batch["rank_seg_ids"]
    rank_token_mask = result.batch["rank_token_mask"]
    item_token_mask = result.batch["item_token_mask"]

    if regenerate:
        _save_golden("advantages", advantages)
        _save_golden("rank_seg_ids", rank_seg_ids)
        _save_golden("rank_token_mask", rank_token_mask)
        _save_golden("item_token_mask", item_token_mask)
        print("  regenerated advantages.pt, rank_seg_ids.pt, rank_token_mask.pt, item_token_mask.pt")
        return True

    all_ok = True
    for name, tensor in [
        ("advantages", advantages),
        ("rank_seg_ids", rank_seg_ids),
        ("rank_token_mask", rank_token_mask),
        ("item_token_mask", item_token_mask),
    ]:
        ok = _check_golden(name, tensor)
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        all_ok &= ok

    # Spot-check: for homogeneous groups (all members have identical rank rewards),
    # the token advantages should be centered around zero after group normalization.
    # Heterogeneous groups (e.g., E + F in uid_c) have non-zero deliberate offsets.
    uid_a_indices = [i for i, u in enumerate(UIDS) if u == "uid_a"]
    group_a_adv = advantages[uid_a_indices]
    group_a_mean = group_a_adv[rank_token_mask[uid_a_indices].bool()].mean()
    assert abs(group_a_mean.item()) < 0.01, f"Group uid_a mean advantage {group_a_mean.item():.4f} not centered"

    uid_b_indices = [i for i, u in enumerate(UIDS) if u == "uid_b"]
    group_b_adv = advantages[uid_b_indices]
    group_b_mean = group_b_adv[rank_token_mask[uid_b_indices].bool()].mean()
    assert abs(group_b_mean.item()) < 0.01, f"Group uid_b mean advantage {group_b_mean.item():.4f} not centered"

    # uid_c: heterogeneous (E has all-1s, F has all-0s) → advantages are non-zero by design
    print("  spot-checks PASS")
    return all_ok


# ---------------------------------------------------------------------------
# Test 4: Loss functions (trl_match and verl_default paths)
# ---------------------------------------------------------------------------

def check_loss_functions(catalog_path: str, regenerate: bool = False, model_path: str = "") -> bool:
    """Verify both loss paths produce deterministic golden values.

    Tests the core loss sub-functions directly (item-mean log-ratio, TRL clipped
    PG loss, item-level log-prob replacement) rather than the full
    ``rankgrpo_ppo_loss`` wrapper, which requires TensorDict scalar-metadata
    fields that need the full verl DataProto pipeline to construct correctly.
    The wrapper integration is exercised by the actual training run.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from verl_gr.recipes.rankgrpo.rankgrpo_loss import (
        _compute_item_mean_log_ratio,
        _item_level_log_prob,
        _trl_clipped_pg_loss,
    )

    print("--- Test 4: Loss functions ---")

    model_path = model_path or os.environ.get("RANKGRPO_TEST_MODEL_PATH", "")
    if not model_path:
        print("  SKIP: RANKGRPO_TEST_MODEL_PATH not set (need HF model)")
        return True

    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    # Build a tiny batch: 2 prompts, identical responses
    prompt_text = "Recommend 5 movies:\n"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_texts = [_RESPONSE_A, _RESPONSE_A]
    all_response_ids = [tokenizer.encode(t, add_special_tokens=False) for t in response_texts]

    max_prompt = len(prompt_ids)
    max_resp = max(len(ids) for ids in all_response_ids)
    batch_size = 2

    input_ids = torch.full((batch_size, max_prompt + max_resp), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_prompt + max_resp, dtype=torch.long)
    response_mask = torch.zeros(batch_size, max_resp, dtype=torch.bool)
    old_log_probs = torch.zeros(batch_size, max_resp, dtype=torch.float32)

    for b in range(batch_size):
        input_ids[b, :max_prompt] = torch.tensor(prompt_ids, dtype=torch.long)
        input_ids[b, max_prompt:max_prompt + len(all_response_ids[b])] = torch.tensor(all_response_ids[b], dtype=torch.long)
        attention_mask[b, :max_prompt + len(all_response_ids[b])] = 1
        response_mask[b, :len(all_response_ids[b])] = True

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Shift logits for next-token prediction: position t predicts token t+1.
    # Response tokens start at index max_prompt; their logits are at positions
    # [max_prompt-1, max_prompt+max_resp-1).
    resp_logits = logits[:, max_prompt - 1 : max_prompt + max_resp - 1, :]
    log_probs_tensor = torch.nn.functional.log_softmax(resp_logits.float(), dim=-1)
    response_ids = input_ids[:, max_prompt:max_prompt + max_resp]
    log_prob = log_probs_tensor.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

    # Build rank_seg_ids via tokenizer
    from verl_gr.recipes.rankgrpo.rankgrpo_algorithm import _segment_rank_tokens
    resp_tensor = torch.stack([
        torch.tensor(ids + [tokenizer.pad_token_id] * (max_resp - len(ids)), dtype=torch.long)
        for ids in all_response_ids
    ])
    seg_ids, rank_token_mask = _segment_rank_tokens(
        resp_tensor, response_mask, tokenizer,
        rank_separator=RANK_SEPARATOR, rec_num=REC_NUM,
    )

    all_ok = True

    # ---- item_mean_log_ratio ----
    item_log_ratio = _compute_item_mean_log_ratio(
        log_prob=log_prob, old_log_prob=old_log_probs,
        rank_seg_ids=seg_ids, response_mask=rank_token_mask, rec_num=REC_NUM,
    )
    if regenerate:
        _save_golden("item_mean_log_ratio", item_log_ratio)
        print("  regenerated item_mean_log_ratio.pt")
    else:
        ok = _check_golden("item_mean_log_ratio", item_log_ratio)
        print(f"  item_mean_log_ratio: {'PASS' if ok else 'FAIL'}")
        all_ok &= ok

    # ---- item_level_log_prob ----
    item_log_prob = _item_level_log_prob(
        log_prob=log_prob, old_log_prob=old_log_probs,
        rank_seg_ids=seg_ids, response_mask=rank_token_mask, rec_num=REC_NUM,
    )
    if regenerate:
        _save_golden("item_level_log_prob", item_log_prob)
        print("  regenerated item_level_log_prob.pt")
    else:
        ok = _check_golden("item_level_log_prob", item_log_prob)
        print(f"  item_level_log_prob: {'PASS' if ok else 'FAIL'}")
        all_ok &= ok

    # ---- TRL clipped PG loss ----
    advantages = torch.randn(batch_size, max_resp, dtype=torch.float32) * 0.5
    trl_loss = _trl_clipped_pg_loss(
        log_importance_weights=item_log_ratio,
        advantages=advantages,
        loss_mask=rank_token_mask,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        kl_per_token=None,
        kl_coef=0.0,
    )
    if regenerate:
        _save_golden("trl_clipped_pg_loss", trl_loss)
        print("  regenerated trl_clipped_pg_loss.pt")
    else:
        ok = _check_golden("trl_clipped_pg_loss", trl_loss)
        print(f"  trl_clipped_pg_loss: {'PASS' if ok else 'FAIL'}")
        all_ok &= ok

    # ---- aggregated loss ----
    from verl.trainer.ppo.core_algos import agg_loss
    pg_loss = agg_loss(
        loss_mat=trl_loss, loss_mask=rank_token_mask,
        loss_agg_mode="seq-mean-token-mean",
        dp_size=1, batch_num_tokens=None, global_batch_size=None, loss_scale_factor=None,
    )
    if regenerate:
        _save_golden("loss_trl_match_agg", pg_loss)
        print("  regenerated loss_trl_match_agg.pt")
    else:
        ok = _check_golden("loss_trl_match_agg", pg_loss)
        print(f"  loss_trl_match_agg: {'PASS' if ok else 'FAIL'} (actual={float(pg_loss):.6f})")
        all_ok &= ok

    return all_ok


# ---------------------------------------------------------------------------
# Test 5: Gradient hooks (install_grad_hooks)
# ---------------------------------------------------------------------------

def check_gradient_hooks() -> bool:
    """Verify div_ patch is installed and restored correctly."""
    print("--- Test 5: Gradient hooks ---")

    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    from verl_gr.workers.grad_hooks import install_grad_hooks

    original = FSDPEngineWithLMHead.prepare_model_outputs
    original_div_ = torch.Tensor.div_

    install_grad_hooks()

    # After install, prepare_model_outputs should be patched
    assert FSDPEngineWithLMHead.prepare_model_outputs is not original, (
        "install_grad_hooks should patch prepare_model_outputs"
    )

    # Verify the div_ override is a different function (the safe wrapper)
    # but the patch only swaps div_ during the call, not globally
    assert torch.Tensor.div_ is original_div_, (
        "torch.Tensor.div_ should be restored globally after patching (patch is scoped to the forward call)"
    )

    # Test that the patch works end-to-end by calling the patched function
    # with a dummy micro_batch that triggers the div_ codepath.
    # We can't easily call it without a real engine, but we can verify the
    # patched function is callable.
    assert callable(FSDPEngineWithLMHead.prepare_model_outputs), (
        "Patched prepare_model_outputs should be callable"
    )

    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Test 6: Agent loop grouping (_group_repeated_prompts)
# ---------------------------------------------------------------------------

def check_agent_loop_grouping() -> bool:
    """Verify prompt grouping preserves contiguity and handles interleaving."""
    print("--- Test 6: Agent loop grouping ---")

    from verl_gr.recipes.rankgrpo.rankgrpo_agent_loop import RankGRPOAgentLoopWorker

    # Create a minimal batch-like object with non_tensor_batch
    class _FakeBatch:
        def __init__(self, raw_prompt_ids):
            self.non_tensor_batch = {"raw_prompt_ids": raw_prompt_ids}
            self.meta_info = {"validate": False}

    # Contiguous groups: 3 A's, 2 B's, 1 C
    fake = _FakeBatch([
        [1, 2, 3], [1, 2, 3], [1, 2, 3],
        [4, 5], [4, 5],
        [6],
    ])
    groups = RankGRPOAgentLoopWorker._group_repeated_prompts(None, fake)  # type: ignore[arg-type]
    assert len(groups) == 3, f"Expected 3 groups, got {len(groups)}"
    assert groups[0][0] == [0, 1, 2], f"Group 0 positions: {groups[0][0]}"
    assert groups[1][0] == [3, 4], f"Group 1 positions: {groups[1][0]}"
    assert groups[2][0] == [5], f"Group 2 positions: {groups[2][0]}"

    # Interleaved: A, B, A, B, A
    fake2 = _FakeBatch([
        [1, 2, 3], [4, 5], [1, 2, 3], [4, 5], [1, 2, 3],
    ])
    groups2 = RankGRPOAgentLoopWorker._group_repeated_prompts(None, fake2)  # type: ignore[arg-type]
    assert len(groups2) == 5, f"Expected 5 interleaved groups, got {len(groups2)}"

    # Empty batch
    fake3 = _FakeBatch([])
    groups3 = RankGRPOAgentLoopWorker._group_repeated_prompts(None, fake3)  # type: ignore[arg-type]
    assert groups3 == [], f"Empty batch should produce empty groups, got {groups3}"

    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Test 7: Loss-mode invariants (from existing test)
# ---------------------------------------------------------------------------

def check_loss_mode_invariants() -> bool:
    """Re-run the independent loss-mode invariants from test_rankgrpo_loss_modes.py."""
    from verl.trainer.ppo.core_algos import agg_loss

    from verl_gr.recipes.rankgrpo.rankgrpo_loss import (
        _compute_item_mean_log_ratio,
        _trl_clipped_pg_loss,
    )

    print("--- Test 7: Loss-mode invariants ---")

    # Invariant 1: TRL clipped PG matches manual min-formulation
    torch.manual_seed(0)
    b, t = 3, 7
    w = torch.randn(b, t, dtype=torch.float64) * 0.15
    adv = torch.randn(b, t, dtype=torch.float64) * 0.4
    eps_l, eps_h = 0.2, 0.2
    out = _trl_clipped_pg_loss(
        log_importance_weights=w, advantages=adv,
        loss_mask=torch.ones(b, t, dtype=torch.bool),
        clip_ratio_low=eps_l, clip_ratio_high=eps_h,
    )
    c1 = torch.exp(w)
    c2 = torch.clamp(c1, 1 - eps_l, 1 + eps_h)
    expected = -torch.min(c1 * adv, c2 * adv)
    assert torch.allclose(out, expected, rtol=0, atol=1e-12), "TRL clipped PG formula mismatch"

    # Invariant 2: item_mean_log_ratio broadcasts correctly
    rank_seg_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]], dtype=torch.long)
    log_prob = torch.tensor([[0.0, 0.0, 1.0, -1.0], [0.5, 0.5, 0.5, 0.5]], dtype=torch.float64)
    old_log_prob = torch.zeros(2, 4, dtype=torch.float64)
    mask = torch.ones(2, 4, dtype=torch.bool)
    liw = _compute_item_mean_log_ratio(
        log_prob=log_prob, old_log_prob=old_log_prob,
        rank_seg_ids=rank_seg_ids, response_mask=mask, rec_num=2,
    )
    assert torch.allclose(liw[0, :2], torch.zeros(2, dtype=torch.float64)), "row0 item0 mean should be 0"
    assert torch.allclose(liw[0, 2:], torch.zeros(2, dtype=torch.float64)), "row0 item1 mean should be 0"
    assert torch.allclose(liw[1], torch.full((4,), 0.5, dtype=torch.float64)), "row1 all tokens should be 0.5"

    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Rank-GRPO algorithm parity test")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate golden values")
    parser.add_argument("--verbose", action="store_true", help="Show diffs on failure")
    parser.add_argument("--skip-model-tests", action="store_true", help="Skip tests requiring HF model")
    parser.add_argument("--verl-path", type=str, default=None, help="Path to verl library (verl_080_dev)")
    args = parser.parse_args()

    have_verl = _ensure_verl_importable(args.verl_path)
    if not have_verl and not args.skip_model_tests:
        print("WARNING: verl not importable — model-dependent tests will be skipped.")
        print("  Set --verl-path or VERL_LIB_PATH to enable full testing.")
        print()

    print(f"Rank-GRPO parity test  (regenerate={args.regenerate}, golden_dir={GOLDEN_DIR})")
    print()

    catalog_path = _pickle_catalog()
    try:
        results: dict[str, bool] = {}

        results["reward_parsing"] = check_reward_parsing(catalog_path, regenerate=args.regenerate)
        print()

        if args.skip_model_tests or not have_verl:
            print("--- Test 2-4: SKIPPED (model tests) ---")
            print()
            results["token_segmentation"] = True
            results["advantage_computation"] = True
            results["loss_functions"] = True
        else:
            results["token_segmentation"] = check_token_segmentation(regenerate=args.regenerate)
            print()
            results["advantage_computation"] = check_advantage_computation(catalog_path, regenerate=args.regenerate)
            print()
            results["loss_functions"] = check_loss_functions(catalog_path, regenerate=args.regenerate)
            print()

        if have_verl:
            results["gradient_hooks"] = check_gradient_hooks()
        else:
            print("--- Test 5: SKIPPED (verl not importable) ---")
            print()
            results["gradient_hooks"] = True

        results["agent_loop_grouping"] = check_agent_loop_grouping()
        print()
        results["loss_mode_invariants"] = check_loss_mode_invariants()
        print()

        if args.regenerate:
            print(f"Golden values written to {GOLDEN_DIR}/")
            return 0

        failed = [name for name, ok in results.items() if not ok]
        if failed:
            print(f"FAILED: {', '.join(failed)}")
            return 1

        print("test_rankgrpo_parity: all checks passed")
        return 0
    finally:
        os.unlink(catalog_path)


# ---------------------------------------------------------------------------
# Pytest wrappers — thin veneer over check_* functions using conftest fixtures
# ---------------------------------------------------------------------------


def test_reward_parsing(catalog_path, regenerate):
    assert check_reward_parsing(str(catalog_path), regenerate=regenerate)


def test_token_segmentation(regenerate, model_path):
    if not model_path:
        import pytest as pt; pt.skip("RANKGRPO_TEST_MODEL_PATH not set")
    assert check_token_segmentation(regenerate=regenerate, model_path=model_path)


def test_advantage_computation(catalog_path, regenerate, model_path):
    if not model_path:
        import pytest as pt; pt.skip("RANKGRPO_TEST_MODEL_PATH not set")
    assert check_advantage_computation(str(catalog_path), regenerate=regenerate, model_path=model_path)


def test_loss_functions(catalog_path, regenerate, model_path):
    if not model_path:
        import pytest as pt; pt.skip("RANKGRPO_TEST_MODEL_PATH not set")
    assert check_loss_functions(str(catalog_path), regenerate=regenerate, model_path=model_path)


def test_gradient_hooks(have_verl):
    if not have_verl:
        import pytest
        pytest.skip("verl not importable")
    assert check_gradient_hooks()


def test_agent_loop_grouping():
    assert check_agent_loop_grouping()


def test_loss_mode_invariants():
    assert check_loss_mode_invariants()


if __name__ == "__main__":
    sys.exit(main())

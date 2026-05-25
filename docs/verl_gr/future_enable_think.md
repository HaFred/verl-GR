# Future Feature: Enable Thinking Mode for OpenOneRec GRPO

**Status:** Proposed (not yet implemented)
**Priority:** Medium — config change only, but may impact training dynamics
**Parent Plan:** `verl-gr-acceleration-plan.md`

## Motivation

OpenOneRec currently has `enable_think: false` in its config. This means the model
generates unstructured CoT reasoning followed by SID tokens, with the reward
function receiving only an SID accuracy signal.

Enabling `enable_think: true` would:

1. Prompt the model with `/think` appended to user messages, instructing it to
   wrap reasoning in `` tags
2. Activate `think_format_reward()` which is already implemented but always
   returns 0.0 when no `` tags are present
3. Give GRPO a **process-level reward** on reasoning tokens alongside the
   existing outcome-level reward on SID tokens

## Current State

```yaml
# configs/verl_gr/openonerec/grpo_trainer.yaml
data:
  enable_think: false
  enable_nonthink: false
```

The `think_format_reward()` in `onerec_recipe.py:520-529` checks for `` tags:

```python
def think_format_reward(prediction: str) -> float:
    if "<think>" not in prediction or "</think>" not in prediction:
        return 0.0        # ← always hit when enable_think=false
    ...
    return 1.0 if len(content_stripped) > 10 else 0.0
```

## Proposed Change

Simply flip the config flag:

```yaml
data:
  enable_think: true      # was false
  enable_nonthink: false  # unchanged
```

This requires no code changes. The dataset pipeline (`extract_prompt_fields`)
already handles `enable_think` by appending `/think` to user prompts.

## Expected Benefits for GRPO

| Aspect | Current (enable_think=false) | With enable_think=true |
|---|---|---|
| Reasoning structure | Unstructured, no delimiters | ``  tags create a natural phase boundary |
| Format reward | Always 0.0 | 1.0 when reasoning content > 10 chars |
| Token-level advantages | Only SID tokens get meaningful advantage | Reasoning tokens get process reward; SID tokens get outcome reward |
| KL regularization | Applied uniformly | Can differentiate between reasoning and answer phases |
| Interpretability | Hard to inspect reasoning quality | `` tags make reasoning extractable and evaluable |

## Risks

1. **Format collapse** — the model may learn to generate minimal `` content
   (just barely > 10 chars) to maximize format reward without genuine reasoning.
   Mitigation: tie format reward threshold to actual reasoning quality or add
   a length scaling factor.

2. **Training dynamics shift** — adding a new reward signal changes the RL
   optimization landscape. Start with a small weight for format_reward and
   monitor SID accuracy.

3. **Prompt compatibility** — the `/think` suffix must be compatible with the
   chat template used by the model. Verify with the target model family
   (Qwen2/3).

## Testing Plan

### Unit: Format Reward Behavior

Test `think_format_reward()` with:
- Input containing `` with content → returns 1.0
- Input with empty `` → returns 0.0
- Input without `` tags → returns 0.0
- Input with `` only (no ``) → returns 0.0
- Input with reversed `` → returns 0.0

### E2E: Training With Thinking Enabled

Run a short training loop (10-20 steps) with `enable_think: true` and verify:
1. Training completes without errors
2. `format_reward` values are non-zero (assert mean > 0.1 after warmup)
3. SID accuracy (`pass_at_1`) does not degrade compared to `enable_think: false`
4. The model consistently produces ``  output structure

## Relationship to Multi-Turn Trajectories

Enabling thinking does NOT create multi-turn RL trajectories in the verl sense.
The model still generates the entire `` sequence in a single
`generate()` call. There is no agent-environment interaction between turns.

For true multi-turn recommendation trajectories, a separate design is needed:
- Turn 1: model proposes coarse category → intermediate reward
- Turn 2: model proposes specific items → final SID accuracy reward
- This requires a multi-turn agent loop (`ToolAgentLoop` instead of
  `SingleTurnAgentLoop`) and intermediate reward computation.

## Implementation Steps

1. **Change config**: set `data.enable_think: true` in `grpo_trainer.yaml`
2. **Verify reward**: run one step with thinking enabled, check `format_reward` is non-zero
3. **Write tests**: add `test_think_format_reward.py` with 5 cases
4. **Benchmark**: compare SID accuracy and training curves with `enable_think: false` over 50-100 steps
5. **Tune**: adjust format_reward weight if needed (currently hardcoded at 1.0 in `compute_score`)

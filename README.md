# verl-GR


<div align="center">
  <img src="docs/assets/arch.png" alt="verl-gr architecture diagram" width="70%">
</div>


## Layer 1: Recommendation Recipes

A **recipe** is not a config file — it is a `RecipeTaskRuntime` subclass that wires together dataset, reward, rollout backend, worker class, and trainer adapter for one generative-recommendation workload.

### OpenOneRec (`OneRecTask`)

**What it trains:** A model that first generates a reasoning block, then emits structured item IDs (SID tuples like `<s_a_X><s_b_Y><s_c_Z>`).

**Code bundle:**
- **Dataset:** `OneRecDataset` — parses chat `messages`, extracts prompts and `reward_model.ground_truth`, supports `/think` / `/no_think` suffixes and optional forced prefix `<|sid_begin|>`.
- **Reward:** `compute_score()` — rule-based scoring over SID tuples: `pass_at_1`, `hit_reward`, `partial_hit_reward` (hierarchical match on tuple prefixes), `think_format_reward`.
- **Rollout:** `rollout.name = "two_stage"` — stage-1 reasoning tokens, then stage-2 beam search over item tokens.
- **Worker:** `OneRecActorRolloutRefWorker` when two-stage rollout is active.
- **Agent loop:** `OpenOneRecAgentLoopManager` / `OpenOneRecTwoStageAgentLoop` — assigns `beam_group_id`, `beam_index`, `stage1_sample_idx` so multiple rollout slots share one stage-1 KV cache.

**Rollout count expansion:** `expand_rollout_counts()` sets `rollout.n = base_n × beam_width` (default beam 32), so the trainer allocates one slot per beam candidate, not one slot per prompt.

### MiniOneRec (`MiniOneRecTask`)

**What it trains:** Single-stage constrained generation over a catalog of semantic item IDs (SIDs), using prefix-trie masking so every decoded token is catalog-valid.

**Code bundle:**
- **Dataset:** `MiniOneRecDataset` — plain-text prompts for item recommendation.
- **Reward:** `minionerec_reward.compute_score()` — rule-based hit/format rewards on constrained outputs.
- **Loss:** `compute_policy_loss_minionerec_reinforce` — REINFORCE mode that allows skipping the old_log_prob forward pass (~20% step time saved).
- **Rollout:** `rollout.name = "constrained_beam"` with `beam_search_params.constraint` pointing to a catalog `info_file`.
- **Worker:** `MiniOneRecActorRolloutRefWorker` — on DDP/FSDP, skips vLLM engine creation entirely; generation goes through HF `model.generate()`.
- **Agent loop:** `MiniOneRecConstrainedBeamAgentLoopManager` — tokenizes plain `raw_prompt_text`, routes to constrained beam server with decode modes (`deterministic_beam`, `stochastic_constrained`, `stochastic_beam`).

**Rollout count expansion:** `rollout.n = num_generations_per_prompt × beam_width`.

**Training optimizations:** `configure_training_optimizations()` enables `completion_only_logprob` on actor/ref engines (logprob only over response tokens via `logits_to_keep`) and registers `minionerec_engine_patch`.

### RankGRPO (`RankGRPOTask`)

**What it trains:** Multi-item ranked lists (title + year per line) with GRPO-style group-relative advantages computed over **ranking slots**, not individual tokens.

**Code bundle:**
- **Dataset:** `RankGRPODataset` — list-generation prompts with ground-truth catalogs.
- **Reward:** `rankgrpo_reward.rank_rewards_from_text()` — parses ranked lines, scores against a GT catalog with year tolerance and seen-item exclusion.
- **Tokenizer:** `build_rankgrpo_tokenizer_and_processor()` — custom rank-separator handling and EOS padding (overrides the generic `prepare()` path entirely).
- **Algorithm:** `compute_rank_grpo_advantage()` — segments response tokens by rank separator, computes per-completion rank rewards, centers within GRPO groups, then **broadcasts scalar rank advantages to all tokens in that rank slot**.
- **Loss:** `rankgrpo_loss` — Rank-GRPO PPO loss with shape penalties.
- **Agent loop:** `RankGRPOAgentLoopManager` — fires concurrent `n=1` vLLM requests (or batched via `generate_n_samples` with `n=N` for KV sharing).

RankGRPO uses the upstream `ActorRolloutRefWorker` and standard vLLM; its specialization is in advantage computation and the concurrent/colocated generation agent loop, not beam search.

---

## Layer 2: Shared Trainer & Task Runtime

The middle tier splits **infrastructure setup** (`RecipeTaskRuntime`) from **RL training loop** (`RLTrainer`).

### RecipeTaskRuntime — what `prepare()` actually does

Base class: `verl_gr/recipes/task_runtime.py`.

| Hook | Purpose (from code) |
|---|---|
| `expand_rollout_counts(config)` | Recipe-specific: multiply `rollout.n` by beam width so trainer/advantage code sees the right number of generations per prompt. |
| `configure_rollout(config)` | Register custom rollout classes into upstream `_ROLLOUT_REGISTRY`, set `agent_loop_manager_class` and `default_agent_loop`. |
| `configure_lora(config)` | Normalize LoRA via `normalize_lora_config()`; for DDP+LoRA, set `ddp_find_unused_parameters=True`. |
| `configure_fsdp_wrap_policy(config, model_path)` | Read HF `model_type`, set `transformer_layer_cls_to_wrap` to the correct decoder layer class (Qwen2, Llama, etc.). |
| `sanitize_fsdp2_wrap_policy(config)` | Normalize wrap-policy values to list form for FSDP2. |
| `get_actor_rollout_ref_worker(config)` | Return recipe-specific worker (`OneRecActorRolloutRefWorker`, `MiniOneRecActorRolloutRefWorker`, or default). |
| `prepare(config)` | Orchestrates all hooks → returns `{tokenizer, processor, actor_rollout_cls, critic_worker, ray_worker_group_cls}`. |

**FSDP / DDP / LoRA in the diagram:** These are not separate boxes — they are side effects inside `prepare()`. FSDP wrap policy is auto-aligned to the HF model family; LoRA normalization runs before worker construction; DDP engine is registered via `import verl_gr.workers.engine.ddp` when strategy is `ddp`.

### RLTrainer — what the three diagram labels mean

Class: `verl_gr/trainers/rl_trainer.py`, extends upstream `RayPPOTrainer`.

#### Worker / Tokenizer

Not a new worker type — this refers to the **recipe-selected** `actor_rollout_cls` and tokenizer returned by `prepare()`. RankGRPO replaces the generic tokenizer build with `build_rankgrpo_tokenizer_and_processor()`. MiniOneRec/OpenOneRec may return custom FSDP workers with `RefSyncMixin` and recipe-specific loss registration.

#### EMA Ref Update

Implemented in two places:
- **`RefSyncMixin.sync_ref_weights(mixup_alpha)`** (`workers/ref_sync.py`) — TRL-style EMA: `ref = (1 − α)·ref + α·actor`, supporting DDP in-place and FSDP gather-mix-load.
- **`RLTrainer._try_sync_ref_model()`** — called after each actor update; default `sync_freq=512` for constrained_beam, `ref_model_mixup_alpha=0.6`.

Upstream VeRL typically keeps a frozen reference or hard-copies weights. VeRL-GR adds periodic soft tracking tuned for noisy recommendation rewards.

#### Ranked Adv (Ranked Advantage)

When `algorithm.rank_grpo.enable=True`, `compute_advantage()` delegates to `compute_rank_grpo_advantage()` instead of `core_algos.compute_grpo_outcome_advantage()`:

1. Decode each completion to text.
2. Score with `rank_rewards_from_text()` → per-completion scalar rank reward.
3. Group by `uid`, center (and optionally normalize by std) within each GRPO group.
4. Segment response tokens by rank separator (`\n` by default) into rank slots (`seg_ids`).
5. Broadcast the rank-level advantage to every token in that slot via `rank_advantages.gather(1, clamped_seg_ids)`.

This is fundamentally different from token-level or sequence-level GRPO: the unit of comparison is **rank position within a generated list**, not the whole completion.

#### TrainerTaskAdapter — recipe hooks into the training loop

`RLTrainer._get_task_adapter()` selects:
- `RankGRPOTrainerAdapter` — custom validation metrics (`ndcg`, `hit`, `rank_reward_sum`).
- `MiniOneRecTrainerAdapter` — reward postprocessing for rule-based RM path.
- `_OpenOneRecTrainerAdapter` — two-stage gen batch prep, pass@1 validation, checkpoint evaluation.

Key override points:

| Method | Used by |
|---|---|
| `prepare_gen_batch` | Injects beam/two-stage metadata into generation batch (`enable_two_stage_rollout`, `constraint`, `beam_width`). |
| `postprocess_rewards` | MiniOneRec: sets `rm_scores` from rule-based reward without a neural RM. |
| `validate` | Recipe-specific metrics and generation dumps. |
| `evaluate_and_prune_checkpoint` | OpenOneRec: checkpoint quality evaluation. |

Additional RLTrainer capabilities beyond the diagram:
- **Top-k checkpoint pruning** by validation metric (`_update_topk_checkpoints`).
- **REINFORCE old_log_prob bypass** when `loss_mode == "minionerec_reinforce"`.
- **Throttled wandb logging** via `logging_steps`.

---

## Layer 3: Rollout Engine

The trainer connects to rollout via **`agent_loop_manager_class`** — a class injected into config by each recipe's `configure_rollout()`. This selects how prompts become token sequences and how many async requests are issued.

### Registration into upstream VeRL

`workers/rollout/registration.py` extends upstream registries:

```python
_ROLLOUT_REGISTRY[("two_stage", "async")]       → TwoStagevLLMRollout
_ROLLOUT_REGISTRY[("constrained_beam", "async")] → ConstrainedBeamvLLMRollout
RolloutReplicaRegistry.register("two_stage", TwoStagevLLMReplica)
RolloutReplicaRegistry.register("constrained_beam", ConstrainedBeamvLLMReplica)
```

### HfConstrainedBeamGenerator — Sequential on μBatch

Class: `recipes/minionerec/hf_constrained_generation.py`.

Used by `MiniOneRecActorRolloutRefWorker` on DDP/FSDP when vLLM is not involved:
- Builds a **prefix-trie** from catalog `info_file` (`_build_hash_dict` / `PrefixTrieConstraint`).
- Generates via HF `model.generate()` with a `LogitsProcessor` that masks illegal tokens at each step.
- Processes prompts in **micro-batches** (`micro_batch_size`, default 16) sequentially to bound VRAM — this is the diagram's "Sequential on μBatch".
- Supports train (stochastic beam, `do_sample=True`) and eval (deterministic beam) modes.

The trie logic is shared with the vLLM path via `workers/rollout/constraints.py` → `PrefixTrieConstraint.from_info_file()`.

### TwoStage vLLMHttpServer — Concurrent on μBatch

Class: `workers/rollout/two_stage_vllm_async.py` — extends upstream `vLLMHttpServer`.

**Stage 1:** Sample reasoning tokens (configurable `stage1_max_tokens`, stop sequences).
**Stage 2:** Run token-by-token beam search over item tokens via `run_async_beam_search()`.

Key mechanisms:
- **`_get_or_build_two_stage_cache_entry()`** — first beam request in a group runs stage-1 + full beam; subsequent beam indices read from cache. Avoids recomputing stage-1 KV for every beam candidate.
- **`_two_stage_engine_request_semaphore`** — caps in-flight vLLM requests (default 8) because beam search fans out one async request per beam step.
- **`beam_group_id`** from agent loop (`{step}:{validate}:{sample_index}:{stage1_sample_idx}`) keys the cache.

"Concurrent on μBatch" refers to the async agent loop dispatching many beam-step requests concurrently (via `asyncio.gather` inside `run_async_beam_search`), while the HF path processes micro-batches sequentially.

### Beam Backend (`run_async_beam_search`)

Module: `workers/rollout/beam_backend.py`.

A reusable **token-by-token async beam search kernel**:
- At each step, calls `generate_one_token` (vLLM async API) for all active beams.
- Scores candidates with length penalty, prunes to `beam_width`.
- Supports `allowed_tokens_fn` callback for trie constraints.
- Decode modes: `deterministic_beam`, `stochastic_constrained`, `stochastic_beam`.

Both `TwoStagevLLMHttpServer` and `ConstrainedBeamvLLMHttpServer` call this kernel — the diagram's "expand" arrow from TwoStage to Beam Backend.

### vLLMHttpServer (standard path)

Used by:
- **OpenOneRec** stage-1 sampling (via `super().generate()` inside two-stage).
- **RankGRPO** via `RankGRPOAgentLoopManager` + `generate_n_samples()` — single vLLM call with `n=N` for colocated multi-sample generation with guaranteed prompt KV sharing.

RankGRPO deliberately uses `n=1` per concurrent request in some paths to match TRL's independent random state per sample; the fast path batches via `n=N` when prompts are contiguous.

---

## Contrast with Upstream VeRL

| Concern | Upstream VeRL | VeRL-GR |
|---|---|---|
| **Task organization** | Single `TaskRunner.run()` + dataset reward fn swap | `TASK_REGISTRY` with `RecipeTaskRuntime` hooks per recipe |
| **Rollout API** | `generate_sequences(prompts) → outputs` (vLLM/SGLang/TRT-LLM) | Same base + `two_stage` and `constrained_beam` backends registered into `_ROLLOUT_REGISTRY` |
| **Beam search** | None | `run_async_beam_search` kernel + trie constraints |
| **Catalog-valid decoding** | None | `PrefixTrieConstraint` + HF/vLLM constrained paths |
| **Two-stage generation** | None | Stage-1 reasoning + cached stage-2 beam in one async request lifecycle |
| **Advantage computation** | Token/sequence-level GAE/GRPO/PPO | + `compute_rank_grpo_advantage` — rank-slot-level GRPO |
| **Reference policy** | Frozen or hard copy | `RefSyncMixin` EMA with configurable α |
| **Trainer extensibility** | Fixed `RayPPOTrainer` lifecycle | `RLTrainer` + `TrainerTaskAdapter` delegate methods |
| **Agent loops** | Generic single-turn | Recipe-specific: two-stage metadata, constrained beam decode modes, RankGRPO concurrent gather |
| **Worker customization** | `ActorRolloutRefWorker` + backend strategy | Recipe workers: skip vLLM (MiniOneRec HF path), custom loss registration, ref sync mixin |

VeRL-GR's rollout is a **programmable generation runtime** where beam width, trie constraints, two-stage KV reuse, and concurrent/async dispatch are first-class.

```
Upstream:  Prompt ──[vLLM: n=8, temp=1.0]──→ 8 independent samples

OpenOneRec: Prompt ──[Stage-1 reasoning]──→ prefix ──[beam × 32, async]──→ 32 SID candidates
MiniOneRec: Prompt ──[trie-guided beam, HF or vLLM]──→ catalog-valid SIDs
RankGRPO:   Prompt ──[vLLM n=N or concurrent n=1]──→ ranked lists → rank-slot GRPO adv
```

# Get Started
## Source Code Overview

- `verl_gr/recipes/`: task-specific implementations and data/reward logic (for example, OpenOneRec runtime preparation and workers).
- `verl_gr/trainers/`: trainer-side wrappers around upstream `verl` trainer code.
- `verl_gr/workers/`: rollout-side extensions that are still useful outside a single recipe.
- `verl_gr/third_party/`: small compatibility helpers for non-`verl` dependencies such as `vllm`.

## Docs

- `docs/verl_gr/openonerec_mapping.md`: maps legacy OpenOneRec runtime modules to the current `verl_gr` layout.
- `docs/verl_gr/openonerec_parity_plan.md`: tracks the current Phase B parity/smoke checklist after the cleanup refactor.
- `docs/verl_gr/minionerec_mapping.md`: MiniOneRec dataset / reward / beam contract.
- `docs/verl_gr/minionerec_pr_changes.md`: workingbranch vs `main` (MiniOneRec + performance).
- `docs/verl_gr/rankgrpo_mapping.md`: RankGRPO vs TRL root-cause comparison and analysis.
- `docs/verl_gr/rankgrpo_target.md`: alignment progress tracker by target item (convergence & efficiency).
- `scripts/README.md`: launcher index for GRPO / SFT / profiling scripts.

## Data preparation

You will need to download `OpenOneRec/OpenOneRec-RecIF` first and then curate the RL data one-stop as follows. The flow is `OpenOneRec-RecIF -> recommendation data preprocessing -> RL data split`. Patch `verl-GR/verl_gr/recipes/openonerec/data/recif_preprocessing.sh` before getting started.

```bash
RECIF_DIR=/YOUR/RECIF/DIR
```

Then run:

```bash
cd verl-GR/verl_gr/recipes/openonerec/data
bash recif_preprocessing.sh
bash prepare_rl.sh
```

You will get the RL training data:
- `verl-GR/verl_gr/recipes/openonerec/output/rl_data/train.parquet` - Training set (remaining data after merging all tasks)
- `verl-GR/verl_gr/recipes/openonerec/output/rl_data/test.parquet` - Test set (1000 samples randomly sampled from merged data)

For Rank-GRPO data, you need to download the Reddit-V2 dataset. Or simply download the preprocessed version [here](https://drive.google.com/file/d/11tOfUMlVOylkkcnwPqGM_0IuiIeHjLle/view)

## Launching Guide

1. Install base dependencies from the official script in `requirements.txt` comments, then install pinned packages in this repo.

```bash
cd verl-GR
pip install -r requirements.txt
```

2. Run the OpenOneRec GRPO launcher (set your model path first).

```bash
cd verl-GR
export BASE_MODEL=/path/to/your/model
bash scripts/run_openonerec_grpo.sh
```

3. MiniOneRec GRPO (DDP, aligned with `MiniOneRec/rl.sh`; requires `bitsandbytes` for `paged_adamw_32bit`):

```bash
cd verl-GR
export BASE_MODEL=/path/to/your/checkpoint
export PYTHON_BIN=/path/to/vllm-gr/bin/python
bash scripts/run_minionerec_grpo_rl_aligned.sh
```

4. Rank-GRPO (set your model path first)
```bash
cd verl-GR
export BASE_MODEL=/path/to/your/checkpoint
bash scripts/run_rankgrpo.sh
```

## Two-Stage Notes

- OpenOneRec `two_stage` is implemented entirely inside `verl-GR`.
- The async path uses `verl_gr/recipes/openonerec/two_stage_agent_loop.py` together with `verl_gr/workers/rollout/two_stage_vllm_async.py`.
- No local source patch to the upstream `verl` repo is required or expected.

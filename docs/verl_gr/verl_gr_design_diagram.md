# `verl_gr` Design Diagram

This diagram shows how the three recipe workloads plug into the shared `verl_gr`
runtime after the recipe refactor. The main path is:

1. `verl_gr.trainers.main_ppo` selects a task runtime and builds datasets.
2. `RecipeTaskRuntime` or a recipe task prepares tokenizer, processor, worker class, and rollout registration.
3. `RLTrainer` delegates recipe-specific generation and validation through `TrainerTaskAdapter`.
4. Custom beam workloads register rollout replicas and async agent loops under `verl_gr.workers.rollout`.

```mermaid
flowchart LR
  subgraph C1["Role 1: Entrypoint & Task Selection"]
    TaskRunner["main_ppo.TaskRunner\n- infer/select task\n- build datasets\n- construct RLTrainer"]
    TaskSpec["TaskSpec registry\nopenonerec | rankgrpo"]
    TaskFactory["task_factory\nconfig-driven loader\nMiniOneRec and custom tasks"]
  end

  subgraph C2["Role 2: Recipe Runtime & Data"]
    RecipeTaskRuntime["RecipeTaskRuntime\n- FSDP wrap cleanup\n- tokenizer/processor setup\n- rollout hooks"]
    OneRecTask["OneRecTask\nOpenOneRec two-stage rollout"]
    MiniOneRecTask["MiniOneRecTask\nconstrained-beam rollout"]
    RankGRPOTask["RankGRPOTask\nvanilla vLLM rollout\nRank-GRPO tokenizer"]
    Datasets["Recipe datasets\nOneRecDataset\nMiniOneRecDataset\nRankGRPODataset"]
    RankGRPOTokenizer["rankgrpo_tokenizer\nbuild tokenizer / processor"]
  end

  subgraph C3["Role 3: Trainer, Validation & Reward"]
    RLTrainer["RLTrainer\n- task adapter delegation\n- recommendation gen batch prep\n- Rank-GRPO advantage hook\n- tqdm timing excludes validate/save"]
    TrainerTaskAdapter["TrainerTaskAdapter\nbase delegation surface"]
    OpenOneRecAdapter["OpenOneRecTrainerAdapter\nvalidate / dump / log generations"]
    MiniOneRecAdapter["MiniOneRecTrainerAdapter\npostprocess rewards / validate"]
    RankGRPOAdapter["RankGRPOTrainerAdapter\nvalidate\nstdout val-generation preview"]
    RankGRPOAlgorithm["rankgrpo_algorithm\nper-rank GRPO advantages\ncatalog-aware reward rows"]
    RankGRPOReward["rankgrpo_reward\nreference-aligned parsing\nGT catalog matching\nscore/rank_rewards/rank_reward_sum"]
    RankGRPOMetrics["Rank-GRPO validation aliases\neval/reward\neval/reward_total\neval/loss"]
  end

  subgraph C4["Role 4: Rollout, Beam Search & Agent Loops"]
    RolloutRegistration["rollout registration\ntwo_stage | constrained_beam"]
    TwoStageRollout["TwoStagevLLMRollout\nTwoStagevLLMHttpServer\nTwoStagevLLMReplica"]
    ConstrainedRollout["ConstrainedBeamvLLMRollout\nConstrainedBeamvLLMHttpServer\nConstrainedBeamvLLMReplica"]
    BeamBackend["BeamBackend / BeamCandidate\nshared async beam ranking"]
    OpenOneRecLoop["OpenOneRecTwoStageAgentLoop\nWorker / Manager"]
    MiniOneRecLoop["MiniOneRecConstrainedBeamAgentLoop\nWorker / Manager"]
    VanillaVLLM["Upstream vanilla vLLM\nused by Rank-GRPO"]
  end

  TaskRunner --> TaskSpec
  TaskRunner --> TaskFactory
  TaskRunner --> RecipeTaskRuntime
  TaskRunner --> Datasets
  TaskRunner --> RLTrainer

  RecipeTaskRuntime --> OneRecTask
  RecipeTaskRuntime --> MiniOneRecTask
  RecipeTaskRuntime --> RankGRPOTask
  RankGRPOTask --> RankGRPOTokenizer

  RLTrainer --> TrainerTaskAdapter
  TrainerTaskAdapter --> OpenOneRecAdapter
  TrainerTaskAdapter --> MiniOneRecAdapter
  TrainerTaskAdapter --> RankGRPOAdapter
  RLTrainer --> RankGRPOAlgorithm
  RankGRPOAlgorithm --> RankGRPOReward
  RankGRPOAdapter --> RankGRPOMetrics

  OneRecTask --> RolloutRegistration
  MiniOneRecTask --> RolloutRegistration
  OneRecTask --> TwoStageRollout
  MiniOneRecTask --> ConstrainedRollout
  RankGRPOTask --> VanillaVLLM
  TwoStageRollout --> BeamBackend
  ConstrainedRollout --> BeamBackend
  OneRecTask --> OpenOneRecLoop
  MiniOneRecTask --> MiniOneRecLoop
```

## Recipe Integration Notes

- OpenOneRec uses `OneRecTask` to expand rollout counts by beam width, register the
  `two_stage` async rollout path, select `OneRecActorRolloutRefWorker`, and wire
  `OpenOneRecAgentLoopManager`. Its dataset, reward, and task runtime still live
  in `verl_gr/recipes/openonerec/onerec_recipe.py`; validation and checkpoint
  pruning live in `verl_gr/recipes/openonerec/onerec_trainer.py`.
- MiniOneRec uses `MiniOneRecTask` to register `constrained_beam`, select
  `MiniOneRecActorRolloutRefWorker`, and wire `MiniOneRecConstrainedBeamAgentLoopManager`.
  Dataset, reward, format helpers, worker shim, agent loop, and trainer adapter
  are separate recipe modules under `verl_gr/recipes/minionerec`.
- Rank-GRPO keeps rollout on the upstream vanilla `vllm` path. Its recipe code is
  now split across `rankgrpo_dataset.py`, `rankgrpo_task.py`,
  `rankgrpo_algorithm.py`, `rankgrpo_trainer.py`, `rankgrpo_reward.py`, and
  `rankgrpo_tokenizer.py`, while `rankgrpo_recipe.py` remains a compatibility
  export module for existing config overrides. Reward computation is
  reference-aligned through `gt_catalog.pkl`; both validation and advantage
  recomputation use the same catalog-aware per-rank reward path. Validation also
  emits Rank-GRPO-style aliases: `eval/reward`, `eval/reward_total`, and
  `eval/loss`.

## Shared Runtime Flow

- `main_ppo.TaskRunner` resolves a task runtime, calls `prepare(config)`, creates
  train and validation datasets through upstream `create_rl_dataset`, then builds
  `RLTrainer`. Its current in-file registry directly maps `openonerec` and
  `rankgrpo`; `task_factory.py` remains the class-path loader for config-driven
  recipe task construction such as MiniOneRec.
- `RecipeTaskRuntime` centralizes common FSDP wrap-policy cleanup, HuggingFace
  tokenizer/processor creation, worker class selection, and rollout configuration
  hooks. OpenOneRec and MiniOneRec override the rollout hooks; Rank-GRPO overrides
  `prepare` to keep its tokenizer behavior unchanged.
- `RLTrainer` owns shared recommendation generation batch preparation. It delegates
  recipe validation and generation logging through task adapters, and calls
  `rankgrpo_algorithm.compute_rank_grpo_advantage` only when
  `algorithm.rank_grpo.enable` is true. Its local wrapper also adjusts the tqdm
  progress clock around validation and checkpoint saving so evaluation latency
  does not inflate the displayed training-step rate.
- `verl_gr.workers.rollout` contains the reusable beam-search infrastructure:
  registration helpers, two async vLLM server subclasses, rollout adapter classes,
  and the shared async beam backend used by both beam-search recipes.

## Diagram Legend

- Each column is a role in the runtime path, from launch-time task selection to
  rollout execution.
- Arrows show the main construction, delegation, registration, or dependency
  relationships used at runtime.
- Boxes group related classes/modules rather than listing every class method.

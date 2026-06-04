# Scripts

## Rank-GRPO

| Script | Purpose |
| --- | --- |
| `run_rankgrpo.sh` | **Primary entry**: Hydra launcher for Rank-GRPO training. Composes from `configs/verl_gr/rankgrpo/rankgrpo_trainer.yaml`. |
| `.match_rankgrpo.sh` | Node-specific wrapper — starts an isolated Ray cluster per GPU set, derives per-run tags, and invokes `run_rankgrpo.sh`. Untracked; customise for your node. |

### Quick start (via node wrapper)

```bash
cd verl-gr-fork-main

# Minimal launch on GPUs 0,1 (edit CUDA_VISIBLE_DEVICES below for other GPUs):
CUDA_VISIBLE_DEVICES=0,1 \
  bash scripts/.match_rankgrpo.sh
```

The wrapper starts Ray on a per-GPU-set port (GPU 0,1→6380; 2,3→6382; …), cleans up stale processes, and fans out to `run_rankgrpo.sh` with all overrides as CLI Hydra arguments.

### Quick start (standalone — Ray already running)

```bash
cd verl-gr-fork-main

# When Ray is already up (e.g. ray start --head --port=6380):
export RAY_ADDRESS="127.0.0.1:6380"
bash scripts/run_rankgrpo.sh
```

### Key env vars

All `run_rankgrpo.sh` parameters accept overrides via environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0,1` | Which GPUs to use. |
| `N_GPUS` | `2` | Number of GPUs (auto-derived from `CUDA_VISIBLE_DEVICES` by the wrapper). |
| `VERL_GR_ENV` | `…/envs/verl_080` | Conda env providing `python` and `ray`. |
| `DATA_DIR` | `<repo>/../rankgrpo_data_ckpts` | Dataset & checkpoint root. |
| `BASE_MODEL` | `<DATA_DIR>/Qwen2.5-0.5B-Instruct/checkpoint-1500` | HF model path. |
| `OUTPUT_DIR` | `<repo>/outputs/<experiment>` | Checkpoints, tensorboard, logs. |
| `TRAIN_BATCH_SIZE` | `6` | Training batch size. |
| `MAX_TOKENS_PER_GPU` | `49152` | Max tokens per GPU for actor / ref / rollout. |
| `ROLLOUT_N` | `8` | Number of generations per prompt. |
| `REC_NUM` | `20` | Number of recommendations to generate. |
| `LEARNING_RATE` | `1e-6` | Actor learning rate. |
| `KL_LOSS_COEF` | `0.001` | KL penalty coefficient. |
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | `0.15` | vLLM GPU memory fraction for KV cache. |
| `ROLLOUT_FREE_CACHE_ENGINE` | `False` | Whether to sleep vLLM between rollout and training (saves GPU memory but slower). |
| `TOTAL_EPOCHS` | `1` | Number of training epochs. |
| `SAVE_FREQ` / `TEST_FREQ` | `200` | Checkpoint save & validation frequency (steps). |
| `RESUME_MODE` | `auto` | `auto`, `disable`, or `resume_path`. |
| `WANDB_MODE` | `offline` | `online` / `offline` / `disabled`. |
| `LOGGER_BACKENDS` | `[tensorboard]` | Logging backends. |

### Quick smoke test (few steps)

```bash
cd verl-gr-fork-main
bash scripts/run_rankgrpo.sh \
  ++data.train_max_samples=64 \
  ++data.val_max_samples=0 \
  ++trainer.total_epochs=1 \
  ++trainer.save_freq=1000000 \
  ++trainer.test_freq=1000000
```

### Resume from checkpoint

```bash
cd verl-gr-fork-main
RESUME_MODE=resume_path \
  RESUME_FROM_PATH=<output_dir>/ckpt/global_step_<N> \
  bash scripts/.match_rankgrpo.sh
```

The `.match_rankgrpo.sh` wrapper auto-detects the latest checkpoint in `OUTPUT_DIR/ckpt/` and sets `RESUME_MODE=resume_path` when found.

## MiniOneRec GRPO (recommended)

| Script | Purpose |
| --- | --- |
| `run_minionerec_grpo_rl_aligned.sh` | **Primary entry**: 4-GPU DDP GRPO aligned with `MiniOneRec/rl.sh` (lr, KL, beam, batch semantics). |
| `run_minionerec_grpo.sh` | Generic Hydra launcher; set `CONFIG_NAME`, paths, and GPU count. |
| `compare_nsys_nvtx.py` | Compare two `nsys stats --report nvtxsum` CSVs for NVTX range diffs. |

### Quick start (aligned training)

```bash
cd verl-GR
export BASE_MODEL=/path/to/checkpoint
export PYTHON_BIN=/path/to/vllm-gr/bin/python
bash scripts/run_minionerec_grpo_rl_aligned.sh
```

Profiling smoke (limit prompts; `train_max_samples` must be ≥ `TRAIN_BATCH_SIZE`):

```bash
bash scripts/run_minionerec_grpo_rl_aligned.sh \
  ++trainer.total_epochs=1 \
  ++data.train_max_samples=64 \
  ++data.val_max_samples=0 \
  ++trainer.test_freq=1000000 \
  ++trainer.save_freq=1000000
```

### Checkpoint / eval utilities

| Script | Purpose |
| --- | --- |
| `convert_ddp_to_hf.py` | Export DDP actor checkpoint to HuggingFace layout. |
| `merge_fsdp_ckpt.py` | Merge FSDP shards. |
| `eval_compare_ckpts.py` | Compare checkpoints on MiniOneRec-style metrics. |

## Other recipes

| Script | Purpose |
| --- | --- |
| `run_openonerec_grpo.sh` | OpenOneRec two-stage GRPO. |
| `run_rankgrpo.sh` | Rank-GRPO. |
| `run_minionerec_sft.sh` / `run_onerec_sft.sh` | SFT launchers. |
| `eval_sft_minionerec.sh` / `eval_sft_onerec.sh` | SFT eval. |

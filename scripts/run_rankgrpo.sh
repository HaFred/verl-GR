#!/usr/bin/env bash
# Rank-GRPO runtime launcher for verl-GR.
# Fixed config lives in configs/verl_gr/rankgrpo/rankgrpo_trainer.yaml.
# This script only handles what is dynamic (paths, batch calculus, Ray init).

set -euo pipefail

# ---- Environment ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
N_GPUS="${N_GPUS:-2}"

# ---- Paths ----
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "${SCRIPT_DIR}")"
PROJECT_ROOT="$(dirname "${VERL_GR_ROOT}")"
WORKSPACE_ROOT="$(dirname "${PROJECT_ROOT}")"
RANKGRPO_RECIPE_PATH="${VERL_GR_ROOT}/verl_gr/recipes/rankgrpo/rankgrpo_recipe.py"
DEFAULT_VERL_LIB_PATH="${WORKSPACE_ROOT}/verl_080_dev"
if [[ ! -d "${DEFAULT_VERL_LIB_PATH}/verl" ]]; then
  DEFAULT_VERL_LIB_PATH=""
fi
VERL_LIB_PATH="${VERL_LIB_PATH:-${DEFAULT_VERL_LIB_PATH}}"
VERL_GR_ENV="${VERL_GR_ENV:-/home/dyvm6xra/dyvm6xrauser45/miniconda3/envs/verl_080}"
PYTHON_BIN="${PYTHON_BIN:-${VERL_GR_ENV}/bin/python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

# ---- Data & model paths ----
SFT_CHECKPOINT="${SFT_CHECKPOINT:-1500}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/rankgrpo_data_ckpts}"
BASE_MODEL="${BASE_MODEL:-${DATA_DIR}/Qwen2.5-0.5B-Instruct/checkpoint-${SFT_CHECKPOINT}}"
BASE_MODEL_DIRNAME="$(basename "${BASE_MODEL%/}")"
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-${DATA_DIR}/processed_datasets/grpo/grpo_dataset/train}"
VAL_DATASET_DIR="${VAL_DATASET_DIR:-${DATA_DIR}/processed_datasets/sft_dataset/validation}"
GT_CATALOG_PATH="${GT_CATALOG_PATH:-${DATA_DIR}/processed_datasets/gt_catalog.pkl}"
TRAIN_FILES="${TRAIN_FILES:-[${TRAIN_DATASET_DIR}]}"
VAL_FILES="${VAL_FILES:-[${VAL_DATASET_DIR}]}"

# ---- Batch size via gradient accumulation ----
# TRL reference: 4 prompts/GPU × 2 GPUs × 6 accumulation = 48 slots.
# With num_generations=8 → 48/8 = 6 unique prompts per optimizer update.
ROLLOUT_N="${ROLLOUT_N:-8}"
REC_NUM="${REC_NUM:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-6}"
GEN_BATCH_SIZE="$((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
if (( GRADIENT_ACCUMULATION_STEPS > 1 )); then
  USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
  _DEFAULT_MBS_PER_GPU="$((TRAIN_BATCH_SIZE * ROLLOUT_N / N_GPUS))"
  ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-${_DEFAULT_MBS_PER_GPU}}"
else
  USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
  ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-32}"
fi

# ---- Token budget ----
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-24576}"
ACTOR_MAX_TOKENS_PER_GPU="${ACTOR_MAX_TOKENS_PER_GPU:-${MAX_TOKENS_PER_GPU}}"
LOG_PROB_MAX_TOKENS_PER_GPU="${LOG_PROB_MAX_TOKENS_PER_GPU:-${MAX_TOKENS_PER_GPU}}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${MAX_TOKENS_PER_GPU}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$((16 * N_GPUS))}"

# ---- Optimizer (betas differ from dp_actor default [0.9,0.999]) ----
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.99}"

# ---- Output & experiment ----
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${BASE_MODEL_DIRNAME}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${VERL_GR_ROOT}/outputs/${EXPERIMENT_NAME}}"
RESUME_MODE="${RESUME_MODE:-auto}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
if [[ "${RESUME_MODE}" == "resume_path" ]]; then
  if [[ -z "${RESUME_FROM_PATH}" || ! -d "${RESUME_FROM_PATH}" ]]; then
    echo "Error: RESUME_FROM_PATH does not exist: ${RESUME_FROM_PATH}" >&2
    exit 2
  fi
fi

# ---- Ray cluster args (only for fresh local cluster) ----
RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR:-/tmp}/vr_${USER:-u}_$(printf '%s_%s' "${EXPERIMENT_NAME}" "${CUDA_VISIBLE_DEVICES}" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-16)}"
RAY_TMPDIR_FALLBACK_ROOT="${RAY_TMPDIR_FALLBACK_ROOT:-${TMPDIR:-/tmp}}"
RAY_TMPDIR_MAX_LEN="${RAY_TMPDIR_MAX_LEN:-60}"
if (( ${#RAY_TMPDIR} > RAY_TMPDIR_MAX_LEN )); then
  SHORT_USER="${USER:-user}"
  SHORT_TAG="$(printf '%s_%s' "${EXPERIMENT_NAME}" "${CUDA_VISIBLE_DEVICES}" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-24)"
  RAY_TMPDIR="${RAY_TMPDIR_FALLBACK_ROOT}/vr_${SHORT_USER}_${SHORT_TAG}"
  echo "Warning: RAY_TMPDIR path too long, fallback to ${RAY_TMPDIR}" >&2
fi
RAY_SPILL_DIR="${RAY_SPILL_DIR:-${RAY_TMPDIR}/spill}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-$((N_GPUS * 24))}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-$((N_GPUS * 32 * 1024 * 1024 * 1024))}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-False}"

# ---- Prepare output ----
mkdir -p "${OUTPUT_DIR}" "${RAY_TMPDIR}" "${RAY_SPILL_DIR}"
if [[ -n "${VERL_LIB_PATH}" ]]; then
  export PYTHONPATH="${VERL_GR_ROOT}:${VERL_LIB_PATH}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${VERL_GR_ROOT}:${PYTHONPATH:-}"
fi
export RAY_TMPDIR TMPDIR="${RAY_TMPDIR}"

echo "==================================="
echo "Rank-GRPO (verl-GR runtime)"
echo "==================================="
echo "Cluster: 1 node(s) x ${N_GPUS} GPU(s)"
echo "Model: ${BASE_MODEL}"
echo "Train files: ${TRAIN_FILES}"
echo "Val files: ${VAL_FILES}"
echo "GT catalog: ${GT_CATALOG_PATH}"
echo "Train batch: ${TRAIN_BATCH_SIZE} (gen: ${GEN_BATCH_SIZE}, accum: ${GRADIENT_ACCUMULATION_STEPS})"
echo "Max tokens/GPU: ${MAX_TOKENS_PER_GPU}"
echo "Rollout N: ${ROLLOUT_N}   Rec num: ${REC_NUM}"
echo "Output: ${OUTPUT_DIR}"
echo "Resume mode: ${RESUME_MODE}"
if [[ -n "${VERL_LIB_PATH}" ]]; then
  echo "verl library path: ${VERL_LIB_PATH}"
fi
echo "==================================="

# ---- Launch ----
"${PYTHON_BIN}" -u -m verl_gr.trainers.main_ppo \
  --config-path "${VERL_GR_ROOT}/configs/verl_gr/rankgrpo" \
  --config-name rankgrpo_trainer \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  ++data.gen_batch_size="${GEN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.custom_cls.path="${RANKGRPO_RECIPE_PATH}" \
  custom_reward_function.path="${RANKGRPO_RECIPE_PATH}" \
  custom_reward_function.reward_kwargs.gt_catalog_path="${GT_CATALOG_PATH}" \
  data.rankgrpo.rec_num="${REC_NUM}" \
  algorithm.rank_grpo.rec_num="${REC_NUM}" \
  algorithm.rank_grpo.gt_catalog_path="${GT_CATALOG_PATH}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  ++actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  ++actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.optim.betas="[${ADAM_BETA1},${ADAM_BETA2}]" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKENS_PER_GPU}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.model.path="${BASE_MODEL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}/ckpt" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
  global_profiler.save_path="${OUTPUT_DIR}/profiles" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WORKER_MULTIPROC_METHOD="'${VLLM_WORKER_MULTIPROC_METHOD}'" \
  $(  # Ray cluster-creation args — only for fresh local cluster
    if [[ -z "${RAY_ADDRESS:-}" ]]; then
      echo "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS}"
      echo "+ray_kwargs.ray_init.object_store_memory=${RAY_OBJECT_STORE_MEMORY}"
      echo "+ray_kwargs.ray_init.include_dashboard=${RAY_INCLUDE_DASHBOARD}"
      echo "+ray_kwargs.ray_init._temp_dir=${RAY_TMPDIR}"
      echo "+ray_kwargs.ray_init.object_spilling_directory=${RAY_SPILL_DIR}"
    fi
  ) \
  "$@"

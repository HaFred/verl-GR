#!/usr/bin/env bash
# Re-run Rank-GRPO validation from a saved checkpoint and print eval/reward_total.
# Uses the same launcher path as training (.match_rankgrpo.sh / run_rankgrpo.sh).
#
# Usage:
#   CKPT_PATH=outputs/debug_june4_2nd/ckpt/global_step_5400 \
#     bash scripts/misc/eval_ckpt_reward_total.sh
#
# Optional: EXPECTED_REWARD_TOTAL=0.4595 to assert against TB reference.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
VERL_GR_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"
cd "${VERL_GR_ROOT}"

CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH to a global_step_* checkpoint directory}"
if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "Error: CKPT_PATH does not exist: ${CKPT_PATH}" >&2
  exit 2
fi
if [[ "${CKPT_PATH}" != /* ]]; then
  CKPT_PATH="${VERL_GR_ROOT}/${CKPT_PATH}"
fi

STEP="$(basename "${CKPT_PATH}")"
STEP="${STEP#global_step_}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_reward_total_${STEP}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${VERL_GR_ROOT}/outputs/${EXPERIMENT_NAME}}"
LOG_FILE="${LOG_FILE:-${VERL_GR_ROOT}/logs/${EXPERIMENT_NAME}.log}"
mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export VERL_GR_ENV="${VERL_GR_ENV:-/home/dyvm6xra/dyvm6xrauser45/miniconda3/envs/verl_080}"
export VERL_LIB_PATH="${VERL_LIB_PATH:-/home/dyvm6xra/dyvm6xrauser45/fred/verl_080_dev}"
export VERL_GR_CONVERGENCE_GATE="${VERL_GR_CONVERGENCE_GATE:-0}"
export VERL_GR_KL_GROWTH_GATE="${VERL_GR_KL_GROWTH_GATE:-0}"
unset RAY_ADDRESS

# debug_june4_2nd / .match_rankgrpo.sh defaults
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-6}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
export ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-24576}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.15}"
export FSDP_STRATEGY="${FSDP_STRATEGY:-fsdp2}"
export ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-2}"
export N_GPUS="${N_GPUS:-2}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-1600}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
export RESUME_MODE=resume_path
export RESUME_FROM_PATH="${CKPT_PATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export LOGGER_BACKENDS="${LOGGER_BACKENDS:-[console]}"

echo "CKPT_PATH=${CKPT_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"

bash scripts/run_rankgrpo.sh \
  trainer.val_only=true \
  trainer.val_before_train=true \
  trainer.total_epochs=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.default_local_dir="${OUTPUT_DIR}/ckpt" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  actor_rollout_ref.actor.checkpoint.load_contents='[model]' \
  "$@" 2>&1 | tee "${LOG_FILE}"

REWARD_TOTAL="$(
  "${VERL_GR_ENV}/bin/python" - <<'PY' "${LOG_FILE}"
import re, sys
log = open(sys.argv[1], encoding="utf-8", errors="replace").read()
# Hydra pprint dict in Ray worker stdout
m = re.findall(r"'eval/reward_total':\s*([0-9.]+)", log)
if not m:
    m = re.findall(r'"eval/reward_total":\s*([0-9.]+)', log)
if not m:
    print("", end="")
else:
    print(m[-1], end="")
PY
)"

if [[ -z "${REWARD_TOTAL}" ]]; then
  echo "Could not parse eval/reward_total from ${LOG_FILE}" >&2
  exit 1
fi

echo "eval/reward_total=${REWARD_TOTAL} (checkpoint step ${STEP})"

if [[ -n "${EXPECTED_REWARD_TOTAL:-}" ]]; then
  "${VERL_GR_ENV}/bin/python" - <<PY
expected = float("${EXPECTED_REWARD_TOTAL}")
got = float("${REWARD_TOTAL}")
tol = float("${REWARD_TOTAL_TOL:-0.002}")
diff = abs(got - expected)
print(f"expected={expected} got={got} diff={diff} tol={tol}")
if diff > tol:
    raise SystemExit(f"eval/reward_total mismatch: {got} vs expected {expected}")
print("PASS: within tolerance")
PY
fi

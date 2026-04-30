# Add post-prepare two-stage rollout sanity logging/assertion
# Add train-time candidate grouping diagnostics
# Add legacy-comparable KL/PG and logprob distribution metrics

# export EXPERIMENT_NAME=2gpu_diagnostics_ttc_lossklpg_legacy_expansion1_32
export EXPERIMENT_NAME=4gpu_notthinking
export N_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=2,3
# export OUTPUT_DIR=/scratch/dyvm6xra/dyvm6xrauser45/fred/run_outputs/4gpu_correctvllm

export OUTPUT_DIR=/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/verl-gr-fork-workingbranch/outputs/4gpu_notthinking
export TENSORBOARD_DIR=/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/verl-gr-fork-workingbranch/tensorboard_logs/${EXPERIMENT_NAME}
export RAY_TMPDIR=$OUTPUT_DIR/ray_tmp
export RAY_SPILL_DIR=$RAY_TMPDIR/spill

export ROLLOUT_N=1
export BASE_MODEL=/scratch/dyvm6xra/dyvm6xrauser45/fred/models--OpenOneRec--OneRec-1.7B-pretrain/snapshots/db455d0bdcf4b5e0b42f30c45d65260a49656a7f
export DATA_DIR=/home/dyvm6xra/dyvm6xrauser45/fred/openonerec_fredfork/data
clear

bash scripts/run_openonerec_grpo.sh \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.resume_mode=disable \
  data.shuffle=True \
  data.train_max_samples=20000 \
  data.val_max_samples=-1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=true \
  trainer.log_val_generations=4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.top_k=50 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.custom.beam_width=32 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
  actor_rollout_ref.rollout.max_num_batched_tokens=12288 \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=true

  # data.train_max_samples=5000 \
  # trainer.test_freq=20 \
  # trainer.save_freq=100 \
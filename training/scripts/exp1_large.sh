#!/usr/bin/env bash
#SBATCH --job-name=llm-calib
#SBATCH --array=0-9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G
#SBATCH --time=23:59:00
#SBATCH --partition=pli-c
#SBATCH --output=logs/calibration_%A_%a.out
#SBATCH --mail-type=fail
#SBATCH --mail-user=jc93@princeton.edu

set -xeuo pipefail

mkdir -p logs

source .venv/bin/activate
module load proxy/default
export WANDB_MODE=online
export VLLM_USE_V1=1

REWARD_FN_PATH=/scratch/gpfs/DANQIC/jeff/llm-calibration/training/eval_utils.py
DATA_DIR=/scratch/gpfs/DANQIC/jeff/llm-calibration/data/math

TRAIN_BATCH_SIZE=256
N_ROLLOUTS=16
ACTOR_MICRO_BSZ_PER_GPU=2
ROLLOUT_MICRO_BSZ_PER_GPU=2

# shared GRPO hyperparams
NORM_ADV=True
USE_KL=True;  KL_COEF=0.0005
CLIP_LOW=0.2; CLIP_HIGH=0.2; CLIP_C=10.0
LOSS_AGG=token-mean
ENTROPY_COEFF=0.0

# ── model selection ──────────────────────────────────────────────────────────
# Tasks 0-4: llama3.1-8b-instruct
# Tasks 5-9: olmo3-7b-instruct
#
# Olmo's attention/recompute path triggers
#   torch.utils.checkpoint.CheckpointError: Recomputed values for the following
#   tensors have different metadata than during the forward pass.
# Workaround: disable gradient checkpointing for olmo. Drop micro-bsz to 1 to
# compensate for the higher activation memory.
if [ "$SLURM_ARRAY_TASK_ID" -lt 5 ]; then
    MODEL_PATH=/scratch/gpfs/DANQIC/jeff/models/llama3.1-8b-instruct
    MODEL_TAG=llama3.1-8b
    SUBTASK=$SLURM_ARRAY_TASK_ID
    ENABLE_GRAD_CKPT=true
else
    MODEL_PATH=/scratch/gpfs/DANQIC/jeff/models/olmo3-7b-instruct
    MODEL_TAG=olmo3-7b
    SUBTASK=$((SLURM_ARRAY_TASK_ID - 5))
    ENABLE_GRAD_CKPT=false
    ACTOR_MICRO_BSZ_PER_GPU=1
fi

# ── per-run config ──────────────────────────────────────────────────────────
case $SUBTASK in
0)
    # Task 1: Standard GRPO — correctness-only reward, no wager prompt
    EXP_NAME=grpo_standard_${MODEL_TAG}_exp1_large
    OUTPUT_DIR=/scratch/gpfs/DANQIC/jeff/llm-calibration/outputs/exp1_large/${MODEL_TAG}/standard_baseline
    TRAIN_FILES=${DATA_DIR}/math_standard_train.parquet
    VAL_FILES=${DATA_DIR}/math_standard_val.parquet
    REWARD_FN=grpo_reward_standard
    SUPPR_COEF=0.0
    CALIB_COEF=0.0
    CALIB_K=1
    SUPPR_DECAY=0
    CALIB_DECAY=0
    PPO_MINI_BSZ=$((TRAIN_BATCH_SIZE / 2))
    ;;
1)
    # Task 2: Wager GRPO baseline — proper scoring rule reward, no aux losses
    EXP_NAME=grpo_wager_${MODEL_TAG}_exp1_large
    OUTPUT_DIR=/scratch/gpfs/DANQIC/jeff/llm-calibration/outputs/exp1_large/${MODEL_TAG}/wager_baseline
    TRAIN_FILES=${DATA_DIR}/math_wager_detailed_train.parquet
    VAL_FILES=${DATA_DIR}/math_wager_detailed_val.parquet
    REWARD_FN=grpo_reward_wager
    SUPPR_COEF=0.0
    CALIB_COEF=0.0
    CALIB_K=1
    SUPPR_DECAY=0
    CALIB_DECAY=0
    PPO_MINI_BSZ=$((TRAIN_BATCH_SIZE / 2))
    ;;
2)
    # Task 3: Wager + calib loss, decays to 0 by step 30
    EXP_NAME=grpo_wager_calib_decay30_${MODEL_TAG}_exp1_large
    OUTPUT_DIR=/scratch/gpfs/DANQIC/jeff/llm-calibration/outputs/exp1_large/${MODEL_TAG}/calib_loss_decay30
    TRAIN_FILES=${DATA_DIR}/math_wager_detailed_train.parquet
    VAL_FILES=${DATA_DIR}/math_wager_detailed_val.parquet
    REWARD_FN=grpo_reward_wager
    SUPPR_COEF=0.0
    CALIB_COEF=0.001
    CALIB_K=5
    SUPPR_DECAY=0
    CALIB_DECAY=30
    PPO_MINI_BSZ=$((TRAIN_BATCH_SIZE / 2))
    ;;
3)
    # Task 4: Wager + suppr loss, no decay
    EXP_NAME=grpo_wager_suppr_${MODEL_TAG}_exp1_large
    OUTPUT_DIR=/scratch/gpfs/DANQIC/jeff/llm-calibration/outputs/exp1_large/${MODEL_TAG}/suppr_loss
    TRAIN_FILES=${DATA_DIR}/math_wager_detailed_train.parquet
    VAL_FILES=${DATA_DIR}/math_wager_detailed_val.parquet
    REWARD_FN=grpo_reward_wager
    SUPPR_COEF=0.01
    CALIB_COEF=0.0
    CALIB_K=1
    SUPPR_DECAY=0
    CALIB_DECAY=0
    PPO_MINI_BSZ=$((TRAIN_BATCH_SIZE / 2))
    ;;
4)
    # Task 5: Wager + suppr (no decay) + calib (decays to 0 by step 30)
    EXP_NAME=grpo_wager_comb_calibdecay30_${MODEL_TAG}_exp1_large
    OUTPUT_DIR=/scratch/gpfs/DANQIC/jeff/llm-calibration/outputs/exp1_large/${MODEL_TAG}/comb_calibdecay30
    TRAIN_FILES=${DATA_DIR}/math_wager_detailed_train.parquet
    VAL_FILES=${DATA_DIR}/math_wager_detailed_val.parquet
    REWARD_FN=grpo_reward_wager
    SUPPR_COEF=0.01
    CALIB_COEF=0.001
    CALIB_K=5
    SUPPR_DECAY=0
    CALIB_DECAY=30
    PPO_MINI_BSZ=$((TRAIN_BATCH_SIZE / 2))
    ;;
*)
    echo "Unknown SUBTASK: $SUBTASK (SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID)"
    exit 1
    ;;
esac

# ── test mode overrides ─────────────────────────────────────────────────────
TEST=${TEST:-0}
if [ "$TEST" = "1" ]; then
    TRAIN_BATCH_SIZE=16
    N_ROLLOUTS=4
    PPO_MINI_BSZ=$((TRAIN_BATCH_SIZE / 2))
    VAL_N=2
    LOG_VAL_GEN=1
    LOGGER='["console"]'
    SAVE_FREQ=-1
else
    VAL_N=32
    LOG_VAL_GEN=16
    LOGGER='["console","wandb"]'
    SAVE_FREQ=10
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=1280 \
    data.max_response_length=8192 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRAD_CKPT} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_HIGH} \
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_C} \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.suppr_coef=${SUPPR_COEF} \
    actor_rollout_ref.actor.suppr_coef_decay_steps=${SUPPR_DECAY} \
    actor_rollout_ref.actor.calib_coef=${CALIB_COEF} \
    actor_rollout_ref.actor.calib_coef_decay_steps=${CALIB_DECAY} \
    actor_rollout_ref.actor.calib_k=${CALIB_K} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.skip_tokenizer_init=True \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ROLLOUT_MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name=${REWARD_FN} \
    algorithm.use_kl_in_reward=${USE_KL} \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    trainer.critic_warmup=0 \
    trainer.logger="${LOGGER}" \
    trainer.project_name=llm-calibration \
    trainer.experiment_name=${EXP_NAME} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.total_training_steps=100 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=10 \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.validation_data_dir="${OUTPUT_DIR}/val_generations" \
    trainer.log_val_generations=${LOG_VAL_GEN} \
    trainer.device=cuda \
    ray_kwargs.ray_init.num_cpus=32

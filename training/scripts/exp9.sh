#!/usr/bin/env bash
#SBATCH --job-name=llm-calib-exp9
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G
#SBATCH --time=23:59:00
#SBATCH --partition=pli-c
#SBATCH --output=logs/calibration_exp9_%A_%a.out
#SBATCH --mail-type=fail
#SBATCH --mail-user=jc93@princeton.edu

set -xeuo pipefail

mkdir -p logs

source .venv/bin/activate
module load proxy/default
export WANDB_MODE=online
export VLLM_USE_V1=1

MODEL_PATH=/scratch/gpfs/DANQIC/jeff/models/llama3.2-3b-instruct
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

# ── per-run config ──────────────────────────────────────────────────────────
# CoT-sketch variant: model writes up to 100 words then declares Confidence: X
TRAIN_FILES=${DATA_DIR}/math_wager_cot_train.parquet
VAL_FILES=${DATA_DIR}/math_wager_cot_val.parquet
REWARD_FN=grpo_reward_wager_cot
CALIB_K=5
CALIB_FILTERING=-1
PPO_MINI_BSZ=$((TRAIN_BATCH_SIZE / 2))

case $SLURM_ARRAY_TASK_ID in
0)
    # pure RL baseline — no auxiliary losses
    EXP_NAME=grpo_wager_cot_rl_llama3.2-3b_exp9
    OUTPUT_DIR=/scratch/gpfs/DANQIC/jeff/llm-calibration/outputs/exp9/rl_only
    LR=2e-6
    SUPPR_COEF=0.0; CALIB_COEF=0.0
    SUPPR_DECAY=0;  CALIB_DECAY=0
    ;;
1)
    # comb: best config from exp7 (calib + suppr, LR=2e-6)
    EXP_NAME=grpo_wager_cot_comb_llama3.2-3b_exp9
    OUTPUT_DIR=/scratch/gpfs/DANQIC/jeff/llm-calibration/outputs/exp9/comb
    LR=2e-6
    SUPPR_COEF=0.02; CALIB_COEF=0.002
    SUPPR_DECAY=0;   CALIB_DECAY=0
    ;;
*)
    echo "Unknown SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
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
    actor_rollout_ref.actor.optim.lr=${LR} \
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
    actor_rollout_ref.actor.calib_filtering_steps=${CALIB_FILTERING} \
    actor_rollout_ref.actor.calib_k=${CALIB_K} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
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
    trainer.total_training_steps=150 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=10 \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.validation_data_dir="${OUTPUT_DIR}/val_generations" \
    trainer.log_val_generations=${LOG_VAL_GEN} \
    trainer.device=cuda \
    ray_kwargs.ray_init.num_cpus=32

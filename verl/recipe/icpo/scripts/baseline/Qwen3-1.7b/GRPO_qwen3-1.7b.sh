set -xeuo pipefail

cd ./verl
HOME="YOUR_HOME_DIR"
STORAGE="YOUR_STORAGE_DIR"

export WANDB_API_KEY=""

project_name='ICPO'

n_rollout=8
max_prompt_length=4096
max_response_length=8192
max_token_len_per_gpu=$((${n_rollout}*${max_prompt_length}+${max_response_length}))

exp_name=GRPO_Qwen3-1.7B_${max_response_length}_temp-1.0_openr1

# [Train Settings]
train_data_shuffle=True
train_batch_size=128
val_batch_size=1024
mini_batch_size=64
micro_batch_size_per_gpu=2
n_gpus_per_node=8
tensor_model_parallel_size=1

temperature=1.0
clip_ratio_high=0.2
clip_ratio_low=0.2
entropy_coeff=0

# [Val Settings]
val_n=2
val_temperature=0.6
val_top_p=0.7
val_top_k=50
val_do_sample=True

model_path=$STORAGE/model/Qwen/Qwen3-1.7B

train_path=$HOME/data/openr1_math_220k-8192/train_instruct.parquet

math500_test_path=$HOME/data/verl_eval/math500/test.parquet
amc23_test_path=$HOME/data/verl_eval/amc23/test.parquet
aime24_test_path=$HOME/data/verl_eval/aime24/test.parquet

test_paths="['$math500_test_path', '$amc23_test_path', '$aime24_test_path']"

python3 -c "
import pandas as pd
df = pd.read_parquet('$train_path')
print(len(df))
print(df.iloc[0]['prompt'])
"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_path \
    data.val_files="$test_paths" \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.shuffle=${train_data_shuffle} \
    data.truncation='error' \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${n_rollout} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((${max_prompt_length} + ${max_response_length})) \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.val_before_train=True \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.default_local_dir=$HOME/checkpoints/$project_name/$exp_name \
    trainer.total_training_steps=400 \
    trainer.total_epochs=20 $@
# set -x


CKPTS_DIR=${1:-"YOUR_CHECKPOINT_PATH/global_step_XXX/actor"}

CUR_CHECKPOINT="${CKPTS_DIR}/hf"
    if [ ! -d "$CUR_CHECKPOINT" ]; then
        echo "$CUR_CHECKPOINT does not exist. Running model merger..."
        LOCAL_DIR="${CKPTS_DIR}"
        TARGET_DIR="${LOCAL_DIR}/hf"
        python -m verl.model_merger merge --backend fsdp --local_dir $LOCAL_DIR --target_dir $TARGET_DIR
    fi
    echo "CUR_CHECKPOINT: $CUR_CHECKPOINT"



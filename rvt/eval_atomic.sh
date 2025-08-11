#!/bin/bash

# === Default ===
DEMO_PATH_ROOT="/data1/cyt/HiMan_data/test_atomic"
TF_CPP_MIN_LOG_LEVEL=3

# === Usage Function ===
usage() {
  echo "Usage: bash eval_atomic.sh --epoch <epoch> --model_folder <path> --device <device>"
  exit 1
}

# === Argument Parsing ===
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --epoch) epoch="$2"; shift ;;
        --model_folder) model_folder="$2"; shift ;;
        --device) device="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# === Check Required Params ===
if [[ -z "$epoch" || -z "$model_folder" ||  -z "$device" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# === Evaluation Tasks ===
root_tasks=(
    "box_in_cupboard"
    "box_out_of_opened_drawer"
    "close_drawer"
    "put_in_opened_drawer"
    "sweep_to_dustpan"
    "box_out_of_cupboard"
    "broom_out_of_cupboard"
    "open_drawer"
    "rubbish_in_dustpan"
    "take_out_of_opened_drawer"
)

# === Evaluation Loop ===
for root_task in "${root_tasks[@]}"; do
    for i in {0..17}; do
        task_name="${root_task}_${i}"
        DATA_PATH="${DEMO_PATH_ROOT}/${task_name}/"
        
        if [ ! -d "$DATA_PATH" ]; then
            echo "[Skip] $DATA_PATH does not exist."
            continue
        fi

        cmd_args=(
            uv run eval.py
            --model-folder "$model_folder"
            --eval-datafolder "$DEMO_PATH_ROOT"
            --tasks "$task_name"
            --eval-episodes 1
            --log-name "epoch_${epoch}"
            --device "$device"
            --headless
            --model-name "model_${epoch}.pth"
            --save-video
            --tasks_type "atomic"
        )

        echo "[Run] Evaluating $task_name ..."
        "${cmd_args[@]}"
    done
done
#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
    echo "usage: $0 <weight|tea|capsule|pot> [gpu_num] [global_batch_size] [cache_workers]" >&2
    exit 2
fi

TASK_NAME=$1
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DEFAULT_EXP_NAME="task_eps_bidir_openpi"

case "${TASK_NAME}" in
    weight)
        DEFAULT_DATA_FILE="${ROOT}/data/weight/generated_dataset.hdf5"
        CONFIG_NAME="score_task_weight"
        REPO_NAME="cn356/isaaclab_weight"
        PROMPT="put pear and apple on the scale"
        DEFAULT_CACHE_DIR="${ROOT}/data/weight/score_task_weight.observations"
        ;;
    tea)
        DEFAULT_DATA_FILE="${ROOT}/data/tea/new_generated_dataset_50.hdf5"
        CONFIG_NAME="score_task_tea"
        REPO_NAME="cn356/isaaclab_tea"
        PROMPT="pour the tea from the teapot into the cup"
        DEFAULT_CACHE_DIR="${ROOT}/data/tea/score_task_tea.observations"
        ;;
    capsule)
        DEFAULT_DATA_FILE="${ROOT}/data/capsule/generated_dataset.hdf5"
        CONFIG_NAME="score_task_capsule"
        REPO_NAME="cn356/isaaclab_capsule"
        PROMPT="open the coffee maker lid and put the pod inside"
        DEFAULT_CACHE_DIR="${ROOT}/data/capsule/score_task_capsule.observations"
        ;;
    pot)
        DEFAULT_DATA_FILE="${ROOT}/data/pot/generated_dataset.hdf5"
        CONFIG_NAME="score_task_pot"
        REPO_NAME="cn356/isaaclab_pot"
        PROMPT="remove the lid of the pot and put egg in it"
        DEFAULT_CACHE_DIR="${ROOT}/data/pot/score_task_pot.observations"
        ;;
    *)
        echo "unknown score task: ${TASK_NAME}; expected weight, tea, capsule, or pot" >&2
        exit 2
        ;;
esac

DATA_FILE="${DATA_FILE:-${DEFAULT_DATA_FILE}}"
DATASET_DIR="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/${REPO_NAME}"
TASK_CACHE_DIR="${TASK_CACHE_DIR:-${DEFAULT_CACHE_DIR}}"
EXP_NAME="${EXP_NAME:-${DEFAULT_EXP_NAME}}"
CONDA_ENV="${CONDA_ENV:-pps}"
GPU_NUM="${1:-${GPU_NUM:-1}}"
BATCH_SIZE="${2:-${BATCH_SIZE:-32}}"
if [[ -n "${3:-}" ]]; then
    NUM_WORKERS=$3
elif [[ -z "${NUM_WORKERS:-}" ]]; then
    if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
        threads_per_rank=$((SLURM_CPUS_PER_TASK / GPU_NUM))
        NUM_WORKERS=$((threads_per_rank - ${OMP_NUM_THREADS:-1}))
        (( NUM_WORKERS > 24 )) && NUM_WORKERS=24
        (( NUM_WORKERS < 1 )) && NUM_WORKERS=1
    else
        NUM_WORKERS=8
    fi
fi

if (( $# > 3 )); then
    echo "usage: $0 <weight|tea|capsule|pot> [gpu_num] [global_batch_size] [cache_workers]" >&2
    exit 2
fi
for value_name in GPU_NUM BATCH_SIZE NUM_WORKERS; do
    value=${!value_name}
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
done
if (( BATCH_SIZE % GPU_NUM != 0 )); then
    echo "BATCH_SIZE=${BATCH_SIZE} must be divisible by GPU_NUM=${GPU_NUM}" >&2
    exit 2
fi

train_launcher=(python)
if (( GPU_NUM > 1 )); then
    train_launcher=(
        torchrun
        --standalone
        --nnodes=1
        --nproc_per_node="${GPU_NUM}"
    )
fi

cd "${ROOT}/openpi"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

WANDB_ENV="${ROOT}/.secrets/wandb.env"
if [[ -f "${WANDB_ENV}" ]]; then
    set -a
    source "${WANDB_ENV}"
    set +a
fi

if [[ ! -f "${DATASET_DIR}/meta/info.json" ]]; then
    if [[ ! -f "${DATA_FILE}" ]]; then
        echo "task dataset is missing: ${DATA_FILE}" >&2
        echo "set DATA_FILE to the annotated HDF5 dataset for ${TASK_NAME}" >&2
        exit 1
    fi
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n "${CONDA_ENV}" \
        python examples/Isaaclab/convert_isaaclab_data_to_lerobot.py \
        --data-file "${DATA_FILE}" \
        --repo-name "${REPO_NAME}" \
        --prompt "${PROMPT}"
fi

mode=()
if [[ -n "${TRAIN_MODE:-}" ]]; then
    case "${TRAIN_MODE}" in
        --resume|--overwrite) mode+=("${TRAIN_MODE}") ;;
        *)
            echo "TRAIN_MODE must be --resume or --overwrite, got: ${TRAIN_MODE}" >&2
            exit 2
            ;;
    esac
elif compgen -G "checkpoints/${CONFIG_NAME}/${EXP_NAME}/[0-9]*" >/dev/null; then
    mode+=(--resume)
else
    mode+=(--overwrite)
fi

printf '[score_task] task=%s config=%s exp=%s mode=%s\n' \
    "${TASK_NAME}" "${CONFIG_NAME}" "${EXP_NAME}" "${mode[0]}"
echo "[score_task] semantics=epsilon,bidirectional,language,openpi-gemma,no-legacy-scale"
echo "[score_task] shared training cache=${TASK_CACHE_DIR}"
echo "[score_task] cache-build workers=${NUM_WORKERS}; mmap training workers/rank=0"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n "${CONDA_ENV}" \
    python scripts/train_proxy_score_pytorch.py prepare-task-cache \
    --config "${CONFIG_NAME}" \
    --cache-path "${TASK_CACHE_DIR}" \
    --num-workers "${NUM_WORKERS}"
export SCORE_TASK_CACHE_PATH="${TASK_CACHE_DIR}"

echo "[score_task] global batch=${BATCH_SIZE}, per-GPU batch=$((BATCH_SIZE / GPU_NUM)), GPUs=${GPU_NUM}"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n "${CONDA_ENV}" \
    "${train_launcher[@]}" scripts/train_proxy_score_pytorch.py \
    "${CONFIG_NAME}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 0 \
    --exp_name "${EXP_NAME}" \
    "${mode[@]}"

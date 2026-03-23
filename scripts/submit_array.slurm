#!/bin/bash
# One array task = one (site, year) line in task_list.txt (1-based line index).
# Adjust --array, --mem, --time, and paths for your cluster.
#
# BLAS thread caps (below) avoid oversubscription when PyMC runs 4 chains in parallel.
# Override at submit time if needed, e.g. OMP_NUM_THREADS=2 sbatch ...
#
# Usage:
#   cp task_list.example.txt task_list.txt   # edit lines: SITE YEAR per line
#   mkdir -p logs
#   sbatch scripts/submit_array.slurm

#SBATCH --job-name=neon_sr
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/neon_%A_%a.out
#SBATCH --error=logs/neon_%A_%a.err
## Match line count in task_list.txt (inclusive range, one line per task):
#SBATCH --array=1-6

set -euo pipefail

export DATA_DIR="${SLURM_SUBMIT_DIR}/data"
export OUTPUT_DIR="${SLURM_SUBMIT_DIR}/results"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

SEED="${SEED:-$((42 + SLURM_ARRAY_TASK_ID))}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

cd "$REPO_ROOT"
mkdir -p logs

TASKS_FILE="${TASKS_FILE:-${REPO_ROOT}/task_list.txt}"
if [[ ! -f "$TASKS_FILE" ]]; then
  echo "Missing task file: $TASKS_FILE (copy from task_list.example.txt)" >&2
  exit 1
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASKS_FILE")
if [[ -z "${LINE// }" ]]; then
  echo "Empty task line ${SLURM_ARRAY_TASK_ID} in $TASKS_FILE" >&2
  exit 1
fi

read -r SITE YEAR <<< "$LINE"

module load micromamba
micromamba activate pymc_py311

python Data_process.py \
      --site "$SITE" \
      --year "$YEAR" \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --seed "$SEED" \
      --cores 4

#!/bin/bash
# =============================================================================
#  ot_match.sh  –  three-phase SLURM submission for large-file OT matching
#
#  Phase 0 – single job : fit transformations on a small sample, save pickle
#  Phase 1 – array job  : each task loads pickle, runs OT on one chunk,
#                         saves a per-chunk HDF5 file
#  Phase 2 – single job : merge all chunk HDF5 files into the data file,
#                         delete chunks and pickle
#
#  Usage:
#    bash ot_match.sh
#    CHUNK_SIZE=2000 TOTAL_SAMPLES=130000 bash ot_match.sh
# =============================================================================

###
#  sbatch \
#   --job-name=ot_merge \
#   --mem=60G \
#   --cpus-per-task=4 \
#   --time=02:00:00 \
#   --partition=serial_requeue \
#   --output=/n/home04/hhanif/AllShowers/logs/ot_merge_%j.out \
#   --error=/n/home04/hhanif/AllShowers/logs/ot_merge_%j.err \
#   --wrap="
#     module load python
#     eval \"\$(mamba shell hook --shell bash)\"
#     mamba config set changeps1 False
#     mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env

#     python /n/home04/hhanif/AllShowers/allshowers/OT_match.py \
#       /n/home04/hhanif/AllShowers/conf/muons.yaml \
#       --merge \
#       --chunk-size 1000 \
#       --pkl-path /n/home04/hhanif/AllShowers/logs/preprocessor.pkl \
#       --with-time
#   "
###

# --------------------------------------------------------------------------- #
#  User-configurable settings                                                  #
# --------------------------------------------------------------------------- #
CONFIG=/n/home04/hhanif/AllShowers/conf/muons.yaml
PYTHON_SCRIPT=/n/home04/hhanif/AllShowers/allshowers/OT_match.py
CONDA_ENV=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env
LOG_DIR=/n/home04/hhanif/AllShowers/logs

# Number of samples per chunk (= per Phase 1 array task)
CHUNK_SIZE=${CHUNK_SIZE:-1000}

# Total number of samples in the data file.
# python -c "import showerdata; print(showerdata.get_file_shape('...')[0])"
TOTAL_SAMPLES=${TOTAL_SAMPLES:-130000}

# Where the fitted PreProcessor pickle is saved (shared across all phases)
PKL_PATH=${PKL_PATH:-/n/home04/hhanif/AllShowers/logs/preprocessor.pkl}

# Optional: override chunk output directory (default: <data_file>_chunks/)
# OUTPUT_DIR=/n/scratch/hhanif/ot_chunks

# Extra flags forwarded to all phases (e.g. "--with-time")
EXTRA_FLAGS="--with-time"
# --------------------------------------------------------------------------- #

mkdir -p "$LOG_DIR"

# Compute number of chunks (ceiling division)
NUM_CHUNKS=$(( (TOTAL_SAMPLES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
LAST_CHUNK=$(( NUM_CHUNKS - 1 ))

echo "=================================================="
echo "  Config          : $CONFIG"
echo "  Pickle path      : $PKL_PATH"
echo "  Chunk size       : $CHUNK_SIZE"
echo "  Total samples    : $TOTAL_SAMPLES"
echo "  Number of chunks : $NUM_CHUNKS  (array indices 0-${LAST_CHUNK})"
echo "=================================================="

# --------------------------------------------------------------------------- #
#  Phase 0 – fit transformations, save pickle                                 #
# --------------------------------------------------------------------------- #
PHASE0_JOB_ID=$(sbatch --parsable \
  --job-name=ot_fit \
  --mem=800G \
  --cpus-per-task=1 \
  --time=00:30:00 \
  --partition=serial_requeue \
  --output="${LOG_DIR}/ot_fit_%j.out" \
  --error="${LOG_DIR}/ot_fit_%j.err" \
  --wrap="
    module load python
    eval \"\$(mamba shell hook --shell bash)\"
    mamba config set changeps1 False
    mamba activate ${CONDA_ENV}

    python ${PYTHON_SCRIPT} ${CONFIG} \
      --fit-only \
      --pkl-path ${PKL_PATH} \
      ${EXTRA_FLAGS}
  "
)

echo "Submitted Phase 0 (fit)   : ${PHASE0_JOB_ID}"

# --------------------------------------------------------------------------- #
#  Phase 1 – array job, one task per chunk (waits for Phase 0)                #
# --------------------------------------------------------------------------- #
PHASE1_JOB_ID=$(sbatch --parsable \
  --job-name=ot_chunk \
  --mem=32G \
  --cpus-per-task=8 \
  --time=01:00:00 \
  --partition=serial_requeue \
  --array="0-${LAST_CHUNK}" \
  --dependency="afterok:${PHASE0_JOB_ID}" \
  --output="${LOG_DIR}/ot_chunk_%A_%a.out" \
  --error="${LOG_DIR}/ot_chunk_%A_%a.err" \
  --wrap="
    module load python
    eval \"\$(mamba shell hook --shell bash)\"
    mamba config set changeps1 False
    mamba activate ${CONDA_ENV}

    python ${PYTHON_SCRIPT} ${CONFIG} \
      --big-file \
      --chunk-size ${CHUNK_SIZE} \
      --chunk-index \$SLURM_ARRAY_TASK_ID \
      --pkl-path ${PKL_PATH} \
      ${EXTRA_FLAGS}
  "
)

echo "Submitted Phase 1 (chunks): ${PHASE1_JOB_ID}  (tasks 0–${LAST_CHUNK}, depends on ${PHASE0_JOB_ID})"

# --------------------------------------------------------------------------- #
#  Phase 2 – merge job (waits for all Phase 1 tasks)                          #
# --------------------------------------------------------------------------- #
PHASE2_JOB_ID=$(sbatch --parsable \
  --job-name=ot_merge \
  --mem=60G \
  --cpus-per-task=4 \
  --time=02:00:00 \
  --partition=serial_requeue \
  --dependency="afterok:${PHASE1_JOB_ID}" \
  --output="${LOG_DIR}/ot_merge_%j.out" \
  --error="${LOG_DIR}/ot_merge_%j.err" \
  --wrap="
    module load python
    eval \"\$(mamba shell hook --shell bash)\"
    mamba config set changeps1 False
    mamba activate ${CONDA_ENV}

    python ${PYTHON_SCRIPT} ${CONFIG} \
      --merge \
      --chunk-size ${CHUNK_SIZE} \
      --pkl-path ${PKL_PATH} \
      ${EXTRA_FLAGS}
  "
)

echo "Submitted Phase 2 (merge) : ${PHASE2_JOB_ID}  (depends on ${PHASE1_JOB_ID})"
echo ""
echo "Monitor with:"
echo "  squeue -j ${PHASE0_JOB_ID},${PHASE1_JOB_ID},${PHASE2_JOB_ID}"
echo "  tail -f ${LOG_DIR}/ot_fit_${PHASE0_JOB_ID}.out"
echo "  tail -f ${LOG_DIR}/ot_chunk_${PHASE1_JOB_ID}_0.out"
echo "  tail -f ${LOG_DIR}/ot_merge_${PHASE2_JOB_ID}.out"
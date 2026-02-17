#!/bin/bash
#SBATCH --job-name=ga_bootstrap
#SBATCH --output=logs/fold${FOLD_INDEX}_gen${GEN}_boot%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=15G
#SBATCH --time=4:00:00

if [ "$SERVER" == "spartan" ]; then
  module load Apptainer
  cd /data/gpfs/projects/punim1993/students/Jente/multiclust
elif [ "$SERVER" == "marvin" ]; then
  cd /home/s45jmeij_hpc/multiclust
fi

BASE_DIR=${BASE_DIR:-$(pwd)}
export BASE_DIR
mkdir -p "${BASE_DIR}/logs"

echo ">>> bootstrap_generation.sh: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  FOLD_INDEX=${FOLD_INDEX:-<unset>}  GEN=${GEN:-<unset>}"

BIDX=${SLURM_ARRAY_TASK_ID}
GA_ROOT=${GA_ROOT:-"${BASE_DIR}/intermediates/fold${FOLD_INDEX}/ga"}
OUTDIR=${GA_ROOT}/gen${GEN}/bootstrap_${BIDX}
mkdir -p ${OUTDIR}

# Record start time
START_TIME=$(date +%s)
echo "=== [Fold ${FOLD_INDEX}] Bootstrap ${BIDX} started at $(date -Is) ==="

# Use all allocated CPUs for parallel clustering
export N_JOBS=${SLURM_CPUS_PER_TASK:-1}

POP_IN=${GA_ROOT}/population_fold${FOLD_INDEX}_gen${GEN}.pkl
POP_INIT=${POP_INIT:-"${GA_ROOT}/population_init_fold${FOLD_INDEX}.pkl"}
LAB_OUT=${GA_ROOT}/gen${GEN}/bootstrap_${BIDX}/labels_${BIDX}.pkl

# retry logic for if one bootstrap fails. 
MAX_RETRIES=3
RETRY_COUNT=0
EXIT_CODE=1

while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ $EXIT_CODE -ne 0 ]; do
  ((RETRY_COUNT++))
  echo "=== [Fold ${FOLD_INDEX}] Bootstrap ${BIDX} attempt $RETRY_COUNT at $(date -Is) ==="
  apptainer exec ${SIF} \
    python -u full_pipeline.py \
      --mode bootstrap \
      --input_csv "${INPUT_CSV}" \
      --meta_csv "${META_CSV}" \
      --fold_index "${FOLD_INDEX}" \
      --generation ${GEN} \
      --population_dir "${GA_ROOT}/gen${GEN}" \
      --population_file "${POP_IN}" \
      --population_initial_file "${POP_INIT}" \
      --bootstrap_index ${BIDX} \
      --n_bootstrap ${N_BOOTSTRAP} \
      --bootstrap_mode ${BOOTSTRAP_MODE} \
      --n_folds "${N_FOLDS}" \
      --col_threshold "${COL_THRESHOLD}" \
      --row_threshold "${ROW_THRESHOLD}" \
      --skew_threshold "${SKEW_THRESHOLD}" \
      --scaler_type "${SCALER_TYPE}" \
      --modalities ${MODALITIES} \
      --dim_reduction "${DIMREDUCTION}" \
      --hidden_dims ${HIDDEN_DIMS} \
      --activation_functions ${ACTIVATION_FUNCTIONS} \
      --learning_rates ${LEARNING_RATES} \
      --batch_sizes ${BATCH_SIZES} \
      --latent_dims ${LATENT_DIMS} \
      --optimisation "${OPTIMISATION}" \
      --ga_objectives ${GA_OBJECTIVES} \
      --fusion_methods ${FUSION_METHODS} \
      --n_jobs "${N_JOBS}" \
      --output_labels "${LAB_OUT}" \
      --mincluster ${MINCLUSTER} \
      --mincluster_n ${MINCLUSTER_N} \
      --TEST "${TEST}" \
      --base_dir "${BASE_DIR}" \
    | tee "${BASE_DIR}/logs/fold${FOLD_INDEX}_gen${GEN}_boot${BIDX}.log"
  EXIT_CODE=${PIPESTATUS[0]}
  if [ $EXIT_CODE -ne 0 ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
    echo "!!! Bootstrap ${BIDX} failed (exit $EXIT_CODE), retrying ..."
  fi
done

# If failed after all retries, print a message
if [ $EXIT_CODE -ne 0 ]; then
  echo "!!! [Fold ${FOLD_INDEX}] Bootstrap ${BIDX} failed after ${MAX_RETRIES} attempts at $(date -Is) !!!"
fi


# Record end time and compute duration
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== [Fold ${FOLD_INDEX}] Bootstrap ${BIDX} finished at $(date -Is) (Duration: ${ELAPSED} seconds) ==="

exit $EXIT_CODE

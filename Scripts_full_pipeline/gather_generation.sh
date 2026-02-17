#!/bin/bash
#SBATCH --job-name=ga_gather
#SBATCH --output=logs/fold${FOLD_INDEX}_gen${GEN}_gather.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00

if [ "$SERVER" == "spartan" ]; then
  module load Apptainer
  cd /data/gpfs/projects/punim1993/students/Jente/multiclust
elif [ "$SERVER" == "marvin" ]; then
  cd /home/s45jmeij_hpc/multiclust
fi

BASE_DIR=${BASE_DIR:-$(pwd)}
export BASE_DIR
mkdir -p "${BASE_DIR}/logs"

GA_ROOT=${GA_ROOT:-"${BASE_DIR}/intermediates/fold${FOLD_INDEX}/ga"}
BOOT_DIR=${GA_ROOT}/gen${GEN}

POP_IN=${GA_ROOT}/population_fold${FOLD_INDEX}_gen${GEN}.pkl
POP_OUT=${GA_ROOT}/population_fold${FOLD_INDEX}_gen$((GEN+1)).pkl
POP_INIT=${POP_INIT:-"${GA_ROOT}/population_init_fold${FOLD_INDEX}.pkl"}

# Use all allocated CPUs for parallelisation
N_JOBS=${SLURM_CPUS_PER_TASK:-1}

apptainer exec ${SIF} \
  python -u full_pipeline.py \
    --mode gather \
    --input_csv          "${INPUT_CSV}" \
    --meta_csv           "${META_CSV}" \
    --fold_index         "${FOLD_INDEX}" \
    --generation         "${GEN}" \
    --bootstrap_dir      "${BOOT_DIR}" \
    --population_dir     "${GA_ROOT}/gen${GEN}" \
    --population_file    "${POP_IN}" \
    --population_initial_file "${POP_INIT}" \
    --n_folds            "${N_FOLDS}" \
    --n_bootstrap       "${N_BOOTSTRAP}" \
    --col_threshold      "${COL_THRESHOLD}" \
    --row_threshold      "${ROW_THRESHOLD}" \
    --skew_threshold     "${SKEW_THRESHOLD}" \
    --scaler_type        "${SCALER_TYPE}" \
    --modalities         ${MODALITIES} \
    --dim_reduction      "${DIMREDUCTION}" \
    --hidden_dims        ${HIDDEN_DIMS} \
    --activation_functions ${ACTIVATION_FUNCTIONS} \
    --learning_rates     ${LEARNING_RATES} \
    --batch_sizes        ${BATCH_SIZES} \
    --latent_dims        ${LATENT_DIMS} \
    --optimisation       "${OPTIMISATION}" \
    --ga_objectives      ${GA_OBJECTIVES} \
    --fusion_methods     ${FUSION_METHODS} \
    --output_population  "${POP_OUT}" \
    --n_jobs             "${N_JOBS}" \
    --mincluster         ${MINCLUSTER} \
    --mincluster_n       ${MINCLUSTER_N} \
    --TEST               "${TEST}" \
    --ga_cxpb            "${GA_CXPB}" \
    --ga_mutpb           "${GA_MUTPB}" \
    --ga_elitism         "${GA_ELITISM}" \
    --base_dir           "${BASE_DIR}" \
  | tee -a "${BASE_DIR}/logs/fold${FOLD_INDEX}_gen${GEN}_gather.log"


exit ${PIPESTATUS[0]}

#!/bin/bash
#SBATCH --job-name=postprocess
#SBATCH --output=logs/fold${FOLD_INDEX}_postprocess.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=10G
#SBATCH --time=02:00:00

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
RESULTS_DIR=${BASE_DIR}/results/fold${FOLD_INDEX}
mkdir -p ${RESULTS_DIR}

POP_FIN=${GA_ROOT}/population_fold${FOLD_INDEX}_gen${N_GENERATIONS}.pkl

apptainer exec ${SIF} \
  python -u full_pipeline.py \
    --mode outer \
    --fold_index "${FOLD_INDEX}" \
    --n_folds "${N_FOLDS}" \
    --n_bootstrap "${N_BOOTSTRAP}" \
    --population_file "${POP_FIN}" \
    --input_csv "${INPUT_CSV}" \
    --meta_csv "${META_CSV}" \
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
    --output_metrics "${RESULTS_DIR}/metrics.pkl" \
    --mincluster ${MINCLUSTER} \
    --mincluster_n ${MINCLUSTER_N} \
    --TEST "${TEST}" \
    --DO_SVM "${DO_SVM}" \
    --base_dir "${BASE_DIR}" \
    | tee -a "${BASE_DIR}/logs/fold${FOLD_INDEX}_postprocess.log"

exit ${PIPESTATUS[0]}

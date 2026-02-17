#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --output=logs/merge.log
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

FINAL_RESULTS_DIR=${BASE_DIR}/results/final/
mkdir -p ${FINAL_RESULTS_DIR}

apptainer exec ${SIF} \
  python -u full_pipeline.py \
    --mode merge \
    --base_dir "${BASE_DIR}" \
    --n_folds "${N_FOLDS}" \
    --n_bootstrap "${N_BOOTSTRAP}" \
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
    --mincluster ${MINCLUSTER} \
    --mincluster_n ${MINCLUSTER_N} \
    --output_final_metrics "${FINAL_RESULTS_DIR}/final_metrics.pkl" \
    --TEST "${TEST}" \
    --DO_SVM "${DO_SVM}"
exit ${PIPESTATUS[0]}

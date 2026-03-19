#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --output=logs/merge.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=64G
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

SIF=${SIF:-multiview_env.sif}
if [ ! -f "${SIF}" ] && [ -f "${BASE_DIR}/${SIF}" ]; then
  SIF="${BASE_DIR}/${SIF}"
fi
if [ ! -f "${SIF}" ]; then
  echo "ERROR: Apptainer image not found: ${SIF}" >&2
  echo "Set SIF to a valid .sif path or place multiview_env.sif in ${BASE_DIR}" >&2
  exit 2
fi
export SIF
echo "Using Apptainer image: ${SIF}"

FINAL_RESULTS_DIR=${BASE_DIR}/results/final/
mkdir -p ${FINAL_RESULTS_DIR}

COMPUTE_CLUSTER_PVALUES=${COMPUTE_CLUSTER_PVALUES:-FALSE}
CLUSTER_PVALUE_MODE=${CLUSTER_PVALUE_MODE:-fast}
CLUSTER_PVALUE_STAT=${CLUSTER_PVALUE_STAT:-composite}
CLUSTER_PVALUE_PERMUTATIONS=${CLUSTER_PVALUE_PERMUTATIONS:-200}
CLUSTER_PVALUE_PERMUTATIONS_QUALITY=${CLUSTER_PVALUE_PERMUTATIONS_QUALITY:-}
CLUSTER_PVALUE_PERMUTATIONS_ARI=${CLUSTER_PVALUE_PERMUTATIONS_ARI:-}
CLUSTER_PVALUE_JOBS=${CLUSTER_PVALUE_JOBS:-0}
CLUSTER_PVALUE_SEED=${CLUSTER_PVALUE_SEED:-314159}
N_JOBS=${SLURM_CPUS_PER_TASK:-1}

PVAL_EXTRA_ARGS=()
if [ -n "${CLUSTER_PVALUE_PERMUTATIONS_QUALITY}" ]; then
  PVAL_EXTRA_ARGS+=(--cluster_pvalue_permutations_quality "${CLUSTER_PVALUE_PERMUTATIONS_QUALITY}")
fi
if [ -n "${CLUSTER_PVALUE_PERMUTATIONS_ARI}" ]; then
  PVAL_EXTRA_ARGS+=(--cluster_pvalue_permutations_ari "${CLUSTER_PVALUE_PERMUTATIONS_ARI}")
fi

apptainer exec "${SIF}" \
  python -u full_pipeline.py \
    --mode merge \
    --base_dir "${BASE_DIR}" \
    --n_folds "${N_FOLDS}" \
    --n_jobs "${N_JOBS}" \
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
    --DO_SVM "${DO_SVM}" \
    --compute_cluster_pvalues "${COMPUTE_CLUSTER_PVALUES}" \
    --cluster_pvalue_mode "${CLUSTER_PVALUE_MODE}" \
    --cluster_pvalue_stat "${CLUSTER_PVALUE_STAT}" \
    --cluster_pvalue_permutations "${CLUSTER_PVALUE_PERMUTATIONS}" \
    "${PVAL_EXTRA_ARGS[@]}" \
    --cluster_pvalue_jobs "${CLUSTER_PVALUE_JOBS}" \
    --cluster_pvalue_seed "${CLUSTER_PVALUE_SEED}"
exit ${PIPESTATUS[0]}

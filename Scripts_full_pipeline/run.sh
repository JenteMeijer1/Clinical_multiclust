#!/bin/bash
#SBATCH --job-name=run        # run everything          
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00

 # Detect server if not explicitly set in the environment
if [ -z "${SERVER:-}" ]; then
  hn=$(hostname -f 2>/dev/null || hostname)
  case "$hn" in
    *marvin*|*hpc.uni-bonn.de*) SERVER="marvin" ;;
    *spartan*|*unimelb*|*melbourne*) SERVER="spartan" ;;
    *) SERVER="marvin" ;; # fallback
  esac
fi
export SERVER

# Partition selection (Marvin defaults to a 1h devel queue; use short for longer jobs)
export PARTITION_OPT=""
if [ "$SERVER" == "marvin" ]; then
  PARTITION_OPT="-p intelsr_short"
fi
if [ "$SERVER" == "spartan" ]; then
  module load Apptainer
  cd /data/gpfs/projects/punim1993/students/Jente/multiclust
elif [ "$SERVER" == "marvin" ]; then
  cd /home/s45jmeij_hpc/multiclust
fi


# Apptainer image with all dependencies
export SIF=multiview_env.sif 
# Import data
#export INPUT_CSV="synthetic_multimodal_spartan.csv" #For synthetic data in testing
export INPUT_CSV="cleaned_discovery_data.csv" #actual data
#export META_CSV="synthetic_multimodal_spartan_meta.csv" #For synthetic data in testing
export META_CSV="merged_meta.csv" #Actual data

if [ "$SERVER" == "spartan" ]; then
  export BASE_DIR="/data/gpfs/projects/punim1993/students/Jente/multiclust" # Base directory for all results and intermediates
elif [ "$SERVER" == "marvin" ]; then
  export BASE_DIR="/home/s45jmeij_hpc/multiclust" # Base directory for all results and intermediates
fi
mkdir -p "${BASE_DIR}/logs"

####### Pipeline parameters ########

# Number of outer CV folds
export N_FOLDS=5
# Threshold for removing columns with too many missing values
export COL_THRESHOLD=0.5 
# Threshold for removing rows with too many missing values
export ROW_THRESHOLD=0.5 
# Threshold for when to log-transform a variable based on its skewness
export SKEW_THRESHOLD=0.75 
# Type of scaler to use: "standard", "minmax", "robust"
export SCALER_TYPE="robust" 
# Actual modalities in the data
export MODALITIES="Internalising Functioning Detachment Psychoticism Cognition" 
#export MODALITIES="m1 m2 m3 m4" # For synthetic test
# Selection of dimensionality reduction method:urrently supported: None, VAE, PCA, AE
export DIMREDUCTION="None" 
# Hidden layer dimensions to try in VAE
export HIDDEN_DIMS="100 250 500 1000" 
# Activation functions to try in VAE
export ACTIVATION_FUNCTIONS="LeakyReLU selu swish" 
# Learning rates to try in VAE
export LEARNING_RATES="0.001 0.0001" 
# Batch sizes to try in VAE
export BATCH_SIZES="32 64 128" 
# Latent dimensions to try in VAE
export LATENT_DIMS="2 5 10 20" 
# Optimisation objective: "single" or "multi" (multi-objective). Multi = optimising both cluster quality and stability for each modality and final clusters.
export OPTIMISATION="multi" 
# Number of hyperparameter combinations in each generation
export N_POPULATION=100
#Number of GA generations
export N_GENERATIONS=10
# Minimum number of clusters tested (for both individual modalities and final clusters)
export K_MIN=2 
# Maximum number of clusters tested (for both individual modalities and final clusters)
export K_MAX=10 
# Which linkages to use in clustering (options: single, complete, average, ward, weighted, default=average). Multiple are allowed for GA optimisation
export CLUSTER_LINKAGES="average complete weighted"
# Space of clustering. Euclidian is default but for ordinal or categorical data, other metrics can be used. Currently supported: Euclidean, 
# Number of bootstraps per generation in which stability is tested
export N_BOOTSTRAP=100
# Bootstrap modes. Options: 'bootstrap' (with replacement) or 'subsample' (without replacement).
export BOOTSTRAP_MODE='subsample' 
# Maximum number of concurrent bootstrap jobs running on server
export MAX_CONCURRENT=200
# Whether to run SVM classification on the final clustering labels in OUTER mode (TRUE/FALSE)
export DO_SVM="TRUE" 
# Whether to enforce a minimum cluster size of 10 in the final clustering step.
export MINCLUSTER="FALSE" 
# Minimum cluster size when enforcing minimum cluster size
export MINCLUSTER_N=1
# Objectives optimised by the GA (order matters). Allowed tokens defined in full_pipeline.py. An example of different options are: min or mean. Example: "avg_view_stability avg_view_quality final_stability final_quality" 
# For the stability measures you need to add which stability metric you want to use. So after stability, you can add: _jaccard, _coassoc, _ccc or _ari
export GA_OBJECTIVES="mean_view_stability_ari mean_view_quality final_stability_ari final_quality"
# Fusion methods allowed during GA search / mutation. Current options: agreement, disagreement, consensus (which is a strict consensus)
export FUSION_METHODS="consensus"
# Set to "TRUE" for testing mode (tests against true labels); "FALSE" for full run
export TEST="FALSE" 
#Crossover rate in GA
export GA_CXPB=0.7 
#Mutation rate in GA
export GA_MUTPB=0.3 
#Number of elite individuals to keep in next generation in GA.
export GA_ELITISM=2 
# Wether to run the final merge or not 
export RUN_MERGE="TRUE"

TEST_phase=0 # Set to 0 for full run or no testing

# Ensure we’ve created the initial GA population
N_POPULATION=${N_POPULATION}
K_MIN=${K_MIN}
K_MAX=${K_MAX}
MODALITIES=${MODALITIES}


########################################
# 1) Schedule outer CV folds
########################################

 # Optional resume controls (0-indexed fold indices)
# Example: RESUME_FROM_FOLD=8 RESUME_TO_FOLD=9 to run only folds 8 and 9.
RESUME_FROM_FOLD=${RESUME_FROM_FOLD:-0}
RESUME_TO_FOLD=${RESUME_TO_FOLD:-$((N_FOLDS-1))}
export RESUME_FROM_FOLD RESUME_TO_FOLD

post_ids=() #Store the job ID of postprocess for each fold

for OUTER_FOLD in $(seq 1 $N_FOLDS); do

  export OUTER_FOLD
  FOLD_INDEX=$((OUTER_FOLD-1))
  export FOLD_INDEX
  if [ ${FOLD_INDEX} -lt ${RESUME_FROM_FOLD} ] || [ ${FOLD_INDEX} -gt ${RESUME_TO_FOLD} ]; then
    echo "[Fold ${FOLD_INDEX}] Skipping (resume range ${RESUME_FROM_FOLD}-${RESUME_TO_FOLD})"
    continue
  fi
  GA_ROOT="${BASE_DIR}/intermediates/fold${FOLD_INDEX}/ga"
  INIT_POP="${GA_ROOT}/population_init_fold${FOLD_INDEX}.pkl"
  POP_INIT=${INIT_POP}
  export GA_ROOT
  export POP_INIT

  ########################################
  # --- Test phases that do not need full pipeline ---
  # Expect: TEST is "TRUE" or "FALSE"; TEST_phase is 0..4
  if [[ "$TEST" == "TRUE" ]]; then
    case "$TEST_phase" in
      1)
        echo "=== Test phase 1: only running single method (KMeans) ==="
        apptainer exec "$SIF" python full_pipeline.py --mode test1 \
            --input_csv          "${INPUT_CSV}" \
            --meta_csv           "${META_CSV}" \
            --fold_index         "${FOLD_INDEX}" \
            --n_folds            "${N_FOLDS}" \
            --col_threshold      "${COL_THRESHOLD}" \
            --row_threshold      "${ROW_THRESHOLD}" \
            --skew_threshold     "${SKEW_THRESHOLD}" \
            --scaler_type        "${SCALER_TYPE}" \
            --modalities         ${MODALITIES} \
            --base_dir           "${BASE_DIR}" \
            --TEST               "${TEST}" 
        exit 0
        ;;
      2)
        echo "=== Test phase 2: only running single method (Spectral) ==="
        apptainer exec "$SIF" python full_pipeline.py --mode test2 \
            --input_csv          "${INPUT_CSV}" \
            --meta_csv           "${META_CSV}" \
            --fold_index         "${FOLD_INDEX}" \
            --n_folds            "${N_FOLDS}" \
            --col_threshold      "${COL_THRESHOLD}" \
            --row_threshold      "${ROW_THRESHOLD}" \
            --skew_threshold     "${SKEW_THRESHOLD}" \
            --scaler_type        "${SCALER_TYPE}" \
            --modalities         ${MODALITIES} \
            --base_dir           "${BASE_DIR}" \
            --TEST               "${TEST}" 
        exit 0
        ;;
      3)
        echo "=== Test phase 3: Verify fusion matrices and individual ensemble cluster ==="
        apptainer exec "$SIF" python full_pipeline.py --mode test3 \
            --input_csv          "${INPUT_CSV}" \
            --meta_csv           "${META_CSV}" \
            --fold_index         "${FOLD_INDEX}" \
            --n_folds            "${N_FOLDS}" \
            --col_threshold      "${COL_THRESHOLD}" \
            --row_threshold      "${ROW_THRESHOLD}" \
            --skew_threshold     "${SKEW_THRESHOLD}" \
            --scaler_type        "${SCALER_TYPE}" \
            --modalities         ${MODALITIES} \
            --base_dir           "${BASE_DIR}" \
            --TEST               "${TEST}" 
        exit 0
        ;;
      4)
        echo "=== Test phase 4: Final clustering correctness on fusion matrix ==="
        apptainer exec "$SIF" python full_pipeline.py --mode test4 \
            --input_csv          "${INPUT_CSV}" \
            --meta_csv           "${META_CSV}" \
            --fold_index         "${FOLD_INDEX}" \
            --n_folds            "${N_FOLDS}" \
            --col_threshold      "${COL_THRESHOLD}" \
            --row_threshold      "${ROW_THRESHOLD}" \
            --skew_threshold     "${SKEW_THRESHOLD}" \
            --scaler_type        "${SCALER_TYPE}" \
            --modalities         ${MODALITIES} \
            --base_dir           "${BASE_DIR}" \
            --TEST               "${TEST}" 
        exit 0
        ;;
    esac
  fi


  ########################################

  echo "[Fold ${FOLD_INDEX}] Initializing GA population…"
  apptainer exec ${SIF} \
      python full_pipeline.py --mode init \
          --population_file "${INIT_POP}" \
          --n_population ${N_POPULATION} \
          --k_min ${K_MIN} --k_max ${K_MAX} \
          --linkages ${CLUSTER_LINKAGES} \
          --ga_objectives ${GA_OBJECTIVES} \
          --fusion_methods ${FUSION_METHODS} \
          --modalities ${MODALITIES} \
          --base_dir "${BASE_DIR}" \
          --seed ${OUTER_FOLD} \

  # Chain GA generations so each bootstrap waits on the previous gather
  prev_gather_id=""

  # ————————————————————————————————————————

  echo "=== Starting outer fold ${FOLD_INDEX} ==="

  # GA settings
  N_GENERATION=${N_GENERATIONS}
  N_BOOTSTRAPS=${N_BOOTSTRAP}
  MAX_CONCURRENT=${MAX_CONCURRENT}

  # initial population (you must generate this once, e.g. with a small helper script;
  # it should be a pickled list of DEAP Individuals)
  POP_INIT=${INIT_POP}
  export POP_INIT


  ########################################
  # 1) Nested‐GA
  ########################################
  for GEN in $(seq 1 $N_GENERATIONS); do
    # Export current generation for child jobs
    export GEN
    echo "[Fold $FOLD_INDEX] Generation $GEN: launching bootstrap array..."
    if [ -z "$prev_gather_id" ]; then
      array_id=$(sbatch ${PARTITION_OPT} --parsable \
        --export=ALL,FOLD_INDEX,GEN,POP_INIT,DIMREDUCTION,MINCLUSTER,N_BOOTSTRAP,BOOTSTRAP_MODE,GA_CXPB,GA_MUTPB,GA_ELITISM \
        --array=1-${N_BOOTSTRAPS}%${MAX_CONCURRENT} \
        bootstrap_generation.sh)
    else
      array_id=$(sbatch ${PARTITION_OPT} --parsable \
        --dependency=afterok:${prev_gather_id} \
        --export=ALL,FOLD_INDEX,GEN,POP_INIT,DIMREDUCTION,MINCLUSTER,N_BOOTSTRAP,BOOTSTRAP_MODE,GA_CXPB,GA_MUTPB,GA_ELITISM \
        --array=1-${N_BOOTSTRAPS}%${MAX_CONCURRENT} \
        bootstrap_generation.sh)
    fi
    echo "  -> array job ID: $array_id"
    if [ -z "$array_id" ]; then
      echo "ERROR: bootstrap array submission failed for fold ${FOLD_INDEX} gen ${GEN}. Aborting to avoid broken dependencies." >&2
      exit 1
    fi

    echo "[Fold $FOLD_INDEX] Generation $GEN: launching gather job..."
    gather_id=$(sbatch ${PARTITION_OPT} --parsable \
      --dependency=afterok:${array_id} \
      --export=ALL,FOLD_INDEX,GEN,POP_INIT,DIMREDUCTION,MINCLUSTER,N_BOOTSTRAP,BOOTSTRAP_MODE,GA_CXPB,GA_MUTPB,GA_ELITISM \
      gather_generation.sh)
    echo "  -> gather job ID: $gather_id"
    if [ -z "$gather_id" ]; then
      echo "ERROR: gather job submission failed for fold ${FOLD_INDEX} gen ${GEN}. Aborting to avoid broken dependencies." >&2
      exit 1
    fi

    prev_gather_id=$gather_id

    # now point to the newly‐evolved population for the next iteration
    POP_IN=${GA_ROOT}/population_fold${FOLD_INDEX}_gen$((GEN+1)).pkl
    export POP_IN
  done

  ########################################
  # 2) When generation $N_GENERATIONS finishes, run post‐processing (AE + Parea)
  ########################################

  echo "[Fold $FOLD_INDEX] Scheduling final post‐processing..."
  outer_id=$(sbatch ${PARTITION_OPT} --parsable --dependency=afterok:${gather_id} \
    --export=ALL,FOLD_INDEX,POP_IN,DIMREDUCTION,ASSEMBLE_HYBRID,N_BOOTSTRAP,DO_SVM,MINCLUSTER \
    postprocess_outer.sh)

  post_ids+=("${outer_id}")
done



########################################
# 2) When outer fold finishes, run final merge and SVM classification
########################################
dep_string=$(IFS=:; echo "${post_ids[*]}")

if [ "${RUN_MERGE}" == "FALSE" ]; then
  echo "Final merge disabled; skipping scheduling of final merge job."
  exit 0

elif [ "${RUN_MERGE}" == "TRUE" ]; then
  echo "Scheduling final merge and SVM classification job..."
  sbatch ${PARTITION_OPT} --dependency=afterok:${dep_string} \
    --export=ALL,DO_SVM,N_BOOTSTRAP,MINCLUSTER \
    final_merge.sh
  exit 0
fi


#sbatch --export=ALL,DO_SVM,N_BOOTSTRAP,MINCLUSTER final_merge.sh



  

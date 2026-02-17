#!/usr/bin/env python
import os

# Limit BLAS/Numexpr threading to avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# full_pipeline.py — Updated to support nested SLURM pipelines

# --- Imports ------------------------------------------------
from Utils import *
import time
import pandas as pd
import re
import dill
import argparse
from itertools import combinations
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from joblib import Parallel, delayed
import warnings

import numpy as np
import os
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from operator import itemgetter
from deap import base, creator, tools, algorithms

# --- SciPy imports for cophenetic correlation calculation ---
import scipy.cluster.hierarchy as hierarchy
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

from sklearn.model_selection import KFold
# SVM imports retained but SVM mode will be commented out
from sklearn.svm import SVC
from sklearn.cluster import KMeans, SpectralClustering
import random
import torch.nn as nn
import torch
torch.set_num_threads(1)
import sys
import gc
from sklearn.decomposition import PCA
from collections import defaultdict
import pickle

# Import own functions
from VAE import run_VAE_complete
from AE  import run_AE_complete
from parea_functions import *
from parea_functions import convert_to_parameters
from SVM import *
import glob

GA_OBJECTIVE_ALIASES = {
    # --- Quality objectives (unchanged semantics) ---
    "av_qual_view": "mean_view_quality",
    "avg_qual_view": "mean_view_quality",
    "mean_view_quality": "mean_view_quality",
    "qual_final": "final_quality",
    "final_quality": "final_quality",
    "min_qual_view": "min_view_quality",
    "min_view_quality": "min_view_quality",

    # --- ARI-based stability (default) ---
    "av_stab_view": "mean_view_stability_ari",
    "avg_stab_view": "mean_view_stability_ari",
    "mean_view_stability": "mean_view_stability_ari",
    "mean_view_stability_ari": "mean_view_stability_ari",
    "mean_view_stab": "mean_view_stability_ari",
    "mean_view_stab_ari": "mean_view_stability_ari",

    "stab_final": "final_stability_ari",
    "final_stability": "final_stability_ari",
    "final_stability_ari": "final_stability_ari",

    "min_stab_view": "min_view_stability_ari",
    "min_view_stability": "min_view_stability_ari",
    "min_view_stability_ari": "min_view_stability_ari",

    # --- Co-association-based stability ---
    "mean_view_stability_coassoc": "mean_view_stability_coassoc",
    "final_stability_coassoc": "final_stability_coassoc",

    # --- Jaccard-based stability ---
    "mean_view_stability_jaccard": "mean_view_stability_jaccard",
    "final_stability_jaccard": "final_stability_jaccard",

    # --- CCC-based stability ---
    "mean_view_stability_ccc": "mean_view_stability_CCC",
    "mean_view_stability_CCC": "mean_view_stability_CCC",
    "final_stability_ccc": "final_stability_CCC",
    "final_stability_CCC": "final_stability_CCC",
}
DEFAULT_GA_OBJECTIVES = [
    "mean_view_stability_ari",
    "mean_view_quality",
    "final_stability_ari",
    "final_quality",
]
DEFAULT_FUSION_METHODS = ["agreement", "consensus", "disagreement"]


def _normalize_method_list(values):
    out = []
    for val in values or []:
        if isinstance(val, str) and "," in val:
            parts = [p for p in val.split(",") if p]
        else:
            parts = [val]
        for part in parts:
            if part is None:
                continue
            text = str(part).strip().lower()
            if text:
                out.append(text)
    return out



def _normalize_objective_tokens(raw_tokens, optimisation_mode):
    """Map user-specified GA objective tokens to canonical names."""
    if not raw_tokens:
        # Defaults: single-objective -> final-stability (ARI); multi -> standard multi-objective set.
        if optimisation_mode == "single":
            tokens = ["final_stability_ari"]
        else:
            tokens = DEFAULT_GA_OBJECTIVES
    else:
        tokens = []
        for tok in raw_tokens:
            if isinstance(tok, str) and "," in tok:
                tokens.extend([t for t in tok.split(",") if t])
            else:
                tokens.append(tok)

    normalized = []
    for tok in tokens:
        key = str(tok).strip().lower()
        if key not in GA_OBJECTIVE_ALIASES:
            valid = ", ".join(sorted(set(GA_OBJECTIVE_ALIASES.values())))
            raise ValueError(f"Unknown GA objective '{tok}'. Valid options: {valid}")
        normalized.append(GA_OBJECTIVE_ALIASES[key])
    return normalized


# Helper to choose the primary metric keys for summary attributes based on GA objectives
def _primary_metric_keys(args):
    """
    Decide which summary keys to use for the convenience attributes
    (mean_view_stab, final_stab, etc.) based on args.ga_objectives.

    Returns
    -------
    stab_view_key, stab_final_key, qual_view_key, qual_final_key
    """
    objs = list(getattr(args, "ga_objectives", []))

    stab_view_key = None
    stab_final_key = None

    for obj in objs:
        if isinstance(obj, str):
            if obj.startswith("mean_view_stability") and stab_view_key is None:
                stab_view_key = obj
            if obj.startswith("final_stability") and stab_final_key is None:
                stab_final_key = obj

    # Fallbacks: default to ARI-based if not explicitly in objectives
    if stab_view_key is None:
        stab_view_key = "mean_view_stability_ari"
    if stab_final_key is None:
        stab_final_key = "final_stability_ari"

    # Quality keys are fixed names in `summary`
    qual_view_key = "mean_view_quality"
    qual_final_key = "final_quality"

    return stab_view_key, stab_final_key, qual_view_key, qual_final_key


def _ensure_multi_fitness_class(args):
    """Ensure the DEAP multi-objective fitness/individual classes exist."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    class_name = None
    if args.optimisation == 'multi':
        class_name = f"FitnessMulti{len(args.ga_objectives)}"
        if hasattr(creator, class_name):
            cls = getattr(creator, class_name)
        else:
            weights = tuple([1.0] * len(args.ga_objectives))
            creator.create(class_name, base.Fitness, weights=weights)
            cls = getattr(creator, class_name)
    else:
        cls = None
    if hasattr(creator, "Individual"):
        delattr(creator, "Individual")
    if args.optimisation == 'multi':
        creator.create("Individual", list, fitness=cls)
    else:
        creator.create("Individual", list, fitness=creator.FitnessMax)
    args.multi_fitness_class_name = class_name
    return cls


def _get_multi_fitness_class(args):
    """Return the DEAP multi-objective fitness class for current args."""
    name = getattr(args, "multi_fitness_class_name", None)
    if not name:
        return None
    return getattr(creator, name)


# --- Utility functions ---------------------------------------
_DATA = {}
def _init_worker(data_list, subject_id_list, args=None):
    _DATA.clear()
    _DATA["data_list"] = data_list
    _DATA["subject_id_list"] = subject_id_list
    if args is not None:
        _DATA["args"] = args

# Helper function for parallel clustering in bootstrap (now only takes the candidate)
def _cluster_candidate(cand, args=None):
    data_list = _DATA["data_list"]
    subject_id_list = _DATA["subject_id_list"]
    if args is None and "args" in _DATA:
        args = _DATA["args"]
    params = convert_to_parameters(len(data_list), cand)
    final_labels, individual_labels, view_scores_per_view, view_score, final_score = parea_2_mv(
        data_list,
        **params,
        subject_id_list=subject_id_list,
        inner_jobs=1,
        pre_inner_jobs=1,
        mincluster=args.mincluster,
        mincluster_n=args.mincluster_n
    )
    return final_labels, individual_labels, view_scores_per_view, view_score, final_score

def save_pickle(path, obj):
    """Save an object to a pickle file safely."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    """Load an object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def _resolve_path(base_dir, path):
    """Return an absolute path anchored at base_dir when a relative path is supplied."""
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))

def _ga_root(base_dir, fold_index):
    """Canonical GA root for a given fold under intermediates."""
    return os.path.join(base_dir, "intermediates", f"fold{fold_index}", "ga")

def preprocessing(df,
                  meta,
                  subject_id_column='src_subject_id',
                  col_threshold=0.5, row_threshold=0.5,
                  skew_threshold=0.75,
                  scaler_type='robust',
                  modalities=['Internalising', 'Functioning', 'Cognition', 'Detachment', 'Psychoticism'],
                  impute_parea=False):
   """
    Preprocess the data by removing high missing data, applying power transformation, dummy coding, scaling, and imputing.
    
    Parameters:
    - df: DataFrame containing the data.
    - meta: Metadata DataFrame.
    - subject_id_column: Column name for subject IDs.
    - col_threshold: Threshold for column missing data.
    - row_threshold: Threshold for row missing data.
    - skew_threshold: Threshold for skewness in data.
    - scaler_type: Type of scaler to use ('robust' or 'standard').
    NOTE: This function also exists in Utils. Duplicated here so if change, please change Utils too. 
    Returns:
    - dict_final: Dictionary containing processed modalities.
    """
   
   
   df_transformed = auto_power_transform(df, skew_threshold = skew_threshold) # Apply power transformation when data is skewed. 


   df_dummy = dummy_code(df_transformed)

   df_scaled = scale_diverse_data(df_dummy, subject_id_column=subject_id_column, scaler_type = scaler_type) # Scale data with robust scaler
   modal_dict = extract_modalities(meta, df_scaled) # Split data into modalities based on meta
   modal_dict_clean = {modality: modal_dict[modality] for modality in modalities if modality in modal_dict}

   # Reattach subject ID column to each modality so downstream steps include it
   for mod in modal_dict_clean:
       modal_dict_clean[mod][subject_id_column] = df_scaled[subject_id_column].loc[modal_dict_clean[mod].index]
   
   if impute_parea is False:
        # Identify any subjects who are entirely missing in any modality, then drop them from all modalities
        subjects_to_drop = set()
        for modality, df_mod in modal_dict_clean.items():
            # Exclude subject ID column when checking
            data_only = df_mod.drop(columns=[subject_id_column]) if subject_id_column in df_mod.columns else df_mod
            # Mask rows where all data columns are NaN
            missing_mask = data_only.isna().all(axis=1)
            if missing_mask.any():
                # Collect subject IDs to drop
                missing_ids = df_mod.loc[missing_mask, subject_id_column]
                subjects_to_drop.update(missing_ids.tolist())
        if subjects_to_drop:
            warnings.warn(
                f"Dropping {len(subjects_to_drop)} participants missing a full modality across all views: {subjects_to_drop}"
            )
            # Remove those subjects from every modality
            for modality in modal_dict_clean:
                df2 = modal_dict_clean[modality]
                modal_dict_clean[modality] = (
                    df2[~df2[subject_id_column].isin(subjects_to_drop)]
                    .reset_index(drop=True)
                )

   dict_imputed = impute_data(modal_dict_clean, subject_id_column=subject_id_column) # Impute missing data in each modality using KNN imputation
   dict_final = scale_data(dict_imputed, subject_id_column=subject_id_column, scaler_type=scaler_type) # Scale data in each modality with robust scaler
   
   # --- Canonical reindex across modalities ---
   id_col = subject_id_column
   mods = [m for m in modalities if m in dict_final]
   id_lists = {m: dict_final[m][id_col].tolist() for m in mods}
   # Sanity check: all views must have identical ID order after imputation and scaling
   # subjects present in all views
   shared = set.intersection(*(set(v) for v in id_lists.values()))
   # preserve order from the first modality in `modalities`
   canonical = [sid for sid in id_lists[mods[0]] if sid in shared]
   
   for m in mods:
       dfm = dict_final[m]
       dict_final[m] = (
           dfm[dfm[id_col].isin(shared)]
           .set_index(id_col)
           .loc[canonical]
           .reset_index()
       )

   # sanity: all views must now have identical ID order
   for m in mods[1:]:
    assert dict_final[m][id_col].tolist() == dict_final[mods[0]][id_col].tolist(), \
        f"Subject-ID order mismatch between {mods[0]} and {m}"

   
   # Build subject-ID list for each modality after imputation and scaling
   subject_id_list = []
   for mod in modalities:
       if mod in dict_final and subject_id_column in dict_final[mod]:
           subject_id_list.append(dict_final[mod][subject_id_column].tolist())
       else:
           subject_id_list.append([])
   ae_data = convert_data_for_vae(dict_final, subject_id_column=subject_id_column)

   return ae_data, subject_id_list, dict_final




# --- Helpers: collapse duplicate subject IDs within a bootstrap (majority vote) ---
def _collapse_duplicates(orig_ids, labels):
    """
    Given per-row orig_ids (possibly with duplicates) and corresponding labels,
    collapse to a single label per unique subject via majority vote.
    Ties are broken deterministically by choosing the smallest label value.

    Returns
    -------
    unique_ids : np.ndarray (ordered by first appearance)
    collapsed_labels : np.ndarray aligned to unique_ids
    """
    orig_ids = np.asarray(orig_ids)
    labels = np.asarray(labels)
    # Preserve first-seen order of subjects
    seen = {}
    order = []
    for sid in orig_ids:
        if sid not in seen:
            seen[sid] = True
            order.append(sid)
    unique_ids = np.array(order)
    # Majority vote per subject
    collapsed = []
    for sid in unique_ids:
        mask = (orig_ids == sid)
        vals, counts = np.unique(labels[mask], return_counts=True)
        # break ties by choosing the smallest label among those with maximal count
        maxc = counts.max()
        choice = vals[counts == maxc].min()
        collapsed.append(choice)
    return unique_ids, np.asarray(collapsed)

def _unique_ids_from_orig_ids(orig_ids):
    """Return unique subject IDs in first-seen order (matches _collapse_duplicates)."""
    orig_ids = np.asarray(orig_ids)
    seen = {}
    order = []
    for sid in orig_ids:
        if sid not in seen:
            seen[sid] = True
            order.append(sid)
    return np.array(order)

def precompute_consensus_cache(label_dicts):
    """
    Precompute union IDs and per-bootstrap union indices for consensus_pac_ccc.
    This avoids rebuilding ID maps and reduces per-call overhead.
    """
    unique_ids_list = []
    id_set = set()
    for d in label_dicts:
        uids = _unique_ids_from_orig_ids(d["orig_ids"])
        unique_ids_list.append(uids)
        id_set.update(uids.tolist())

    union_ids = np.array(sorted(id_set))
    index = {sid: i for i, sid in enumerate(union_ids)}
    idxs_list = [
        np.fromiter((index[sid] for sid in uids), dtype=int, count=len(uids))
        for uids in unique_ids_list
    ]
    return {"union_ids": union_ids, "idxs_list": idxs_list}

def precompute_bootstrap_pair_alignment(label_dicts):
    """
    Precompute bootstrap-pair alignment indices based on orig_ids only.
    This avoids repeating np.intersect1d for every individual.
    """
    unique_ids_list = []
    for d in label_dicts:
        unique_ids_list.append(_unique_ids_from_orig_ids(d["orig_ids"]))

    pair_indices = []
    n = len(unique_ids_list)
    for i in range(n):
        for j in range(i + 1, n):
            common, idx1, idx2 = np.intersect1d(
                unique_ids_list[i],
                unique_ids_list[j],
                return_indices=True
            )
            if len(common) > 1:
                pair_indices.append((i, j, idx1, idx2))

    return {
        "unique_ids_list": unique_ids_list,
        "pair_indices": pair_indices,
    }

# --- Co-association stability across bootstraps ---
def coassociation_stability(label_dicts, label_key):
    """
    Compute per-cluster co-association stability across bootstraps.

    Parameters
    ----------
    label_dicts : list of dict
        One dict per bootstrap, each with:
          - "orig_ids": 1D array-like of subject IDs in this bootstrap
          - label_key : 1D array-like of labels for these IDs
        Duplicates within a bootstrap are allowed and are collapsed by
        majority vote before use.

    label_key : str
        Key in each dict to use for labels (e.g. "labels").

    Returns
    -------
    cluster_stability: mean of individual cluster stabilities
        Stability score for each cluster in the *reference* partition,
        in ascending order of cluster label. For a given cluster k, the
        score is the mean consensus value M(i,j) over all pairs of
        subjects i,j that belong to cluster k in the reference
        clustering.
    •	✅ Report it as “fraction of times items in the cluster appear together”
	•	✅ High = stable, low = unstable

    Notes
    -----
    1. We first construct the N x N consensus matrix M where
         M(i,j) = (# times i and j are clustered together)
                  / (# times i and j co-occur in a bootstrap),
       using all bootstraps.

    2. We then take the clustering from the *first* bootstrap (after
       collapsing duplicates) as the reference partition and compute
       a per-cluster stability:
         m(k) = mean of M(i,j) over all pairs i,j in cluster k.

    3. This matches the "cluster consensus" idea from consensus
       clustering: values close to 1 mean that cluster k is very
       stable across resampling.
    """
    # --- Step 1: collapse duplicates and build union of subject IDs ---
    collapsed = []
    id_set = set()
    for d in label_dicts:
        uids, labs = _collapse_duplicates(d["orig_ids"], d[label_key])
        collapsed.append((np.asarray(uids), np.asarray(labs)))
        id_set.update(uids.tolist())

    union_ids = np.array(sorted(id_set))
    n = len(union_ids)
    if n <= 1:
        return -3, -3 # Not enough subjects for co-association stability, Error code.

    # Map subject id -> index in [0, n)
    index = {sid: i for i, sid in enumerate(union_ids)}

    # --- Step 2: accumulate co-presence and same-cluster counts ---
    same = np.zeros((n, n), dtype=np.uint32)
    co   = np.zeros((n, n), dtype=np.uint32)

    for uids, labs in collapsed:
        vec = np.full(n, -1, dtype=int)  # -1 means "absent in this bootstrap"
        idxs = np.fromiter((index[sid] for sid in uids), dtype=int, count=len(uids))
        vec[idxs] = labs

        present = (vec != -1)
        co_mask = np.outer(present, present)
        eq_mask = (vec[:, None] == vec[None, :]) & co_mask

        same += eq_mask
        co   += co_mask

    # If no pair ever co-occurs, nothing to compute
    if not np.any(co[np.triu_indices(n, k=1)] > 0):
        return -9, -9 # Not enough co-occurrences for co-association stability, Error code.

    # --- Step 3: build consensus matrix M(i,j) in [0,1] ---
    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.zeros((n, n), dtype=float)
        mask = co > 0
        consensus[mask] = same[mask].astype(float) / co[mask].astype(float)

    # --- Step 4: choose a reference clustering (first bootstrap) ---
    ref_uids, ref_labs = collapsed[0]
    ref_vec = np.full(n, -1, dtype=int)
    ref_idxs = np.fromiter((index[sid] for sid in ref_uids), dtype=int, count=len(ref_uids))
    ref_vec[ref_idxs] = ref_labs

    valid_mask = (ref_vec != -1)
    if not np.any(valid_mask):
        return -333, -333 # No subjects in reference clustering, Error code.

    cluster_labels = np.unique(ref_vec[valid_mask])

    # --- Step 5: compute per-cluster stability m(k) ---
    cluster_stabilities = []
    for k in cluster_labels:
        members = np.where(ref_vec == k)[0]
        if len(members) < 2:
            # Cluster of size 0 or 1: define stability as 0.0
            cluster_stabilities.append(0.0)
            continue

        # Consensus submatrix for this cluster
        subM = consensus[np.ix_(members, members)]
        iu_k = np.triu_indices(len(members), k=1)
        vals = subM[iu_k]

        if vals.size == 0:
            cluster_stabilities.append(0.0)
        else:
            cluster_stabilities.append(float(np.mean(vals)))

    # --- Step 6: compute overall CCC for consensus matrix ---
    # Distances from consensus
    D = 1.0 - consensus
    # Convert to condensed vector
    dvec = squareform(D, checks=False)

    # Hierarchical linkage on these distances
    Z = linkage(dvec, method="average")

    # cophenet returns (cophenetic_correlation, cophenetic_distances)
    ccc, _ = cophenet(Z, dvec)

    cluster_stability = np.mean(cluster_stabilities) if cluster_stabilities else 0

    return float(cluster_stability), float(ccc)


import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform

def consensus_pac_ccc(
    label_dicts,
    label_key,
    range_min=0.1,
    range_max=0.9,
    linkage_method="average",
    return_consensus=False,
    return_ecdf=False,
    precomputed_cache=None,
):
    """
    Compute MATLAB-style (Doms code) consensus diagnostics for a FIXED k:
      - consensus matrix (co-association)
      - ECDF of off-diagonal consensus values
      - PAC (Proportion of Ambiguous Clustering) over [range_min, range_max]
      - Cophenetic correlation coefficient (CCC) from hierarchical clustering on (1 - consensus)

    This matches the MATLAB pipeline conceptually:
      dd_calconsens  -> consensus matrix
      dd_ecdf        -> ECDF of consensus values
      dd_ecdfmin3    -> PAC over [range_min, range_max] (here: fraction in that interval)
      dd_cophenetic  -> CCC on the consensus-derived distances

    Parameters
    ----------
    label_dicts : list of dict
        One dict per bootstrap. Each dict must have:
          - "orig_ids": 1D array-like of subject IDs in this bootstrap
          - label_key : 1D array-like of labels for these IDs
        Duplicates within a bootstrap are allowed and collapsed by majority vote.

    label_key : str
        Key in each dict to use for labels (e.g. "labels").

    range_min, range_max : float
        Ambiguity interval for PAC, typically 0.1 and 0.9.

    linkage_method : str
        Linkage method for hierarchical clustering used in CCC (e.g. "average").

    precomputed_cache : dict or None
        Optional cache from precompute_consensus_cache(label_dicts) to reuse
        union ID mapping across calls.

    Returns
    -------
    out : dict
        {
          "consensus": (n, n) float array in [0,1],
          "union_ids": (n,) array of subject IDs (sorted),
          "ecdf_x": (m,) sorted values of off-diagonal consensus entries,
          "ecdf_f": (m,) ECDF values in [0,1],
          "PAC": float,
          "CCC": float,
          "meta": {...}
        }

    Notes
    -----
    - PAC here is computed as the fraction of *off-diagonal* consensus entries
      that fall strictly between [range_min, range_max], i.e.
        PAC = P(range_min < M(i,j) < range_max) over i<j with defined co-occurrence.
      Lower PAC => crisper / more stable clustering structure.
    """

    # --- Step 1: collapse duplicates and build union of subject IDs ---
    collapsed = []
    id_set = set() if precomputed_cache is None else None
    for d in label_dicts:
        if "orig_ids" not in d or label_key not in d:
            raise KeyError(f"Each dict must contain 'orig_ids' and '{label_key}'")
        uids, labs = _collapse_duplicates(d["orig_ids"], d[label_key])
        uids = np.asarray(uids)
        labs = np.asarray(labs)
        if uids.shape[0] != labs.shape[0]:
            raise ValueError("orig_ids and labels must have the same length after collapsing.")
        collapsed.append((uids, labs))
        if id_set is not None:
            id_set.update(uids.tolist())

    if precomputed_cache is None:
        union_ids = np.array(sorted(id_set))
        idxs_list = None
    else:
        union_ids = precomputed_cache.get("union_ids")
        idxs_list = precomputed_cache.get("idxs_list")

    n = len(union_ids)
    if n <= 1:
        return {
            "consensus": None,
            "union_ids": union_ids,
            "ecdf_x": np.array([]),
            "ecdf_f": np.array([]),
            "PAC": np.nan,
            "CCC": np.nan,
            "meta": {"error": "Not enough subjects to compute consensus."},
        }

    index = None
    if idxs_list is None or len(idxs_list) != len(collapsed):
        index = {sid: i for i, sid in enumerate(union_ids)}

    # --- Step 2: accumulate co-presence and same-cluster counts ---
    same = np.zeros((n, n), dtype=np.uint32)
    co = np.zeros((n, n), dtype=np.uint32)

    for b, (uids, labs) in enumerate(collapsed):
        if idxs_list is not None and b < len(idxs_list) and len(idxs_list[b]) == len(labs):
            idxs = idxs_list[b]
        else:
            if index is None:
                index = {sid: i for i, sid in enumerate(union_ids)}
            idxs = np.fromiter((index[sid] for sid in uids), dtype=int, count=len(uids))

        if len(idxs) == 0:
            continue

        eq_mask = (labs[:, None] == labs[None, :]).astype(np.uint32)
        same[np.ix_(idxs, idxs)] += eq_mask
        co[np.ix_(idxs, idxs)] += 1

    # Only consider pairs that ever co-occurred at least once
    iu = np.triu_indices(n, k=1)
    co_u = co[iu]
    if not np.any(co_u > 0):
        return {
            "consensus": None,
            "union_ids": union_ids,
            "ecdf_x": np.array([]),
            "ecdf_f": np.array([]),
            "PAC": np.nan,
            "CCC": np.nan,
            "meta": {"error": "No co-occurring pairs across bootstraps."},
        }

    # --- Step 3: build consensus matrix M(i,j) in [0,1] ---
    consensus = np.zeros((n, n), dtype=float)
    mask = co > 0
    consensus[mask] = same[mask].astype(float) / co[mask].astype(float)
    np.fill_diagonal(consensus, 1.0)  # conventional; doesn't affect off-diagonal PAC/CCC

    # --- Step 4: ECDF of off-diagonal consensus entries (defined pairs only) ---
    # Use only pairs with co-occurrence > 0
    vals = consensus[iu][co_u > 0]
    # ECDF: x sorted, f = (1..m)/m
    ecdf_x = np.sort(vals)
    m = ecdf_x.size
    ecdf_f = (np.arange(1, m + 1) / m) if m > 0 else np.array([])

    # Optionally drop ECDF arrays if not requested
    if not return_ecdf:
        ecdf_x = np.array([])
        ecdf_f = np.array([])

    # --- Step 5: PAC over [range_min, range_max] ---
    # MATLAB PAC is “mass in the ambiguous middle” between two thresholds.
    # Use strict interior by default; if you prefer inclusive, change comparisons.
    if m == 0:
        pac = np.nan
    else:
        pac = float(np.mean((np.sort(vals) > range_min) & (np.sort(vals) < range_max)))

    # --- Step 6: Cophenetic correlation coefficient (CCC) ---
    # Build distances from consensus: D = 1 - M
    D = 1.0 - consensus
    # condensed vector for linkage/cophenet
    dvec = squareform(D, checks=False)
    Z = linkage(dvec, method=linkage_method)
    ccc, _ = cophenet(Z, dvec)

    return {
        "consensus": consensus if return_consensus else None,
        "union_ids": union_ids,
        "ecdf_x": ecdf_x,
        "ecdf_f": ecdf_f,
        "PAC": pac,
        "CCC": float(ccc),
        "meta": {
            "n_subjects": int(n),
            "n_bootstraps": int(len(collapsed)),
            "range_min": float(range_min),
            "range_max": float(range_max),
            "linkage_method": linkage_method,
            "n_pairs_used": int(m),
        },
    }



# --- ARI and Jaccard-based stability across bootstraps ---
def ari_stability_common_subjects(label_dicts, label_key, precomputed_alignment=None):
    """
    Compute mean pairwise ARI for label arrays after collapsing duplicates
    within each bootstrap via majority vote, then aligning on common subjects.
    Each dict must have "orig_ids" and the labels under label_key.
    """
    scores = []
    if precomputed_alignment is None:
        for d1, d2 in combinations(label_dicts, 2):
            u1, l1 = _collapse_duplicates(d1["orig_ids"], d1[label_key])
            u2, l2 = _collapse_duplicates(d2["orig_ids"], d2[label_key])
            common, idx1, idx2 = np.intersect1d(u1, u2, return_indices=True)
            if len(common) > 1:
                scores.append(adjusted_rand_score(l1[idx1], l2[idx2]))
            else:
                warnings.warn("No common subjects between these two bootstraps; ARI stability contribution skipped.")
        return float(np.mean(scores)) if scores else 0.0

    labels_collapsed = []
    for d in label_dicts:
        _, labs = _collapse_duplicates(d["orig_ids"], d[label_key])
        labels_collapsed.append(np.asarray(labs))

    for b1, b2, idx1, idx2 in precomputed_alignment["pair_indices"]:
        l1 = labels_collapsed[b1][idx1]
        l2 = labels_collapsed[b2][idx2]
        scores.append(adjusted_rand_score(l1, l2))
    return float(np.mean(scores)) if scores else 0.0

def _partition_jaccard_from_labels(labels1, labels2):
    """
    Compute Jaccard index between two partitions defined on the same set of items.
    Partitions are given as integer label vectors of equal length.
    Jaccard is computed over the set of co-clustered pairs.
    """
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    n = len(labels1)
    if n < 2:
        return 0.0
    same1 = labels1[:, None] == labels1[None, :]
    same2 = labels2[:, None] == labels2[None, :]
    iu = np.triu_indices(n, k=1)
    a = same1[iu]
    b = same2[iu]
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def jaccard_stability_common_subjects(label_dicts, label_key, precomputed_alignment=None):
    """
    Compute mean pairwise Jaccard stability for label arrays after collapsing duplicates
    within each bootstrap via majority vote, then aligning on common subjects.
    The Jaccard index is computed over co-clustered pairs for each pair of bootstraps.
    """
    scores = []
    if precomputed_alignment is None:
        for d1, d2 in combinations(label_dicts, 2):
            u1, l1 = _collapse_duplicates(d1["orig_ids"], d1[label_key])
            u2, l2 = _collapse_duplicates(d2["orig_ids"], d2[label_key])
            common, idx1, idx2 = np.intersect1d(u1, u2, return_indices=True)
            if len(common) > 1:
                l1_aligned = l1[idx1]
                l2_aligned = l2[idx2]
                scores.append(_partition_jaccard_from_labels(l1_aligned, l2_aligned))
            else:
                warnings.warn("No common subjects between these two bootstraps; Jaccard stability contribution skipped.")
        return float(np.mean(scores)) if scores else 0.0

    labels_collapsed = []
    for d in label_dicts:
        _, labs = _collapse_duplicates(d["orig_ids"], d[label_key])
        labels_collapsed.append(np.asarray(labs))

    for b1, b2, idx1, idx2 in precomputed_alignment["pair_indices"]:
        l1_aligned = labels_collapsed[b1][idx1]
        l2_aligned = labels_collapsed[b2][idx2]
        scores.append(_partition_jaccard_from_labels(l1_aligned, l2_aligned))
    return float(np.mean(scores)) if scores else 0.0



# --- Fitness computation for gather ---
def _compute_fitness_for_ind(
    i,
    label_dicts,
    modalities,
    objectives,
    cache_dir=None,
    fold_index=None,
    bootstrap_index=None,
    precomputed_alignment=None,
    precomputed_consensus_cache=None
    ):
    """
    Compute per-view stability and quality, plus final stability and final quality,
    for individual i. Used in multi-objective GA optimisation.
    """

    n_views = len(modalities)

    # Decide which additional stability flavours to compute based on GA objectives
    objectives_set = set(objectives)
    need_coassoc = any("stability_coassoc" in obj for obj in objectives_set)
    need_ccc = any("stability_CCC" in obj or "stability_ccc" in obj for obj in objectives_set)
    need_jaccard = any("stability_jaccard" in obj for obj in objectives_set)

    # --- Helpers: detect degenerate (no-cluster) solutions ---
    def _n_unique_or_zero(labels):
        """Return number of unique labels; 0 if labels missing/empty."""
        try:
            a = np.asarray(labels)
        except Exception:
            return 0
        return int(len(np.unique(a))) if a.size > 0 else 0

    def _is_degenerate(labels):
        """Degenerate = no meaningful clustering (0 or 1 unique label)."""
        return _n_unique_or_zero(labels) <= 1

    # Degenerate flags across bootstraps for this individual
    final_degenerate = any(_is_degenerate(d.get("final_labels", [])[i]) for d in label_dicts)
    view_degenerate = []
    for v in range(n_views):
        deg_v = any(_is_degenerate(d.get("view_labels", [])[i][v]) for d in label_dicts)
        view_degenerate.append(deg_v)

    boot_dicts_final = [
        {"orig_ids": d["orig_ids"], "labels": d["final_labels"][i]}
        for d in label_dicts
    ]
    # Final-cluster stability
    # ARI is always computed as the primary stability metric
    final_stab_ari = ari_stability_common_subjects(
        boot_dicts_final,
        label_key="labels",
        precomputed_alignment=precomputed_alignment
    )

    # Additional stability metrics
    #final_stab_coassoc, final_stab_CCC = coassociation_stability(boot_dicts_final, label_key="labels")
    final_stab_jaccard = jaccard_stability_common_subjects(
        boot_dicts_final,
        label_key="labels",
        precomputed_alignment=precomputed_alignment
    )
    final_stab_SUM_MAT = consensus_pac_ccc(
        boot_dicts_final,
        label_key="labels",
        return_consensus=False,
        return_ecdf=False,
        precomputed_cache=precomputed_consensus_cache,
    )


    # If the final solution is degenerate (<= 1 unique cluster in any bootstrap),
    # force ALL final stability metrics to 0 so trivial one-cluster solutions
    # cannot look "perfectly stable" and dominate GA selection.
    if final_degenerate:
        final_stab_ari = 0.0
        #final_stab_coassoc = 0.0
        #final_stab_CCC = 0.0
        final_stab_jaccard = 0.0

    # Per-view stability
    view_stabs_ari = []
    #view_stabs_CCC = []
    #view_stabs_coassoc = []
    view_stabs_jaccard = []
    view_stabs_SUM_MAT = []

    for v in range(n_views):
        boot_dicts_view = [
            {"orig_ids": d["orig_ids"], "labels": d["view_labels"][i][v]}
            for d in label_dicts
        ]

        # ARI always computed per view (primary stability)
        stab_v_ari = ari_stability_common_subjects(
            boot_dicts_view,
            label_key="labels",
            precomputed_alignment=precomputed_alignment
        )

        #stab_v_coassoc, stab_v_CCC = coassociation_stability(boot_dicts_view, label_key="labels")
        stab_v_jaccard = jaccard_stability_common_subjects(
            boot_dicts_view,
            label_key="labels",
            precomputed_alignment=precomputed_alignment
        )
        stab_v_SUM_MAT = consensus_pac_ccc(
            boot_dicts_view,
            label_key="labels",
            return_consensus=False,
            return_ecdf=False,
            precomputed_cache=precomputed_consensus_cache,
        )

        # Force PER-VIEW stability metrics to 0 when this view is degenerate (<= 1 cluster)
        if view_degenerate[v]:
            stab_v_ari = 0.0
            #stab_v_coassoc = 0.0
            #stab_v_CCC = 0.0
            stab_v_jaccard = 0.0

        #view_stabs_coassoc.append(float(stab_v_coassoc))
        #view_stabs_CCC.append(float(stab_v_CCC))
        view_stabs_ari.append(float(stab_v_ari))
        view_stabs_jaccard.append(float(stab_v_jaccard))
        view_stabs_SUM_MAT.append({
            "PAC": stab_v_SUM_MAT.get("PAC", np.nan),
            "CCC": stab_v_SUM_MAT.get("CCC", np.nan),
            "meta": stab_v_SUM_MAT.get("meta", {}),
        })

    # --- Quality ---
    has_final_q = all("final_scores" in d for d in label_dicts)
    has_view_q = all("view_scores_per_view" in d for d in label_dicts)

    # Per-view quality: mean across bootstraps
    view_quals = []
    if has_view_q:
        for v in range(n_views):
            # Force PER-VIEW quality to 0 when this view is degenerate (<= 1 cluster)
            if view_degenerate[v]:
                view_quals.append(0.0)
            else:
                q_v = np.mean([float(d["view_scores_per_view"][i][v]) for d in label_dicts])
                view_quals.append(float(q_v))
    else:
        view_quals = [0.0] * n_views

    mean_final_q = float(np.mean([float(d["final_scores"][i]) for d in label_dicts])) if has_final_q else 0.0

    # Force FINAL quality to 0 when final solution is degenerate (<= 1 cluster)
    if final_degenerate:
        mean_final_q = 0.0

    mean_view_stab_ari = np.mean(view_stabs_ari) if view_stabs_ari else 0.0
    mean_view_qual = np.mean(view_quals) if view_quals else 0.0
    min_view_stab_ari = float(np.min(view_stabs_ari)) if view_stabs_ari else 0.0
    min_view_qual = float(np.min(view_quals)) if view_quals else 0.0

    # Also compute mean coassociation- and Jaccard-based stability for reporting
    #mean_view_stab_coassoc = np.mean(view_stabs_coassoc) if view_stabs_coassoc else 0.0
    #mean_view_stab_CCC = np.mean(view_stabs_CCC) if view_stabs_CCC else 0.0
    mean_view_stab_jaccard = np.mean(view_stabs_jaccard) if view_stabs_jaccard else 0.0
    if view_stabs_SUM_MAT:
        mean_view_stab_MAT_CCC = float(np.nanmean([d.get("CCC", np.nan) for d in view_stabs_SUM_MAT]))
        mean_view_stab_MAT_PAC = float(np.nanmean([d.get("PAC", np.nan) for d in view_stabs_SUM_MAT]))
    else:
        mean_view_stab_MAT_CCC = 0.0
        mean_view_stab_MAT_PAC = 0.0


    # Force FINAL stability metrics to 0 when final solution is degenerate (<= 1 cluster)
    if final_degenerate:
        final_stab_ari = 0.0
        #final_stab_coassoc = 0.0
        #final_stab_CCC = 0.0
        final_stab_jaccard = 0.0

    fitness_record = {
        "ind_id": i,
        "view_stabs_ari": view_stabs_ari,
        #"view_stabs_coassoc": view_stabs_coassoc,
        #"view_stabs_CCC": view_stabs_CCC,
        "view_stabs_jaccard": view_stabs_jaccard,
        "view_stabs_SUM_MAT": view_stabs_SUM_MAT,
        "view_quals": view_quals,
        "final_stab_ari": final_stab_ari,
        #"final_stab_coassoc": final_stab_coassoc,
        #"final_stab_CCC": final_stab_CCC,
        "final_stab_jaccard": final_stab_jaccard,
        "final_qual": mean_final_q,
    }
    if cache_dir:
        if fold_index is not None and bootstrap_index is not None:
            fname = f"fitness_{fold_index}_{bootstrap_index}_{i}.pkl"
        elif fold_index is not None:
            fname = f"fitness_{fold_index}_{i}.pkl"
        else:
            fname = f"fitness_{i}.pkl"
        save_pickle(os.path.join(cache_dir, fname), fitness_record)
    

    # Build summary dict and return requested objectives
    summary = {
        # ARI-based stability
        "mean_view_stability_ari": float(mean_view_stab_ari),
        "final_stability_ari": float(final_stab_ari),
        "min_view_stability_ari": float(min_view_stab_ari),
        "view_stabs_ari": tuple(view_stabs_ari),

        # Quality metrics
        "mean_view_quality": float(mean_view_qual),
        "view_quals": tuple(view_quals),
        "final_quality": float(mean_final_q),
        "min_view_quality": float(min_view_qual),

        # Additional stability flavours for reporting
        #"mean_view_stability_coassoc": float(mean_view_stab_coassoc),
        #"view_stabs_coassoc": tuple(view_stabs_coassoc),
        #"mean_view_stability_CCC": float(mean_view_stab_CCC),
        #"view_stabs_CCC": tuple(view_stabs_CCC),
        #"final_stability_coassoc": float(final_stab_coassoc),
        #"final_stability_CCC": float(final_stab_CCC),
        "mean_view_stability_jaccard": float(mean_view_stab_jaccard),
        "view_stabs_jaccard": tuple(view_stabs_jaccard),
        "final_stability_jaccard": float(final_stab_jaccard),
        "mean_view_stability_MAT_CCC": float(mean_view_stab_MAT_CCC),
        "mean_view_stability_MAT_PAC": float(mean_view_stab_MAT_PAC),
        "view_stabs_SUM_MAT": tuple(view_stabs_SUM_MAT),
        "final_stability_SUM_MAT": {
            "PAC": final_stab_SUM_MAT.get("PAC", np.nan),
            "CCC": final_stab_SUM_MAT.get("CCC", np.nan),
            "meta": final_stab_SUM_MAT.get("meta", {}),
        }
    }
    values = tuple(summary[obj] for obj in objectives)
    return values, tuple(view_stabs_ari), tuple(view_quals), summary



# Modes

def do_bootstrap(args):
    """
    Run one bootstrap iteration for GA population stability.
    """

    base_dir = os.path.abspath(getattr(args, "base_dir", "."))
    if args.fold_index is None:
        raise ValueError("For bootstrap mode, --fold_index must be specified")
    ga_root = _ga_root(base_dir, args.fold_index)
    population_file = _resolve_path(base_dir, args.population_file)
    population_initial_file = _resolve_path(base_dir, args.population_initial_file)
    output_labels_path = _resolve_path(base_dir, args.output_labels)
    if population_file is None and args.generation is not None:
        population_file = os.path.join(ga_root, f"population_fold{args.fold_index}_gen{args.generation}.pkl")
    if population_initial_file is None:
        population_initial_file = os.path.join(ga_root, f"population_init_fold{args.fold_index}.pkl")
    if output_labels_path is None:
        gen_dir = os.path.join(ga_root, f"gen{args.generation or 0}", f"bootstrap_{args.bootstrap_index or 0}")
        output_labels_path = os.path.join(gen_dir, f"labels_{args.bootstrap_index or 0}.pkl")

    # Deterministic unique seed per fold and bootstrap
    boot_index = getattr(args, "bootstrap_index", 0)
    if boot_index is None:
        boot_index = 0
    seed = 1000 * args.fold_index + boot_index + 17
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


    # Ensure DEAP’s creator classes exist before unpickling the population
    if args.optimisation == 'multi':
        _ensure_multi_fitness_class(args)
    else:
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # Load data and split according to outer CV
    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        # No CV split: use all rows for training to allow fast synthetic-data tests
        train_df = df.reset_index(drop=True)
    else:
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        train_idx, _ = list(kf.split(df))[args.fold_index]
        train_df = df.iloc[train_idx].reset_index(drop=True)



    # Load population
    if args.generation == 1:
        # Load initial population from args.population_initial_file
        if not population_initial_file:
            raise ValueError(" --population_initial_file must be specified")
        with open(population_initial_file, 'rb') as f:
            population = dill.load(f)
    else:
        print(f"Loading generation {args.generation} population from {population_file}")
        # Otherwise load the specified generation
        if args.generation is None:
            raise ValueError("For bootstrap mode, --generation must be specified")
        with open(population_file, 'rb') as f:
            population = dill.load(f)

    # Normalize fitness container type across all individuals to match optimisation mode
    if args.optimisation == 'multi':
        multi_cls = _get_multi_fitness_class(args)
        if multi_cls is None:
            multi_cls = _ensure_multi_fitness_class(args)
        for ind in population:
            ind.fitness = multi_cls()
    else:
        for ind in population:
            ind.fitness = creator.FitnessMax()

    # --- Bootstrap / Subsampling selection ---
    mode = getattr(args, "bootstrap_mode", "bootstrap").lower()
    if mode not in ["bootstrap", "subsample"]:
        raise ValueError(f"Invalid --bootstrap_mode '{mode}'. Must be 'bootstrap' or 'subsample'.")

    if mode == "bootstrap":
        print(f"[Fold {args.fold_index}] Running standard bootstrap (with replacement).")
        bdf = train_df.sample(n=len(train_df), replace=True, random_state=args.bootstrap_index).reset_index(drop=True)
        bdf = bdf.reset_index(drop=True)
    elif mode == "subsample":
        print(f"[Fold {args.fold_index}] Running subsampling mode (without replacement, 80% sample).")
        # Draw 80% of subjects without replacement, avoid degenerate samples
        frac = 0.8
        for attempt in range(100):
            bdf = train_df.sample(frac=frac, replace=False, random_state=args.bootstrap_index + attempt).reset_index(drop=True)
            bdf = bdf.reset_index(drop=True)
            if len(bdf.drop_duplicates(subset=args.subject_id_column)) >= 3:
                break
        else:
            raise RuntimeError(f"Failed to create a valid subsample after 100 attempts.")
    bdf["orig_subject_id"] = bdf[args.subject_id_column]
    # Assign a unique processed ID to each row for alignment
    bdf["proc_subject_id"] = np.arange(len(bdf))

    # Preprocess and train AE once on the bootstrap sample
    try:
        print("Start running Preprocessing")
        t_prep_start = time.time()
        ae_data, subject_id_list, dict_final = preprocessing(
            bdf, meta,
            subject_id_column='proc_subject_id',
            col_threshold=args.col_threshold,
            row_threshold=args.row_threshold,
            skew_threshold=args.skew_threshold,
            scaler_type=args.scaler_type,
            modalities=args.modalities
        )

        # Sanity check: all modalities must share identical subject order
        for lst in subject_id_list[1:]:
            assert lst == subject_id_list[0], "Subject-ID order mismatch across modalities"
        # Save subject IDs in the order or the processed data
        kept_proc_ids = subject_id_list[0]  # same order as final embeddings/labels
        # Map proc -> orig using bdf
        proc_to_orig = dict(zip(bdf["proc_subject_id"], bdf["orig_subject_id"]))
        kept_orig_ids = [proc_to_orig[p] for p in kept_proc_ids]


        t_prep_end = time.time()
        print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] Preprocessing took {t_prep_end - t_prep_start:.2f}s")

        if args.dim_reduction is None or args.dim_reduction.lower() == 'none':
            print("Skipping VAE and using preprocessed features as latent representations.")
            # Build VAE-like output from the fully preprocessed & scaled per-modality dataframes
            # Drop the subject ID column and convert to float32 arrays, preserving row order
            subj_col = 'proc_subject_id'
            ae_res = {}
            for mod in args.modalities:
                df_mod = dict_final[mod]
                X = df_mod.drop(columns=[subj_col]) if subj_col in df_mod.columns else df_mod
                X = X.to_numpy(dtype=np.float32, copy=True)
                ae_res[mod] = {"final_latent": X}
            data_list = [ae_res[mod]['final_latent'] for mod in args.modalities]
            # Free temporary container to keep parity with non-TEST branch
            del ae_res
            gc.collect()
        elif args.dim_reduction.lower() == 'vae':
            print("Start running VAE")
            # Seed RNGs for reproducibility
            np.random.seed(args.bootstrap_index)
            random.seed(args.bootstrap_index)
            torch.manual_seed(args.bootstrap_index)
            t_ae_start = time.time()
            ae_res = run_VAE_complete(
                ae_data,
                hidden_dims=args.hidden_dims,
                activation_functions=activation_functions,
                learning_rates=args.learning_rates,
                batch_sizes=args.batch_sizes,
                latent_dims=args.latent_dims
            )
            t_ae_end = time.time()
            print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] VAE nested CV took {t_ae_end - t_ae_start:.2f}s")
            # Extract embeddings once per modality (store as float32 to reduce memory)
            data_list = [np.asarray(ae_res[mod]['final_latent'], dtype=np.float32, copy=False)
                        for mod in args.modalities]
            # Free large autoencoder results to reduce peak memory before clustering
            del ae_res
            gc.collect()
        elif args.dim_reduction.lower() == "ae":
            print("Start running AE")
            # Seed RNGs for reproducibility
            np.random.seed(42)
            random.seed(42)
            torch.manual_seed(42)
            t_ae_start = time.time()
            ae_res = run_AE_complete(
                ae_data,
                hidden_dims=args.hidden_dims,
                activation_functions=activation_functions,
                learning_rates=args.learning_rates,
                batch_sizes=args.batch_sizes,
                latent_dims=args.latent_dims
            )
            t_ae_end = time.time()
            print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] AE nested CV took {t_ae_end - t_ae_start:.2f}s")
            # Extract embeddings once per modality (store as float32 to reduce memory)
            data_list = [np.asarray(ae_res[mod]['final_latent'], dtype=np.float32, copy=False)
                        for mod in args.modalities]
            # Free large autoencoder results to reduce peak memory before clustering
            del ae_res
            gc.collect()
        elif args.dim_reduction.lower() == "pca":
            print("Start running PCA")
            data_list = []
            for mod in args.modalities:
                df_mod = dict_final[mod]
                X = df_mod.drop(columns=['proc_subject_id']) if 'proc_subject_id' in df_mod.columns else df_mod
                pca = PCA(n_components=min(50, X.shape[1], X.shape[0]-1), random_state=args.bootstrap_index)
                X_pca = pca.fit_transform(X.to_numpy(dtype=np.float32, copy=True))
                data_list.append(X_pca)
            print("PCA dimensionality reduction completed.")
        else:
            raise ValueError(f"Unknown dim_reduction method: {args.dim_reduction}")

        print("Start running Parea on bootstrap sample...")
        # Seed RNGs before clustering
        np.random.seed(args.bootstrap_index)
        random.seed(args.bootstrap_index)
        torch.manual_seed(args.bootstrap_index)
        # Check GA individual gene names
        #print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] Sample gene_names from first 3 individuals: {[ind.gene_names for ind in population[:3]]}")
        t_clust_start = time.time()
        # Evaluate labels for each candidate in parallel using requested n_jobs (SLURM_CPUS_PER_TASK) if provided, otherwise all CPUs
        n_workers = args.n_jobs or (os.cpu_count() or 1)
        #print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] using {n_workers} workers for clustering")
        chunksize = max(1, len(population) // (n_workers * 4) if n_workers else len(population))
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(data_list, subject_id_list, args)
        ) as executor:
            # Map candidates; workers reuse shared read-only inputs
            all_results = list(executor.map(partial(_cluster_candidate, args=args), population, chunksize=chunksize))

        t_clust_end = time.time()
        print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] "
              f"Clustering {len(population)} candidates took {t_clust_end - t_clust_start:.2f}s")

        # Ensure lengths of IDs and labels align
        if len(all_results) > 0:
            assert len(kept_orig_ids) == len(all_results[0][0]), "orig_ids and final_labels length mismatch"
        # Prepare output dict
        #  - final_labels: list of arrays, one per candidate
        #  - view_labels:  list of lists-of-arrays, shape [n_candidates][n_views]
        final_labels = [res[0] for res in all_results]
        view_labels  = [res[1] for res in all_results]
        # Prepare output dict for quality scores (mean over views already in res[2])
        view_scores_per_view = [res[2] for res in all_results]
        view_scores_mean = [res[3] for res in all_results]
        final_scores = [res[4] for res in all_results]

        # NEW: compute per-candidate **min per-view composite quality** using returned per-view labels
        # Composite quality per view: average of three normalized indices in [0,1]:
        #  - Silhouette (normalized from [-1,1] to [0,1])
        #  - Calinski–Harabasz normalized via ch/(ch+1)
        #  - Davies–Bouldin transformed via 1/(1+db) (higher is better)
        def _per_view_composite_qualities(X_list, indiv_labs):
            def _k1_quality_features(X):
                # Try Dip on pairwise distance distribution; fall back to Hopkins
                try:
                    import diptest
                    from sklearn.metrics import pairwise_distances
                    D = pairwise_distances(np.asarray(X))
                    iu = np.triu_indices_from(D, k=1)
                    d = D[iu]
                    if d.size < 3:
                        return 1.0
                    _, p = diptest.diptest(np.asarray(d, dtype=float))
                    return float(max(0.0, min(1.0, p)))
                except Exception:
                    from sklearn.neighbors import NearestNeighbors
                    X = np.asarray(X, dtype=float)
                    n, d = X.shape
                    if n < 3:
                        return 1.0
                    m = min(50, n // 2)
                    rng = np.random.RandomState(42)
                    idx = rng.choice(n, size=m, replace=False)
                    X_m = X[idx]
                    mins, maxs = X.min(axis=0), X.max(axis=0)
                    U = rng.uniform(low=mins, high=maxs, size=(m, d))
                    nn = NearestNeighbors(n_neighbors=1).fit(X)
                    w, _ = nn.kneighbors(X_m, return_distance=True)
                    u, _ = nn.kneighbors(U,   return_distance=True)
                    W = np.power(w.ravel(), d).sum()
                    Uv = np.power(u.ravel(), d).sum()
                    H = Uv / (Uv + W + 1e-12)
                    return float(max(0.0, min(1.0, 1.0 - H)))

            vals = []
            for X, labs in zip(X_list, indiv_labs):
                labs = np.asarray(labs)
                if len(np.unique(labs)) <= 1:
                    vals.append(float(_k1_quality_features(X)))
                    continue
                try:
                    sil = silhouette_score(X, labs)           # [-1,1]
                    sil_n = (sil + 1.0) / 2.0                 # [0,1]
                except Exception:
                    sil_n = 0.0
                try:
                    ch = calinski_harabasz_score(X, labs)     # [0, +inf)
                    ch_n = ch / (ch + 1.0)                    # (0,1)
                except Exception:
                    ch_n = 0.0
                try:
                    db = davies_bouldin_score(X, labs)        # [0, +inf), lower is better
                    db_inv = 1.0 / (1.0 + db)                 # (0,1], higher is better
                except Exception:
                    db_inv = 0.0
                cq = (sil_n + ch_n + db_inv) / 3.0
                vals.append(float(cq))
            return vals

        view_scores_min = []
        for cand_indiv_labels in view_labels:
            quals = _per_view_composite_qualities(data_list, cand_indiv_labels)
            view_scores_min.append(float(np.min(quals)) if len(quals) else 0.0)
        #proc_ids_list = [res[2] for res in all_results]
        # Ensure all proc_id_lists are identical
        #if not all(proc_ids_list[0] == pid for pid in proc_ids_list):
        #    warnings.warn("Inconsistent subject-ID lists across cluster candidates")
        #proc_ids = proc_ids_list[0]

        to_dump = {
            "orig_ids":        kept_orig_ids,
            "final_labels":    final_labels,
            "view_labels":     view_labels,
            "view_scores_mean": view_scores_mean,
            "view_scores_per_view": view_scores_per_view,
            "view_scores_min":  view_scores_min,
            "final_scores":    final_scores,
        }

        # Save
        os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
        with open(output_labels_path, 'wb') as f:
            dill.dump(to_dump, f)
        print(f"Bootstrap labels {args.bootstrap_index} saved to {output_labels_path}")
        return
    except Exception as e:
        # Sentinel on bootstrap failure: write an empty labels dict and exit 0 so the batch can proceed
        try:
            os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
            sentinel = {
                "orig_ids": [],
                "final_labels": [],
                "view_labels": [],
                "view_scores_mean": [],
                "view_scores_per_view": [],
                "view_scores_min": [],
                "final_scores": []
            }
            with open(output_labels_path, 'wb') as f:
                dill.dump(sentinel, f)
            print(f"[Fold {args.fold_index}] Bootstrap {args.bootstrap_index} marked as SKIPPED due to error: {e}")
        except Exception as ee:
            print(f"[Fold {args.fold_index}] Failed to write sentinel for bootstrap {args.bootstrap_index}: {ee}")
        return


def do_gather(args):
    """
    Gather stabilities and evolve GA population one generation.
    """
    base_dir = os.path.abspath(getattr(args, "base_dir", "."))
    if args.fold_index is None:
        raise ValueError("For gather mode, --fold_index must be specified")
    ga_root = _ga_root(base_dir, args.fold_index)
    bootstrap_dir = _resolve_path(base_dir, args.bootstrap_dir)
    population_dir = _resolve_path(base_dir, args.population_dir) if args.population_dir else None
    population_file = _resolve_path(base_dir, args.population_file) if args.population_file else None
    population_initial_file = _resolve_path(base_dir, args.population_initial_file) if args.population_initial_file else None
    output_population = _resolve_path(base_dir, args.output_population) if args.output_population else None
    if population_dir is None and args.generation is not None:
        population_dir = os.path.join(ga_root, f"gen{args.generation}")
    if population_file is None and args.generation is not None:
        population_file = os.path.join(ga_root, f"population_fold{args.fold_index}_gen{args.generation}.pkl")
    if population_initial_file is None:
        population_initial_file = os.path.join(ga_root, f"population_init_fold{args.fold_index}.pkl")
    if output_population is None and args.generation is not None:
        output_population = os.path.join(ga_root, f"population_fold{args.fold_index}_gen{args.generation + 1}.pkl")

    if args.fold_index is None:
        raise ValueError("For gather mode, --fold_index must be specified")
    if args.generation is None:
        raise ValueError("For gather mode, --generation must be specified")
    if not bootstrap_dir:
        raise ValueError("For gather mode, --bootstrap_dir must be specified")
    if not population_dir:
        raise ValueError("For gather mode, --population_dir must be specified")

    # Deterministic unique seed per fold and bootstrap
    boot_index = getattr(args, "bootstrap_index", 0)
    if boot_index is None:
        boot_index = 0
    seed = 1000 * args.fold_index + boot_index + 17
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    hof_dir = ga_root
    fitness_cache_dir = None
    if args.generation is not None:
        fitness_cache_dir = os.path.join(
            hof_dir,
            "fitness_cache",
            f"gen{args.generation:03d}"
        )

    # Ensure fitness class exists BEFORE unpickling populations/HOF created in prior runs
    if args.optimisation == 'multi':
        _ensure_multi_fitness_class(args)
    if args.optimisation == 'single' and not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Find and load all bootstrap label dicts under the specified directory
    pattern = os.path.join(bootstrap_dir, 'bootstrap_*', 'labels_*.pkl')

    import re

    def numeric_boot_dirs(path):
        m = re.search(r'bootstrap_(\d+)', path)
        return int(m.group(1)) if m else -1

    files = sorted(glob.glob(pattern), key=numeric_boot_dirs)

    if not files:
        raise FileNotFoundError(f"No label files found in {bootstrap_dir}; expected bootstrap_*/labels_*.pkl")

    # Each file contains a dict with keys "final_labels", "view_labels", "view_scores" and "final_scores". The scores are the qualities.
    label_dicts_all = [dill.load(open(fn, 'rb')) for fn in files]
    def _usable(d):
        return isinstance(d, dict) and len(d.get("final_labels", [])) > 0 and len(d.get("orig_ids", [])) > 0
    label_dicts = [d for d in label_dicts_all if _usable(d)]

    min_needed = max(1, args.n_bootstrap - 5)
    if len(label_dicts) < min_needed:
        raise RuntimeError(f"Only {len(label_dicts)} usable bootstraps (min required {min_needed}). Check for sentinel/failed runs in {bootstrap_dir}.")

    # Precompute caches to avoid repeated intersect1d and consensus ID mapping
    precomputed_alignment = precompute_bootstrap_pair_alignment(label_dicts)
    precomputed_consensus_cache = precompute_consensus_cache(label_dicts)

    # Unpack lists across bootstraps
    # final_label_sets: list of lists, one per bootstrap, each list of length pop_size
    final_label_sets = [d["final_labels"] for d in label_dicts]
    # Evaluate GA objectives for each individual using ProcessPoolExecutor
    pop_size = len(final_label_sets[0])
    n_workers = args.n_jobs or (os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(
            _compute_fitness_for_ind,
            range(pop_size),
            repeat(label_dicts),
            repeat(args.modalities),
            repeat(tuple(args.ga_objectives)),
            repeat(fitness_cache_dir),
            repeat(args.fold_index),
            repeat(boot_index),
            repeat(precomputed_alignment),
            repeat(precomputed_consensus_cache)
        ))
        fitness = [tuple(map(float, res[0])) for res in results]
        per_view_stabs = [tuple(map(float, res[1])) for res in results]
        per_view_quals = [tuple(map(float, res[2])) for res in results]
        summary_metrics = [res[3] for res in results]
    # Save fitness history for this fold
    os.makedirs(hof_dir, exist_ok=True)
    history_file = os.path.join(hof_dir, "fitness_history.pkl")
    # Ensure the directory exists before writing the history file
    os.makedirs(hof_dir, exist_ok=True)
    if os.path.exists(history_file):
        with open(history_file, "rb") as hf:
            fitness_history = pickle.load(hf)
    else:
        fitness_history = {}
    # Record this generation's fitness tuples
    fitness_history[args.generation] = fitness
    with open(history_file, "wb") as hf:
        pickle.dump(fitness_history, hf)

    if args.generation == 1:
        # Load initial population from args.population_initial_file
        if not args.population_initial_file:
            raise ValueError("For gather mode, --population_initial_file must be specified")
        with open(population_initial_file, 'rb') as f:
            population = dill.load(f)
    elif args.generation > 1:
        # Otherwise load the specified generation
        if args.generation is None:
            raise ValueError("For gather mode, --generation must be specified")
        with open(population_file, 'rb') as f:
            population = dill.load(f)

    # Build gene_names in the same per-view order used at init: for each view -> k, method, then global pre_method
    n_views = len(args.modalities)
    gene_names = []
    for i in range(1, n_views + 1):
        gene_names.append(f"c_{i}_k")
        gene_names.append(f"c_{i}_method")
    gene_names.append("pre_method")
    gene_names.append("k_final")
    gene_names.append("fusion_method")
    for ind in population:
        ind.gene_names = gene_names

    # Ensure DEAP’s creator classes exist before unpickling the population
    if args.optimisation == 'multi':
        multi_cls = _get_multi_fitness_class(args)
        if multi_cls is None:
            multi_cls = _ensure_multi_fitness_class(args)
    else:
        multi_cls = None
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    # Reset fitness container type across all individuals to match optimisation mode
    if args.optimisation == 'multi':
        for ind in population:
            if not isinstance(ind.fitness, multi_cls):
                ind.fitness = multi_cls()
    else:
        for ind in population:
            ind.fitness = creator.FitnessMax()

    # Clear any stale fitness values to avoid mismatches
    for ind in population:
        try:
            del ind.fitness.values
        except AttributeError:
            pass


    # Assign fitness to each individual based on optimisation mode
    if args.optimisation == 'single':
        # Single-objective: fitness tuples contain only final stability
        stab_view_key, stab_final_key, qual_view_key, qual_final_key = _primary_metric_keys(args)
        for idx, (final_stab,) in enumerate(fitness):
            ind = population[idx]
            ind.fitness.values = (final_stab,)
            ind.view_stabs_per_view = per_view_stabs[idx]
            ind.view_quals_per_view = per_view_quals[idx]
            summary = summary_metrics[idx]
            ind.metrics_summary = summary
            ind.mean_view_stab = summary.get(stab_view_key)
            ind.mean_view_qual = summary.get(qual_view_key)
            ind.final_stab = summary.get(stab_final_key)
            ind.final_qual = summary.get(qual_final_key)
    elif args.optimisation == 'multi':
        stab_view_key, stab_final_key, qual_view_key, qual_final_key = _primary_metric_keys(args)
        for idx, ind in enumerate(population):
            fitvals = tuple(map(float, fitness[idx]))
            assert len(fitvals) == len(ind.fitness.weights), (
                f"Mismatch: {len(fitvals)} values vs {len(ind.fitness.weights)} weights"
            )
            ind.fitness.values = fitvals
            ind.view_stabs_per_view = per_view_stabs[idx]
            ind.view_quals_per_view = per_view_quals[idx]
            summary = summary_metrics[idx]
            ind.metrics_summary = summary
            ind.mean_view_stab = summary.get(stab_view_key)
            ind.mean_view_qual = summary.get(qual_view_key)
            ind.final_stab = summary.get(stab_final_key)
            ind.final_qual = summary.get(qual_final_key)
    else:
        raise ValueError(f"Unknown optimisation mode: {args.optimisation}")


    # ---------------- Generate-only GA step (ask/tell style) ----------------
    # Record statistics over the evaluated current population
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    if args.optimisation == 'multi':
        for idx, name in enumerate(args.ga_objectives):
            stats.register(
                f"avg_{name}",
                lambda vals, idx=idx: float(np.mean([v[idx] for v in vals]))
            )
            stats.register(
                f"max_{name}",
                lambda vals, idx=idx: float(np.max([v[idx] for v in vals]))
            )
    else:
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

    cur_stats = stats.compile(population)
    print("[Gather] Current-pop stats:", cur_stats)
    if args.optimisation == 'multi' and per_view_stabs and len(per_view_stabs[0]) > 0:
        stab_matrix = np.array(per_view_stabs, dtype=float)
        qual_matrix = np.array(per_view_quals, dtype=float)
        summary_rows = []
        avg_stabs_per_mod = []
        avg_quals_per_mod = []

        # Also build matrices for coassociation-, CCC-, and Jaccard-based stability from the summaries
        coassoc_matrix = np.array(
            [m.get("view_stabs_coassoc", [np.nan] * len(args.modalities)) for m in summary_metrics],
            dtype=float
        )
        ccc_matrix = np.array(
            [m.get("view_stabs_CCC", [np.nan] * len(args.modalities)) for m in summary_metrics],
            dtype=float
        )
        jaccard_matrix = np.array(
            [m.get("view_stabs_jaccard", [np.nan] * len(args.modalities)) for m in summary_metrics],
            dtype=float
        )

        for idx, mod in enumerate(args.modalities):
            avg_stab = float(np.mean(stab_matrix[:, idx]))
            avg_qual = float(np.mean(qual_matrix[:, idx]))
            avg_stabs_per_mod.append((mod, avg_stab))
            avg_quals_per_mod.append((mod, avg_qual))
            summary_rows.append(f"{mod}: mean_stab={avg_stab:.3f}, mean_qual={avg_qual:.3f}")
        min_stab_mod, min_stab_val = min(avg_stabs_per_mod, key=lambda x: x[1])
        min_qual_mod, min_qual_val = min(avg_quals_per_mod, key=lambda x: x[1])
        print(
            "[Gather] Per-view means -> "
            + " | ".join(summary_rows)
            + f" || min_mean_stab={min_stab_val:.3f} ({min_stab_mod}), "
              f"min_mean_qual={min_qual_val:.3f} ({min_qual_mod})"
        )
        # Persist per-modality best stability/quality for this generation
        best_stab_per_mod_ari = np.max(stab_matrix, axis=0)
        best_stab_per_mod_coassoc = np.nanmax(coassoc_matrix, axis=0)
        best_stab_per_mod_ccc = np.nanmax(ccc_matrix, axis=0)
        best_stab_per_mod_jaccard = np.nanmax(jaccard_matrix, axis=0)

        best_qual_per_mod = np.max(qual_matrix, axis=0)

        view_hist_path = os.path.join(hof_dir, "view_history.pkl")
        if os.path.exists(view_hist_path):
            with open(view_hist_path, "rb") as vh:
                view_history = pickle.load(vh)
        else:
            view_history = {"modalities": args.modalities, "generations": {}}
        view_history["generations"][args.generation] = {
            # ARI-based best per modality (primary stability)
            "best_stab": best_stab_per_mod_ari.tolist(),
            # Additional metrics for diagnostics
            "best_stab_coassoc": best_stab_per_mod_coassoc.tolist(),
            "best_stab_ccc": best_stab_per_mod_ccc.tolist(),
            "best_stab_jaccard": best_stab_per_mod_jaccard.tolist(),
            "best_qual": best_qual_per_mod.tolist()
        }
        with open(view_hist_path, "wb") as vh:
            pickle.dump(view_history, vh)

    # ——— Persistent Hall-of-Fame across ALL generations in this fold ———
    hof_path = os.path.join(hof_dir, "halloffame.pkl")
    if os.path.exists(hof_path):
        with open(hof_path, 'rb') as f:
            hall_of_fame = dill.load(f)
    else:
        hall_of_fame = tools.ParetoFront() if args.optimisation == 'multi' else tools.HallOfFame(maxsize=1)

    # Update and persist Hall of Fame / Pareto front with the evaluated population
    hall_of_fame.update(population)
    with open(hof_path, 'wb') as f:
        dill.dump(hall_of_fame, f)

    # Breeding operators
    toolbox = base.Toolbox()
    # GA mutation indices for layout: [c_1_k, c_1_method, ..., c_V_k, c_V_method, pre_method, k_final, fusion_method]
    n_views = len(args.modalities)
    k_positions = [2*i for i in range(n_views)]
    linkage_positions = [2*i + 1 for i in range(n_views)]
    pre_linkage_index = 2 * n_views
    k_final_index = 2 * n_views + 1
    f_index = 2 * n_views + 2

    fusion_methods = list(args.fusion_methods)
    linkages = list(args.linkages)
    if not fusion_methods:
        raise ValueError("At least one fusion method must be supplied via --fusion_methods.")
    #linkages = ['complete','average','weighted']

    def mutate(individual):
        idx = random.randint(0, len(individual) - 1)
        if idx in k_positions:
            individual[idx] = random.randint(args.k_min, args.k_max)
        elif idx in linkage_positions:
            individual[idx] = random.choice(linkages)
        elif idx == pre_linkage_index:
            individual[idx] = random.choice(linkages)
        elif idx == k_final_index:
            individual[idx] = random.randint(args.k_min, args.k_max)
        elif idx == f_index:
            individual[idx] = random.choice(fusion_methods)
        return (individual,)

    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate)
    if args.optimisation == 'single':
        toolbox.register("select_parents", tools.selTournament, tournsize=3)
    else:
        toolbox.register("select_parents", tools.selNSGA2)

    # Select parents and create children for the *next* generation
    parents = toolbox.select_parents(population, k=len(population))
    offspring = algorithms.varOr(
        parents,
        toolbox,
        lambda_=len(population),
        cxpb=args.ga_cxpb,
        mutpb=args.ga_mutpb
    )

    # Preserve gene names for all offspring
    for ind in offspring:
        ind.gene_names = population[0].gene_names

    # Ensure children have empty fitness so the bootstrap stage will evaluate them
    for ind in offspring:
        try:
            del ind.fitness.values
        except AttributeError:
            pass
        if hasattr(population[0], 'gene_names') and not hasattr(ind, 'gene_names'):
            ind.gene_names = population[0].gene_names

    # --- Validation: enforce consistent fitness tuple lengths before elitism ---
    if args.optimisation == 'multi':
        expected_len = len(args.ga_objectives)
        if len(population) != len(fitness):
            raise RuntimeError(f"Population size ({len(population)}) != fitness list size ({len(fitness)}).")
        for idx, ind in enumerate(population):
            vals = getattr(ind.fitness, 'values', ())
            if len(vals) != expected_len:
                try:
                    fitvals = fitness[idx]
                except Exception:
                    raise RuntimeError(f"Missing or invalid fitness for individual {idx}: {fitness[idx] if idx < len(fitness) else 'N/A'}")
                ind.fitness.values = tuple(map(float, fitvals))
    else:
        # Single-objective must have length-1 tuples
        if len(population) != len(fitness):
            raise RuntimeError(f"Population size ({len(population)}) != fitness list size ({len(fitness)}).")
        for idx, ind in enumerate(population):
            vals = getattr(ind.fitness, 'values', ())
            if len(vals) != 1:
                try:
                    (fs,) = fitness[idx]
                except Exception:
                    raise RuntimeError(f"Missing or invalid single fitness for individual {idx}: {fitness[idx] if idx < len(fitness) else 'N/A'}")
                ind.fitness.values = (float(fs),)

    # Elitism: carry the best 2 individuals from the evaluated population
    elite_count = max(1, min(args.ga_elitism, len(population)))
    if args.optimisation == 'single':
        elites = tools.selBest(population, k=elite_count)
    else:
        elites = tools.selNSGA2(population, k=elite_count)

    # Build next generation: elites + (offspring trimmed to fill the rest)
    slots = max(0, len(population) - len(elites))
    next_population = list(elites) + list(offspring[:slots])

    # Also preserve gene_names on HOF individuals
    try:
        for ind in hall_of_fame:
            ind.gene_names = gene_names
    except Exception:
        pass

    # Save the newly generated population for the next generation of bootstraps
    out_path = output_population
    if not population_dir:
        raise ValueError("For gather mode, --population_dir must be specified")
    if not out_path:
        raise ValueError("For gather mode, --output_population must be specified")
    os.makedirs(population_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, 'wb') as f:
        dill.dump(next_population, f)
    print(f"Generated next population (elitism={elite_count}) saved to {out_path}")
    return


def do_outer(args):
    """
    Final retrain AE & Parea on outer fold; SVM commented.
    """
    base_dir = os.path.abspath(getattr(args, "base_dir", "."))
    if args.fold_index is None:
        raise ValueError("For outer mode, --fold_index must be specified")
    ga_root = _ga_root(base_dir, args.fold_index)
    population_file = _resolve_path(base_dir, args.population_file) if args.population_file else None
    if population_file is None:
        population_file = os.path.join(ga_root, f"population_fold{args.fold_index}_gen{args.generation or 0}.pkl")
    output_metrics_path = _resolve_path(base_dir, args.output_metrics) if args.output_metrics else None
    output_metrics_merged_path = output_metrics_path

    if not population_file:
        raise ValueError("For outer mode, --population_file must be specified")
    if not output_metrics_path:
        raise ValueError("For outer mode, --output_metrics must be specified")

    # Deterministic unique seed per fold and bootstrap
    boot_index = getattr(args, "bootstrap_index", 0)
    if boot_index is None:
        boot_index = 0
    seed = 1000 * args.fold_index + boot_index + 17
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        # Train on the full dataset; leave test empty for fast single-fold runs
        train_df = df.reset_index(drop=True)
        test_df  = df.iloc[0:0].copy()
        # Create dummy indices for downstream ID capture
        train_idx = train_df.index.tolist()
        test_idx  = []
    else:
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        # Extract train and test indices for this outer fold
        train_idx, test_idx = list(kf.split(df))[args.fold_index]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

    # Retrain AE on the train split only:
    ae_data, subject_id_list, dict_final = preprocessing(
        train_df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities
    )

    # Assert identical subject order across modalities after preprocessing
    base_ids = dict_final[args.modalities[0]][args.subject_id_column].tolist()
    for m in args.modalities[1:]:
        assert dict_final[m][args.subject_id_column].tolist() == base_ids, \
            f"Subject-ID order mismatch between {args.modalities[0]} and {m} after preprocessing"

    if args.dim_reduction is None or args.dim_reduction.lower() == 'none':
        print("Skipping VAE and using preprocessed features as latent representations.")
        # Build VAE-like output from the fully preprocessed & scaled per-modality dataframes
        # Drop the subject ID column and convert to float32 arrays, preserving row order
        subj_col = args.subject_id_column
        ae_res = {}
        for mod in args.modalities:
            df_mod = dict_final[mod]
            X = df_mod.drop(columns=[subj_col]) if subj_col in df_mod.columns else df_mod
            X = X.to_numpy(dtype=np.float32, copy=True)
            ae_res[mod] = {"final_latent": X}
    elif args.dim_reduction.lower() == 'vae':
        print("Start running VAE")
        # Seed RNGs for reproducibility
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)
        ae_res = run_VAE_complete(
            ae_data,
            hidden_dims=args.hidden_dims,
            activation_functions=activation_functions,
            learning_rates=args.learning_rates,
            batch_sizes=args.batch_sizes,
            latent_dims=args.latent_dims
        )
    elif args.dim_reduction.lower() == "ae":
            print("Start running AE")
            # Seed RNGs for reproducibility
            np.random.seed(42)
            random.seed(42)
            torch.manual_seed(42)
            t_ae_start = time.time()
            ae_res = run_AE_complete(
                ae_data,
                hidden_dims=args.hidden_dims,
                activation_functions=activation_functions,
                learning_rates=args.learning_rates,
                batch_sizes=args.batch_sizes,
                latent_dims=args.latent_dims
            )
            t_ae_end = time.time()
            print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] AE nested CV took {t_ae_end - t_ae_start:.2f}s")
            # Extract embeddings once per modality (store as float32 to reduce memory)
            data_list = [np.asarray(ae_res[mod]['final_latent'], dtype=np.float32, copy=False)
                        for mod in args.modalities]
            # Free large autoencoder results to reduce peak memory before clustering
            del ae_res
            gc.collect()
    elif args.dim_reduction.lower() == "pca":
        print("Start running PCA")
        ae_res = {}
        for mod in args.modalities:
            df_mod = dict_final[mod]
            X = df_mod.drop(columns=['proc_subject_id']) if 'proc_subject_id' in df_mod.columns else df_mod
            pca = PCA(n_components=min(50, X.shape[1], X.shape[0]-1), random_state=42)
            X_pca = pca.fit_transform(X.to_numpy(dtype=np.float32, copy=True))
            ae_res[mod] = {"final_latent": X_pca}
        print("PCA dimensionality reduction completed.")
    else:
        raise ValueError(f"Unknown dim_reduction method: {args.dim_reduction}")

    # Ensure multi-objective fitness class is defined for Pareto comparisons and unpickling HOF
    if args.optimisation == 'multi':
        _ensure_multi_fitness_class(args)
    if args.optimisation == 'single' and not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # Load the persistent Hall-of-Fame for this fold and take its champion
    ga_root = os.path.dirname(population_file)
    hof_path = os.path.join(ga_root, "halloffame.pkl")
    with open(hof_path, 'rb') as f:
        hall_of_fame = dill.load(f)

    # Select best individual that yields at least 2 clusters on the training fold
    # Prepare latent representations per modality
    ae_cluster = {mod: ae_res[mod]['final_latent'] for mod in args.modalities}
    data_list = [ae_cluster[mod] for mod in args.modalities]
    # Build candidate list for champion selection
    if args.optimisation == 'single':
        candidates = list(hall_of_fame)
    else:
        candidates = list(hall_of_fame)
    # For multi-objective: sort candidates by distance to ideal point
    if args.optimisation == 'multi':
        vals = np.array([ind.fitness.values for ind in candidates], dtype=float)
        # Normalize objectives to [0,1] across candidates (avoid div-by-zero)
        mins, maxs = vals.min(axis=0), vals.max(axis=0)
        rng = np.where(maxs > mins, maxs - mins, 1.0)
        norm = (vals - mins) / rng
        d = np.linalg.norm(1.0 - norm, axis=1)
        order = np.argsort(d)  # ascending distance to ideal
        candidates = [candidates[i] for i in order]

    # --- DIAGNOSTICS (TEST mode): evaluate top-K Pareto candidates on ARI and save CSV ---
    if getattr(args, 'TEST', 'FALSE').upper() == "TRUE":
        try:
            K = min(20, len(candidates))
            eval_rows = []
            df_truth = pd.read_csv("/data/gpfs/projects/punim1993/students/Jente/multiclust/synthetic_multimodal_spartan.csv")
            truth_map = df_truth.set_index(args.subject_id_column)
            ref_ids = dict_final[args.modalities[0]][args.subject_id_column].to_numpy()
            true_cols = [
                truth_map.loc[ref_ids, f"subgroup_m{i+1}"].to_numpy() for i in range(len(args.modalities))
            ]
            true_ints_list = []
            for arr in true_cols:
                uniq = np.unique(arr)
                l2i = {name: idx for idx, name in enumerate(uniq)}
                true_ints_list.append(np.array([l2i[v] for v in arr], dtype=int))

            diag_dir = ga_root
            os.makedirs(diag_dir, exist_ok=True)
            for rank, ind in enumerate(candidates[:K]):
                params = convert_to_parameters(len(args.modalities), ind)
                labels, indiv_labels, view_scores_per_view, view_q, final_q = parea_2_mv(
                    data_list,
                    **params,
                    subject_id_list=subject_id_list,
                    inner_jobs=args.n_jobs,
                    pre_inner_jobs=args.n_jobs,
                    mincluster=args.mincluster,
                    mincluster_n=args.mincluster_n
                )
                aris = []
                for i, pred in enumerate(indiv_labels):
                    pred = np.asarray(pred, dtype=int)
                    ari = adjusted_rand_score(true_ints_list[i], pred)
                    aris.append(float(ari))
                row = {"rank": rank, "fitness": ind.fitness.values}
                if args.optimisation == 'multi':
                    for name, val in zip(args.ga_objectives, ind.fitness.values):
                        row[name] = float(val)
                else:
                    row["final_stability"] = float(ind.fitness.values[0])
                # attach ARIs per modality
                for i, mod in enumerate(args.modalities):
                    row[f"ARI_{mod}"] = aris[i]
                # attach gene params
                if hasattr(ind, 'gene_names'):
                    for gname, gval in zip(ind.gene_names, list(ind)):
                        row[gname] = gval
                eval_rows.append(row)
            eval_df = pd.DataFrame(eval_rows)
            csv_path = os.path.join(diag_dir, f"pareto_eval_top{K}.csv")
            eval_df.to_csv(csv_path, index=False)
            print(f"[Fold {args.fold_index}] Pareto diagnostics written to {csv_path}")
            # Quick console summary of best ARI per view
            for i, mod in enumerate(args.modalities):
                best_idx = int(np.nanargmax(eval_df[f"ARI_{mod}"].to_numpy()))
                best_row = eval_df.iloc[best_idx]
                print(f"[Diag] Best ARI_{mod}={best_row[f'ARI_{mod}']:.3f} at rank {best_row['rank']}, params subset: k_s={[best_row.get(f'c_{j+1}_k') for j in range(len(args.modalities))]}, k_final={best_row.get('k_final')}, fusion={best_row.get('fusion_method')}")
        except Exception as e:
            print(f"[Fold {args.fold_index}] WARNING: Pareto diagnostics failed: {e}")

    # Select the best candidate that yields at least 2 clusters on training data
    ind = candidates[0]
    params = convert_to_parameters(len(args.modalities), ind)

    labels, indiv_labels, view_scores_per_view, view_score_mean, final_score = parea_2_mv(
        data_list,
        subject_id_list=subject_id_list,
        inner_jobs=args.n_jobs,
        pre_inner_jobs=args.n_jobs,
        mincluster=args.mincluster,
        mincluster_n=args.mincluster_n,
        **params
    )

    best = ind
    best_params = params
    train_final_labels = labels
    train_individual_labels = indiv_labels
    train_view_scores_per_view = view_scores_per_view
    train_view_quality_mean = view_score_mean
    train_final_quality = final_score

    summary = getattr(ind, "metrics_summary", {})
    stab_view_key, stab_final_key, qual_view_key, qual_final_key = _primary_metric_keys(args)
    mean_view_stab = summary.get(stab_view_key)
    mean_view_qual = summary.get(qual_view_key)
    final_stab = summary.get(stab_final_key)
    final_qual = summary.get(qual_final_key)
    
    # Additional stability flavours for output
    #mean_view_stab_coassoc = summary.get("mean_view_stability_coassoc")
    #mean_view_stab_ccc = summary.get("mean_view_stability_CCC")
    mean_view_stab_jaccard = summary.get("mean_view_stability_jaccard")
    #final_stab_coassoc = summary.get("final_stability_coassoc")
    #final_stab_ccc = summary.get("final_stability_CCC")
    final_stab_jaccard = summary.get("final_stability_jaccard")
    mean_view_stability_MAT_CCC = summary.get("mean_view_stability_MAT_CCC")
    mean_view_stability_MAT_PAC = summary.get("mean_view_stability_MAT_PAC")
    final_stability_SUM_MAT = summary.get("final_stability_SUM_MAT", {})
    # Per-view MATLAB-style (lightweight) diagnostics from gather
    view_stabs_SUM_MAT = summary.get("view_stabs_SUM_MAT", [])
    # Normalize to a plain Python list for dill/pickle friendliness
    view_stabs_SUM_MAT = list(view_stabs_SUM_MAT) if view_stabs_SUM_MAT is not None else []



    if args.optimisation == 'single' and final_stab is None:
        final_stab = float(ind.fitness.values[0])
        mean_view_stab = final_stab
    view_stabs = getattr(ind, "view_stabs_per_view", None)
    view_quals = getattr(ind, "view_quals_per_view", None)


    metrics_dir = os.path.dirname(output_metrics_path) or "."

    # Save chosen individual's stabilities if available
    gen = getattr(args, "generation", 0)
    chosen = ind

    if hasattr(chosen, "view_stabs_per_view") and chosen.view_stabs_per_view:
        np.save(
            os.path.join(metrics_dir, f"chosen_view_stabs_gen{gen}.npy"),
            np.array(chosen.view_stabs_per_view, dtype=float)
        )

    # (Optional) Warn if bottleneck per-view quality is low
    if args.optimisation == 'multi' and view_quals:
        try:
            min_view_qual = float(np.min(view_quals))
            if min_view_qual < 0.3:
                print(f"[Fold {args.fold_index}] WARNING: bottleneck per-view quality is low (min view qual={min_view_qual:.3f}).")
        except Exception:
            pass


    # Capture original train subject IDs before modality filtering
    train_ids = df.loc[train_idx, args.subject_id_column].tolist()
    # Also capture test IDs (may be empty when n_folds == 1)
    test_ids = df.loc[test_idx, args.subject_id_column].tolist() if len(test_idx) > 0 else []

    #If in TEST mode, test label accuracy with true labels from synthetic data (aligned by subject IDs)
    if getattr(args, 'TEST', 'FALSE').upper() == "TRUE":
        print("TEST mode: computing Adjusted Rand Index against ground truth labels (aligned by subject IDs).")
        df_truth = pd.read_csv("/data/gpfs/projects/punim1993/students/Jente/multiclust/synthetic_multimodal_spartan.csv")
        truth_map = df_truth.set_index(args.subject_id_column)

        # Expanded ARI diagnostics per modality
        for i, pred in enumerate(train_individual_labels):
            mod = args.modalities[i]
            ref_ids_i = dict_final[mod][args.subject_id_column].to_numpy()
            true_labels = truth_map.loc[ref_ids_i, f"subgroup_m{i+1}"].to_numpy()

            # Map true labels (possibly strings) to stable integers
            uniq = np.unique(true_labels)
            l2i = {v: k for k, v in enumerate(uniq)}
            true_ints = np.array([l2i[v] for v in true_labels], dtype=int)
            pred = np.asarray(pred, dtype=int)

            # Safety: ensure lengths match
            if true_ints.shape[0] != pred.shape[0]:
                print(f"[DEBUG] Length mismatch for {mod}: true={true_ints.shape[0]} pred={pred.shape[0]}")

            # ARI
            ari = adjusted_rand_score(true_ints, pred)
            print(f"Adjusted Rand Index modality {i} ({mod}): {ari:.3f}")

            # --- Extra diagnostics ---
            try:
                # Show first 10 rows for sanity
                preview = list(zip(ref_ids_i[:10], true_ints[:10].tolist(), pred[:10].tolist()))
                print(f"[DEBUG] First 10 (id, true, pred) for {mod}: {preview}")

                # Confusion table (true vs pred)
                conf = pd.crosstab(pd.Series(true_ints, name='true'),
                                   pd.Series(pred, name='pred'),
                                   dropna=False)
                print(f"[DEBUG] Confusion table for {mod}:\n{conf}")
            except Exception as e:
                print(f"[DEBUG] Could not print diagnostics for {mod}: {e}")


    # Save outer metrics, including best individual's fitness and IDs
    os.makedirs(metrics_dir, exist_ok=True)
    # Per-view lists from summary (all stability types)
    view_stabs_ari = summary.get("view_stabs_ari")
    #view_stabs_coassoc = summary.get("view_stabs_coassoc")
    #view_stabs_ccc = summary.get("view_stabs_CCC")
    view_stabs_jaccard = summary.get("view_stabs_jaccard")
    view_stabs_SUM_MAT = summary.get("view_stabs_SUM_MAT", [])


    # Map primary mean-view key to its corresponding per-view key
    primary_views_stab = None
    if stab_view_key == "mean_view_stability_ari":
        primary_views_stab = view_stabs_ari
    #elif stab_view_key == "mean_view_stability_coassoc":
    #    primary_views_stab = view_stabs_coassoc
    #elif stab_view_key == "mean_view_stability_CCC":
    #    primary_views_stab = view_stabs_ccc
    elif stab_view_key == "mean_view_stability_jaccard":
        primary_views_stab = view_stabs_jaccard
    else:
        # Fallback to ARI per-view stability
        primary_views_stab = view_stabs_ari

    view_stabs_list = list(primary_views_stab) if primary_views_stab is not None else None
    view_quals_list = list(view_quals) if view_quals else None

    best_fitness_payload = {
        # Primary stability measures (aligned with GA objectives)
        'mean_view_stability': mean_view_stab,
        'final_stability': final_stab,

        # All stability variants for reporting
        'mean_view_stability_ari': summary.get("mean_view_stability_ari"),
        'final_stability_ari': summary.get("final_stability_ari"),
        #'mean_view_stability_coassoc': mean_view_stab_coassoc,
        #'final_stability_coassoc': final_stab_coassoc,
        #'mean_view_stability_CCC': mean_view_stab_ccc,
        #'final_stability_CCC': final_stab_ccc,
        'mean_view_stability_jaccard': mean_view_stab_jaccard,
        'final_stability_jaccard': final_stab_jaccard,
        # MATLAB-style stability diagnostics (from consensus_pac_ccc during GA evaluation)
        'mean_view_stability_MAT_CCC': mean_view_stability_MAT_CCC,
        'mean_view_stability_MAT_PAC': mean_view_stability_MAT_PAC,
        'final_stability_SUM_MAT': final_stability_SUM_MAT,
        'views_stability_SUM_MAT': view_stabs_SUM_MAT,

        # Per-view stability/quality (primary plus full breakdowns)
        'views_stability': view_stabs_list,
        'views_quality': view_quals_list,
        'views_stability_ari': list(view_stabs_ari) if view_stabs_ari is not None else None,
        #'views_stability_coassoc': list(view_stabs_coassoc) if view_stabs_coassoc is not None else None,
        #'views_stability_CCC': list(view_stabs_ccc) if view_stabs_ccc is not None else None,
        'views_stability_jaccard': list(view_stabs_jaccard) if view_stabs_jaccard is not None else None,
        'views_stability_SUM_MAT': list(view_stabs_SUM_MAT) if view_stabs_SUM_MAT is not None else None,

        # Quality measures
        'mean_view_quality': mean_view_qual,
        'final_quality': final_qual,

        # Existing view-wise quality and final composite quality metric
        'view_scores_per_view': view_scores_per_view,
        'final_quality_metric': final_score
    }
    metrics = {
        'data': dict_final,
        'ae_res': ae_res,
        'train_final_labels': train_final_labels,
        'train_individual_labels': train_individual_labels,
        'best_fitness': best_fitness_payload,
        'train_ids': train_ids,
        'test_ids': test_ids,
        'best_params': best_params
    }
    with open(output_metrics_path, 'wb') as f:
        dill.dump(metrics, f)
    print(f"Outer metrics saved to {output_metrics_path}")
    return



def do_merge(args):
    """
    Select the best metrics across folds and reapply to full data.
    Recalculate the stability across the entire data. 
    Then apply SVM if requested.
    """
    base_dir = os.path.abspath(getattr(args, 'base_dir', "."))
    results_root = _resolve_path(base_dir, "results")
    output_final_metrics_path = _resolve_path(base_dir, args.output_final_metrics) if args.output_final_metrics else None

    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)

    # Find all fold directories that contain a metrics.pkl
    metrics_files = sorted(glob.glob(os.path.join(results_root, 'fold*', 'metrics.pkl')))

    # Load metrics dynamically
    metrics = {}
    for metrics_file in metrics_files:
        fold_name = os.path.basename(os.path.dirname(metrics_file))  # e.g., 'fold0'
        with open(metrics_file, 'rb') as f:
            metrics[fold_name] = pickle.load(f)

    #########################
    # Identify the hyperparameters that were most consistently chosen across folds
    #########################
    # Collect parameters from folds
    param_list = [data["best_params"] for fold_name, data in metrics.items()]
    param_df = pd.DataFrame(param_list)

    reconstructed = {}
    for col in param_df.columns:
        s = param_df[col]

        # Column contains lists/tuples → per-position mode
        if s.apply(lambda x: isinstance(x, (list, tuple))).any():
            # turn each list into a row of a small DF
            tmp = pd.DataFrame(s.tolist())
            # mode per column, take first mode; convert back to list
            reconstructed[col] = tmp.mode().iloc[0].tolist()
        else:
            # scalar column → simple mode
            m = s.mode()
            reconstructed[col] = m.iloc[0] if not m.empty else None

    final_params = reconstructed
    print("Final selected parameters across folds:", final_params)


    # Retrain AE on the train split only:
    ae_data, subject_id_list, dict_final = preprocessing(
        df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities
    )

    # Assert identical subject order across modalities after preprocessing
    base_ids = dict_final[args.modalities[0]][args.subject_id_column].tolist()
    for m in args.modalities[1:]:
        assert dict_final[m][args.subject_id_column].tolist() == base_ids, \
            f"Subject-ID order mismatch between {args.modalities[0]} and {m} after preprocessing"

    if args.dim_reduction is None or args.dim_reduction.lower() == 'none':
        print("Skipping VAE and using preprocessed features as latent representations.")
        # Build VAE-like output from the fully preprocessed & scaled per-modality dataframes
        # Drop the subject ID column and convert to float32 arrays, preserving row order
        subj_col = args.subject_id_column
        ae_res = {}
        for mod in args.modalities:
            df_mod = dict_final[mod]
            X = df_mod.drop(columns=[subj_col]) if subj_col in df_mod.columns else df_mod
            X = X.to_numpy(dtype=np.float32, copy=True)
            ae_res[mod] = {"final_latent": X}
    elif args.dim_reduction.lower() == 'vae':
        print("Start running VAE")
        # Seed RNGs for reproducibility
        np.random.seed(42)
        random.seed(42)
        torch.manual_seed(42)
        ae_res = run_VAE_complete(
            ae_data,
            hidden_dims=args.hidden_dims,
            activation_functions=activation_functions,
            learning_rates=args.learning_rates,
            batch_sizes=args.batch_sizes,
            latent_dims=args.latent_dims
        )
    elif args.dim_reduction.lower() == "ae":
            print("Start running AE")
            # Seed RNGs for reproducibility
            np.random.seed(42)
            random.seed(42)
            torch.manual_seed(42)
            t_ae_start = time.time()
            ae_res = run_AE_complete(
                ae_data,
                hidden_dims=args.hidden_dims,
                activation_functions=activation_functions,
                learning_rates=args.learning_rates,
                batch_sizes=args.batch_sizes,
                latent_dims=args.latent_dims
            )
            t_ae_end = time.time()
            print(f"[Fold {args.fold_index} Boot {args.bootstrap_index}] AE nested CV took {t_ae_end - t_ae_start:.2f}s")
            # Extract embeddings once per modality (store as float32 to reduce memory)
            data_list = [np.asarray(ae_res[mod]['final_latent'], dtype=np.float32, copy=False)
                        for mod in args.modalities]
            # Free large autoencoder results to reduce peak memory before clustering
            del ae_res
            gc.collect()
    elif args.dim_reduction.lower() == "pca":
        print("Start running PCA")
        ae_res = {}
        for mod in args.modalities:
            df_mod = dict_final[mod]
            X = df_mod.drop(columns=['proc_subject_id']) if 'proc_subject_id' in df_mod.columns else df_mod
            pca = PCA(n_components=min(50, X.shape[1], X.shape[0]-1), random_state=42)
            X_pca = pca.fit_transform(X.to_numpy(dtype=np.float32, copy=True))
            ae_res[mod] = {"final_latent": X_pca}
        print("PCA dimensionality reduction completed.")
    else:
        raise ValueError(f"Unknown dim_reduction method: {args.dim_reduction}")


   # Prepare latent representations per modality
    ae_cluster = {mod: ae_res[mod]['final_latent'] for mod in args.modalities}
    data_list = [ae_cluster[mod] for mod in args.modalities]

    # Apply Parea with best parameters on full data 
    #labels, indiv_labels, view_scores_per_view, view_score_mean, final_score = parea_2_mv(
    #    data_list,
    #    subject_id_list=subject_id_list,
    #    inner_jobs=args.n_jobs,
    #    pre_inner_jobs=args.n_jobs,
    #    mincluster=args.mincluster,
    #    mincluster_n=args.mincluster_n,
    #    **final_params
    #)

    view_scores_per_view = None
    view_score_mean = None
    final_score = None


    # Calculate stability for this best individual in the entire data

    ## Resampling
    if not subject_id_list:
        raise ValueError("Subject ID list is empty; cannot perform stability estimation.")
    try:
        ref_subject_ids = next(ids for ids in subject_id_list if ids)
    except StopIteration:
        raise ValueError("No subject IDs available across modalities; cannot perform stability estimation.")
    full_subject_ids = ref_subject_ids
    n_samples = len(full_subject_ids)
    for ids in subject_id_list:
        if ids and len(ids) != n_samples:
            raise ValueError("Subject-ID lists per modality must have identical lengths for bootstrapping.")

    n_boot_full = getattr(args, 'n_bootstrap', 50)
    full_label_dicts_final = []
    full_label_dicts_views = [[] for _ in args.modalities]

    # Placeholders for aggregated full-data stability estimates
    full_final_stab_ari = None
    #full_final_stab_coassoc = None
    #full_final_stab_CCC = None
    full_final_stab_jaccard = None

    full_views_stab_ari = None
    #full_views_stab_coassoc = None
    #full_views_stab_CCC = None
    full_views_stab_jaccard = None

    # MATLAB-style consensus diagnostics (PAC/CCC) for full-data stability
    full_final_stab_SUM_MAT_full = None   # full dict from consensus_pac_ccc (may include matrix)
    full_v_stab_SUM_MAT_full = None        # lightweight dict: {PAC, CCC, meta}


    # Also keep the raw coassociation-based per-cluster lists if needed
    #full_final_coassoc = None         # list-of-floats per cluster (coassoc cluster stabilities)
    #full_views_coassoc = None         # list of lists, per view -> per cluster

    # Aggregate view-level MATLAB-style summary statistics (means across views)
    full_mean_view_stab_MAT_CCC = None
    full_mean_view_stab_MAT_PAC = None

    

    def _run_bootstrap(seed_value):
        np.random.seed(seed_value)
        random.seed(seed_value)
        if torch is not None:
            torch.manual_seed(seed_value)

        frac = 0.8
        m = max(3, int(round(frac * n_samples)))    # ensure enough points
        rng = np.random.default_rng(seed_value)
        idx = rng.choice(n_samples, size=m, replace=False)

        data_list_boot = [X[idx, :] for X in data_list]
        subject_id_list_boot = []
        for ids in subject_id_list:
            if not ids:
                subject_id_list_boot.append([])
            else:
                subject_id_list_boot.append([ids[i] for i in idx])

        boot_labels, boot_indiv_labels, _, _, _ = parea_2_mv(
            data_list_boot,
            subject_id_list=subject_id_list_boot,
            inner_jobs=args.n_jobs,
            pre_inner_jobs=args.n_jobs,
            mincluster=args.mincluster,
            mincluster_n=args.mincluster_n,
            **final_params
        )

        final_entry = {
            "orig_ids": subject_id_list_boot[0],
            "labels": boot_labels
        }
        per_view_entries = []
        for v in range(len(args.modalities)):
            per_view_entries.append({
                "orig_ids": subject_id_list_boot[v],
                "labels": boot_indiv_labels[v] if v < len(boot_indiv_labels) else []
            })
        return final_entry, per_view_entries

    seeds = [12345 + b for b in range(n_boot_full)]

    raw_workers = getattr(args, 'bootstrap_jobs', None)
    if raw_workers is None:
        raw_workers = getattr(args, 'n_jobs', 1)
        if raw_workers in (-1, None):
            raw_workers = os.cpu_count() or 1
    bootstrap_workers = max(1, min(int(raw_workers), n_boot_full)) if n_boot_full > 0 else 1


    final_labels = None
    indiv_labels = None

    def _consume_bootstrap(result_iter):
        nonlocal full_label_dicts_final, full_label_dicts_views
        nonlocal full_final_stab_ari, full_final_stab_jaccard
        nonlocal full_views_stab_ari, full_views_stab_jaccard
        # nonlocal full_final_stab_coassoc, full_final_stab_CCC, full_views_stab_coassoc, full_views_stab_CCC
        nonlocal full_final_stab_SUM_MAT_full
        nonlocal full_v_stab_SUM_MAT_full
        nonlocal full_mean_view_stab_MAT_CCC, full_mean_view_stab_MAT_PAC
        # nonlocal full_final_coassoc, full_views_coassoc
        nonlocal full_final_stab_SUM_MAT_full
        nonlocal final_labels, indiv_labels
        nonlocal view_scores_per_view, view_score_mean, final_score

        # First: just accumulate all bootstraps
        for b, (final_entry, per_view_entries) in enumerate(result_iter, 1):
            full_label_dicts_final.append(final_entry)
            for v, entry in enumerate(per_view_entries):
                full_label_dicts_views[v].append(entry)

        # After consuming all bootstraps, compute full-data stability estimates
        if full_label_dicts_final:
            # Coassociation (scalar) + CCC for final clustering
            # full_final_stab_coassoc, full_final_stab_CCC = coassociation_stability(
            #     full_label_dicts_final, label_key="labels"
            # )
            # full_final_coassoc = full_final_stab_coassoc

            # ARI and Jaccard for final clustering
            full_final_stab_ari = ari_stability_common_subjects(full_label_dicts_final, label_key="labels")
            full_final_stab_jaccard = jaccard_stability_common_subjects(full_label_dicts_final, label_key="labels")

            # MATLAB-style consensus diagnostics (PAC/CCC) for final clustering
            full_final_stab_SUM_MAT_full = consensus_pac_ccc(
                full_label_dicts_final,
                label_key="labels",
                return_consensus=True,
                return_ecdf=True,
            )

        if any(full_label_dicts_views):
            # Per-view coassociation (scalar) + CCC
            # view_results = [
            #     coassociation_stability(view_dicts, label_key="labels")
            #     for view_dicts in full_label_dicts_views
            # ]
            # Each element: (stab_coassoc_view, ccc_view)
            # full_views_stab_coassoc, full_views_stab_CCC = zip(*view_results)
            # full_views_stab_coassoc = [float(s) for s in full_views_stab_coassoc]
            # full_views_coassoc = full_views_stab_coassoc

            # MATLAB-style consensus diagnostics (PAC/CCC) per view (include consensus matrices)
            stab_v_SUM_MAT_full = []
            for view_dicts in full_label_dicts_views:
                diag_v = consensus_pac_ccc(
                    view_dicts,
                    label_key="labels",
                    return_consensus=True,
                    return_ecdf=False,
                )
                stab_v_SUM_MAT_full.append({
                    "consensus": diag_v.get("consensus", None),
                    "union_ids": diag_v.get("union_ids", None),
                    "PAC": diag_v.get("PAC", np.nan),
                    "CCC": diag_v.get("CCC", np.nan),
                    "meta": diag_v.get("meta", {}),
                })

            full_v_stab_SUM_MAT_full = stab_v_SUM_MAT_full

            if full_v_stab_SUM_MAT_full:
                full_mean_view_stab_MAT_CCC = float(np.nanmean([d.get("CCC", np.nan) for d in full_v_stab_SUM_MAT_full]))
                full_mean_view_stab_MAT_PAC = float(np.nanmean([d.get("PAC", np.nan) for d in full_v_stab_SUM_MAT_full]))
            else:
                full_mean_view_stab_MAT_CCC = -3  # If no views, set to error code
                full_mean_view_stab_MAT_PAC = -3

            # Per-view ARI and Jaccard
            full_views_stab_ari = [
                ari_stability_common_subjects(view_dicts, label_key="labels")
                for view_dicts in full_label_dicts_views
            ]
            full_views_stab_jaccard = [
                jaccard_stability_common_subjects(view_dicts, label_key="labels")
                for view_dicts in full_label_dicts_views
            ]

            def _align_labels_to_ids(union_ids, labels, target_ids, fill_value=-1):
                if union_ids is None or labels is None:
                    return None
                idx_map = {sid: i for i, sid in enumerate(union_ids)}
                aligned = np.full(len(target_ids), fill_value, dtype=int)
                for j, sid in enumerate(target_ids):
                    i = idx_map.get(sid)
                    if i is not None:
                        aligned[j] = labels[i]
                if np.any(aligned == fill_value):
                    missing = int(np.sum(aligned == fill_value))
                    warnings.warn(f"{missing} subjects missing in consensus labels; filled with {fill_value}.")
                return aligned

            def _k1_quality(X_or_D, precomputed=False, method="auto", random_state=42):
                rng = np.random.RandomState(random_state)

                def _pairwise_dists(A):
                    from sklearn.metrics import pairwise_distances
                    D = pairwise_distances(np.asarray(A))
                    iu = np.triu_indices_from(D, k=1)
                    return D[iu]

                if method in ("dip", "auto"):
                    try:
                        import diptest
                        if precomputed:
                            D = np.asarray(X_or_D, dtype=float)
                            iu = np.triu_indices_from(D, k=1)
                            d = D[iu]
                        else:
                            d = _pairwise_dists(X_or_D)
                        d = d[np.isfinite(d)]
                        if d.size < 3:
                            return 1.0
                        _, p = diptest.diptest(np.asarray(d, dtype=float))
                        return float(max(0.0, min(1.0, p)))
                    except Exception:
                        pass

                from sklearn.neighbors import NearestNeighbors
                if precomputed:
                    D = np.asarray(X_or_D, dtype=float)
                    n = D.shape[0]
                    if n < 3:
                        return 1.0
                    J = np.eye(n) - np.ones((n, n)) / n
                    B = -0.5 * J.dot(D ** 2).dot(J)
                    evals, evecs = np.linalg.eigh(B)
                    idx = np.argsort(evals)[::-1]
                    evals, evecs = evals[idx], evecs[:, idx]
                    pos = evals > 1e-12
                    if not np.any(pos):
                        X = D[:, :1]
                    else:
                        m = min(10, pos.sum())
                        X = evecs[:, pos][:, :m] * np.sqrt(evals[pos][:m])
                else:
                    X = np.asarray(X_or_D, dtype=float)

                n, d = X.shape
                if n < 3:
                    return 1.0
                m = min(50, n // 2)
                idx = rng.choice(n, size=m, replace=False)
                X_m = X[idx]
                mins, maxs = X.min(axis=0), X.max(axis=0)
                U = rng.uniform(low=mins, high=maxs, size=(m, d))
                nn = NearestNeighbors(n_neighbors=1).fit(X)
                w, _ = nn.kneighbors(X_m, return_distance=True)
                u, _ = nn.kneighbors(U, return_distance=True)
                W = np.power(w.ravel(), d).sum()
                Uv = np.power(u.ravel(), d).sum()
                H = Uv / (Uv + W + 1e-12)
                return float(max(0.0, min(1.0, 1.0 - H)))

            def _silhouette_norm(mat, labels, precomputed=False):
                labels = np.asarray(labels)
                if len(np.unique(labels)) <= 1:
                    result = 0.0
                else:
                    try:
                        if precomputed:
                            sil = silhouette_score(mat, labels, metric="precomputed")
                        else:
                            sil = silhouette_score(mat, labels)
                        result = (sil + 1.0) / 2.0
                    except Exception:
                        result = 0.0
                return float(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0))

            def _ch_norm(X, labels):
                labels = np.asarray(labels)
                if len(np.unique(labels)) <= 1:
                    result = 0.0
                else:
                    try:
                        ch = calinski_harabasz_score(X, labels)
                        result = ch / (ch + 1.0)
                    except Exception:
                        result = 0.0
                return float(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0))

            def _db_inv(X, labels):
                labels = np.asarray(labels)
                if len(np.unique(labels)) <= 1:
                    result = 0.0
                else:
                    try:
                        db = davies_bouldin_score(X, labels)
                        result = 1.0 / (1.0 + db)
                    except Exception:
                        result = 0.0
                return float(np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0))

            def _composite_view_quality(X, labels):
                labels = np.asarray(labels)
                if len(np.unique(labels)) <= 1:
                    return float(_k1_quality(X, precomputed=False, method="auto"))
                s = _silhouette_norm(X, labels, precomputed=False)
                c = _ch_norm(X, labels)
                d = _db_inv(X, labels)
                return (s + c + d) / 3.0

            def _classical_mds(D, p=10):
                D = np.asarray(D, dtype=float)
                n = D.shape[0]
                if n == 0:
                    return np.zeros((0, 0), dtype=float)
                J = np.eye(n) - np.ones((n, n)) / n
                D2 = D ** 2
                B = -0.5 * J.dot(D2).dot(J)
                evals, evecs = np.linalg.eigh(B)
                idx = np.argsort(evals)[::-1]
                evals, evecs = evals[idx], evecs[:, idx]
                pos = evals > 1e-12
                if not np.any(pos):
                    return np.zeros((n, 1), dtype=float)
                evals_pos = evals[pos]
                evecs_pos = evecs[:, pos]
                m = min(p, evecs_pos.shape[1])
                X = evecs_pos[:, :m] * np.sqrt(evals_pos[:m])
                return X

            ### Ensemble across bootstraps final labels
            M_final = full_final_stab_SUM_MAT_full.get("consensus", None)
            union_ids_final = full_final_stab_SUM_MAT_full.get("union_ids", None)
            if M_final is None:
                warnings.warn("Final consensus matrix is empty; skipping ensemble labels.")
                final_labels = None
                final_labels_union = None
            else:
                M_final = np.asarray(M_final, dtype=float)
                assert M_final.ndim == 2 and M_final.shape[0] == M_final.shape[1], "Consensus must be square."
                M_final = (M_final + M_final.T) / 2.0
                np.fill_diagonal(M_final, 1.0)
                D_final = 1.0 - M_final
                dvec_final = squareform(D_final, checks=False)
                final_linkage = final_params.get("pre_method", "average")
                Z_final = hierarchy.linkage(dvec_final, method=final_linkage)
                print("Hierarchical clustering with k=", final_params.get("k_final", 2))
                final_labels_union = hierarchy.cut_tree(
                    Z_final, n_clusters=final_params.get("k_final", 2)
                ).reshape(-1)
                final_labels = _align_labels_to_ids(
                    union_ids_final, final_labels_union, base_ids
                )

            ### Ensemble across bootstraps individual labels (saved in stab_v_SUM_MAT_full)
            indiv_labels = []
            indiv_labels_union = []
            view_scores_per_view = [None] * len(full_v_stab_SUM_MAT_full)
            for i in range(len(full_v_stab_SUM_MAT_full)):
                modality = full_v_stab_SUM_MAT_full[i]
                mod_name = args.modalities[i]
                M_view = modality.get("consensus", None)
                union_ids_view = modality.get("union_ids", None)
                if M_view is None:
                    warnings.warn(f"Consensus matrix missing for view {mod_name}; skipping.")
                    indiv_labels.append(None)
                    indiv_labels_union.append(None)
                    continue
                M_view = np.asarray(M_view, dtype=float)
                assert M_view.ndim == 2 and M_view.shape[0] == M_view.shape[1], "Consensus must be square."
                M_view = (M_view + M_view.T) / 2.0
                np.fill_diagonal(M_view, 1.0)
                D_view = 1.0 - M_view
                dvec_view = squareform(D_view, checks=False)
                view_linkage = final_params.get(f"c_{i+1}_method", "average")
                Z_view = linkage(dvec_view, method=view_linkage)
                print("Hierarchical clustering with k=", final_params.get("c_"+str(i+1)+"_k", 2))
                labels_i_union = hierarchy.cut_tree(
                    Z_view, n_clusters=final_params.get("c_"+str(i+1)+"_k", 2)
                ).reshape(-1)
                indiv_labels_union.append(labels_i_union)
                aligned_i = _align_labels_to_ids(
                    union_ids_view, labels_i_union, base_ids
                )
                indiv_labels.append(aligned_i)

            for i, labels_i in enumerate(indiv_labels):
                if labels_i is None:
                    continue
                labels_i = np.asarray(labels_i)
                valid_mask = labels_i >= 0
                if not np.any(valid_mask):
                    warnings.warn(f"No valid labels for view {args.modalities[i]}; skipping quality.")
                    continue
                X_sub = data_list[i][valid_mask]
                labs_sub = labels_i[valid_mask]
                view_scores_per_view[i] = float(_composite_view_quality(X_sub, labs_sub))

            valid_view_scores = [v for v in view_scores_per_view if v is not None]
            view_score_mean = float(np.mean(valid_view_scores)) if valid_view_scores else None

            final_score = None
            if final_labels is not None and M_final is not None and union_ids_final is not None:
                final_labels_arr = np.asarray(final_labels)
                valid_mask = final_labels_arr >= 0
                if np.any(valid_mask):
                    target_ids = [base_ids[i] for i in np.where(valid_mask)[0]]
                    labels_valid = final_labels_arr[valid_mask]
                    idx_map = {sid: i for i, sid in enumerate(union_ids_final)}
                    idxs = []
                    labs_sub = []
                    for sid, lab in zip(target_ids, labels_valid):
                        idx = idx_map.get(sid)
                        if idx is not None:
                            idxs.append(idx)
                            labs_sub.append(lab)
                    if len(idxs) >= 2:
                        M_sub = M_final[np.ix_(idxs, idxs)]
                        D_sub = 1.0 - M_sub
                        labs_sub = np.asarray(labs_sub)
                        if len(np.unique(labs_sub)) <= 1:
                            final_score = float(_k1_quality(D_sub, precomputed=True, method="auto"))
                        else:
                            sil_final = _silhouette_norm(D_sub, labs_sub, precomputed=True)
                            X_mds = _classical_mds(D_sub, p=min(10, D_sub.shape[0] - 1))
                            ch_final = _ch_norm(X_mds, labs_sub)
                            dbi_final = _db_inv(X_mds, labs_sub)
                            final_score = float((sil_final + ch_final + dbi_final) / 3.0)





            
    if n_boot_full > 0:
        if bootstrap_workers == 1:
            _consume_bootstrap(_run_bootstrap(seed) for seed in seeds)
        else:
            with ThreadPoolExecutor(max_workers=bootstrap_workers) as executor:
                _consume_bootstrap(executor.map(_run_bootstrap, seeds))

    print(f"Completed bootstrap for full-data stability estimation.")
    if full_final_stab_ari is not None:
        print(f"Full-data final clustering stability (ARI): {full_final_stab_ari:.4f}")
    #if full_final_stab_coassoc is not None:
    #    print(f"Full-data final clustering stability (coassoc mean): {full_final_stab_coassoc:.4f}")
    #if full_final_stab_CCC is not None:
    #    print(f"Full-data final clustering stability (CCC): {full_final_stab_CCC:.4f}")
    if full_final_stab_jaccard is not None:
        print(f"Full-data final clustering stability (Jaccard): {full_final_stab_jaccard:.4f}")
    if full_final_stab_SUM_MAT_full is not None:
        print(f"Full-data final clustering stability (CCC MAT): {full_final_stab_SUM_MAT_full['CCC']:.4f}")
    if full_final_stab_SUM_MAT_full is not None:
        print(f"Full-data final clustering stability (PAC MAT): {full_final_stab_SUM_MAT_full['PAC']:.4f}")
    if full_views_stab_ari is not None:
        print(f"Full-data per-view clustering stabilities (ARI): {[f'{s:.4f}' for s in full_views_stab_ari]}")
    #if full_views_stab_coassoc is not None:
    #    print(f"Full-data per-view clustering stabilities (coassoc mean): {[f'{s:.4f}' for s in full_views_stab_coassoc]}")
    #if full_views_stab_CCC is not None:
    #    print(f"Full-data per-view clustering stabilities (CCC): {[f'{s:.4f}' for s in full_views_stab_CCC]}")
    if full_views_stab_jaccard is not None:
        print(f"Full-data per-view clustering stabilities (Jaccard): {[f'{s:.4f}' for s in full_views_stab_jaccard]}")
    if full_v_stab_SUM_MAT_full is not None:
        ccc_mat_per_view = [d.get("CCC", np.nan) for d in (full_v_stab_SUM_MAT_full or [])]
        print(f"Full-data per-view clustering stability (CCC MAT): {[f'{v:.4f}' for v in ccc_mat_per_view]}")
    if full_v_stab_SUM_MAT_full is not None:
        pac_mat_per_view = [d.get("PAC", np.nan) for d in (full_v_stab_SUM_MAT_full or [])]
        print(f"Full-data per-view clustering stability (PAC MAT): {[f'{v:.4f}' for v in pac_mat_per_view]}")



    
    # Decide which stability metric is primary based on GA objectives
    stab_view_key, stab_final_key, _, _ = _primary_metric_keys(args)

    # Map scalar final stability according to chosen objective
    if stab_final_key == "final_stability_ari":
        final_stability_primary = full_final_stab_ari
    #elif stab_final_key == "final_stability_coassoc":
    #    final_stability_primary = full_final_stab_coassoc
    #elif stab_final_key == "final_stability_CCC":
    #    final_stability_primary = full_final_stab_CCC
    elif stab_final_key == "final_stability_jaccard":
        final_stability_primary = full_final_stab_jaccard
    else:
        # Fallback
        final_stability_primary = full_final_stab_ari

    # Map per-view stability list according to chosen objective
    if stab_view_key == "mean_view_stability_ari":
        per_view_stabilities_primary = full_views_stab_ari
    #elif stab_view_key == "mean_view_stability_coassoc":
    #    per_view_stabilities_primary = full_views_stab_coassoc
    #elif stab_view_key == "mean_view_stability_CCC":
    #    per_view_stabilities_primary = full_views_stab_CCC
    elif stab_view_key == "mean_view_stability_jaccard":
        per_view_stabilities_primary = full_views_stab_jaccard
    else:
        per_view_stabilities_primary = full_views_stab_ari

    # Safe defaults if no bootstraps were run
    if per_view_stabilities_primary is None:
        per_view_stabilities_primary = []
    mean_view_stability_primary = float(np.mean(per_view_stabilities_primary)) if per_view_stabilities_primary else None
    min_view_stability_primary = float(np.min(per_view_stabilities_primary)) if per_view_stabilities_primary else None



    if not output_final_metrics_path:
        raise ValueError("For merge mode, --output_final_metrics must be specified")
    final_metrics_dir = os.path.dirname(output_final_metrics_path) or "."
    os.makedirs(final_metrics_dir, exist_ok=True)


    ########## ------ #########
    # SVM classification
    ########## ------ #########

    if getattr(args, 'DO_SVM', 'FALSE').upper() == 'TRUE':

        # Prepare  data for SVM. We need to combine the data from all modalities and drop the SRC_SUBJECT_ID column
        X_train_list = []
        for mod in args.modalities:
            df_mod = dict_final[mod]
            X_mod = df_mod.drop(columns=[args.subject_id_column], errors='ignore')
            X_train_list.append(X_mod.reset_index(drop=True))
        X_train = pd.concat(X_train_list, axis=1)
        

        # Use the final cluster labels as training labels
        clusters = final_labels
        clusters_indiv = indiv_labels

        if clusters is None:
            print("Warning: Final consensus labels missing; skipping SVM classification.")
            results = None
            final_model = None
            metrics_merged = {
                'data': dict_final,
                'ae_res': ae_res,
                'final_labels': final_labels,
                'individual_labels': indiv_labels,
                'final_params': final_params,
                'view_scores_per_view': view_scores_per_view,
                'view_quality_mean': view_score_mean,
                'final_quality': final_score,

                # Primary stability metrics
                'final_stability': final_stability_primary,
                'per_view_stabilities': per_view_stabilities_primary,
                'mean_view_stability': mean_view_stability_primary,
                'min_view_stability': min_view_stability_primary,

                # All stability variants
                'final_stability_ari': full_final_stab_ari,
                #'final_stability_coassoc': full_final_stab_coassoc,
                #'final_stability_CCC': full_final_stab_CCC,
                'final_stability_jaccard': full_final_stab_jaccard,
                'per_view_stabilities_ari': full_views_stab_ari,
                #'per_view_stabilities_coassoc': full_views_stab_coassoc,
                #'per_view_stabilities_CCC': full_views_stab_CCC,
                'per_view_stabilities_jaccard': full_views_stab_jaccard,
                # MATLAB-style consensus diagnostics (PAC/CCC)
                'final_stability_SUM_MAT_full': full_final_stab_SUM_MAT_full,
                'per_view_stabilities_SUM_MAT_full': full_v_stab_SUM_MAT_full,
                'mean_view_stability_MAT_CCC': full_mean_view_stab_MAT_CCC,
                'mean_view_stability_MAT_PAC': full_mean_view_stab_MAT_PAC

            }
            with open(output_final_metrics_path, 'wb') as f:
                dill.dump(metrics_merged, f)
            print(f"Outer metrics saved to {output_final_metrics_path}")
            return

        clusters = np.asarray(clusters)
        valid_mask = clusters >= 0
        clusters_valid = clusters[valid_mask]
        if len(set(clusters_valid)) < 2:
            print("Warning: Less than 2 clusters found in final clustering; skipping SVM classification.")
            results = None
            final_model = None
            # Save outer metrics without SVM results
            metrics_merged = {
                'data': dict_final,
                'ae_res': ae_res,
                'final_labels': final_labels,
                'individual_labels': indiv_labels,
                'final_params': final_params,
                'view_scores_per_view': view_scores_per_view,
                'view_quality_mean': view_score_mean,
                'final_quality': final_score,

                # Primary stability metrics
                'final_stability': final_stability_primary,
                'per_view_stabilities': per_view_stabilities_primary,
                'mean_view_stability': mean_view_stability_primary,
                'min_view_stability': min_view_stability_primary,

                # All stability variants
                'final_stability_ari': full_final_stab_ari,
                #'final_stability_coassoc': full_final_stab_coassoc,
                #'final_stability_CCC': full_final_stab_CCC,
                'final_stability_jaccard': full_final_stab_jaccard,
                'per_view_stabilities_ari': full_views_stab_ari,
                #'per_view_stabilities_coassoc': full_views_stab_coassoc,
                #'per_view_stabilities_CCC': full_views_stab_CCC,
                'per_view_stabilities_jaccard': full_views_stab_jaccard,
                # MATLAB-style consensus diagnostics (PAC/CCC)
                'final_stability_SUM_MAT_full': full_final_stab_SUM_MAT_full,
                'per_view_stabilities_SUM_MAT_full': full_v_stab_SUM_MAT_full,
                'mean_view_stability_MAT_CCC': full_mean_view_stab_MAT_CCC,
                'mean_view_stability_MAT_PAC': full_mean_view_stab_MAT_PAC

            }


            with open(output_final_metrics_path, 'wb') as f:
                dill.dump(metrics_merged, f)
            print(f"Outer metrics saved to {output_final_metrics_path}")
            return
        else:
            print(f"Training SVM classifier on {len(set(clusters_valid))} clusters.")
            X_train_valid = X_train.loc[valid_mask].reset_index(drop=True)
            y_train = pd.Series(clusters_valid, name='cluster')

            results, final_model = SVM_nested_cv(X_train_valid, y_train)

            # --- NEW: persist feature names + (optional) traceability indices for alignment ---
            svm_feature_names = list(X_train_valid.columns) if isinstance(X_train_valid, pd.DataFrame) else None
            # Useful if you ever want to map OOF rows back to original rows on the server
            svm_train_index = X_train.loc[valid_mask].index.to_numpy() if hasattr(X_train, "loc") else None


            results_modalities = []
            final_models_modalities = []
            svm_feature_names_modalities = []     # NEW
            svm_train_index_modalities = []       # NEW

            for i, mod in enumerate(args.modalities):
                if clusters_indiv is None or clusters_indiv[i] is None:
                    print(f"Warning: No individual labels for modality {mod}; skipping SVM classification for this modality.")
                    results_mod = None
                    final_model_mod = None
                    results_modalities.append(results_mod)
                    final_models_modalities.append(final_model_mod)
                    svm_feature_names_modalities.append(None)   # NEW
                    svm_train_index_modalities.append(None)     # NEW
                    continue

                labels_mod = np.asarray(clusters_indiv[i])
                valid_mask_mod = labels_mod >= 0
                labels_mod_valid = labels_mod[valid_mask_mod]
                print(f"Training SVM classifier on {len(set(labels_mod_valid))} clusters for modality {mod}")

                if len(set(labels_mod_valid)) < 2:
                    print("Warning: Less than 2 clusters found in individual modality clustering; skipping SVM classification for this modality.")
                    results_mod = None
                    final_model_mod = None
                    results_modalities.append(results_mod)
                    final_models_modalities.append(final_model_mod)
                    svm_feature_names_modalities.append(None)   # NEW
                    svm_train_index_modalities.append(None)     # NEW
                else:
                    # Get training data
                    df_mod = dict_final[mod]
                    X_mod = df_mod.drop(columns=[args.subject_id_column], errors='ignore')
                    X_train_mod = X_mod.loc[valid_mask_mod].reset_index(drop=True)

                    y_train_mod = pd.Series(labels_mod_valid, name='cluster')

                    results_mod, final_model_mod = SVM_nested_cv(X_train_mod, y_train_mod)

                    results_modalities.append(results_mod)
                    final_models_modalities.append(final_model_mod)

                    # --- NEW: save feature order + traceability indices for this modality ---
                    svm_feature_names_modalities.append(list(X_train_mod.columns) if isinstance(X_train_mod, pd.DataFrame) else None)
                    svm_train_index_modalities.append(X_mod.loc[valid_mask_mod].index.to_numpy() if hasattr(X_mod, "loc") else None)


            # --- NEW: pack SVM outputs into plain python for portability (optional but recommended) ---
            def _pack_svm_results(res):
                """Convert pandas objects inside SVM results to plain python containers (more robust across machines)."""
                if res is None:
                    return None

                out = dict(res)

                # OOF uncertainty table (DataFrame -> dict of lists)
                oof = out.get("oof_uncertainty", None)
                if isinstance(oof, pd.DataFrame):
                    out["oof_uncertainty"] = oof.to_dict(orient="list")

                # Feature importances (Series -> dict)
                fim = out.get("feature_importance_mean", None)
                if isinstance(fim, pd.Series):
                    out["feature_importance_mean"] = fim.to_dict()

                fis = out.get("feature_importance_std", None)
                if isinstance(fis, pd.Series):
                    out["feature_importance_std"] = fis.to_dict()

                return out

            svm_results_packed = _pack_svm_results(results)
            svm_results_modalities_packed = [_pack_svm_results(r) for r in results_modalities]


            # Save outer metrics, including best individual's fitness and IDs
            metrics_merged = {
                'data': dict_final,
                'ae_res': ae_res,
                'final_labels': final_labels,
                'individual_labels': indiv_labels,
                'final_params': final_params,
                'view_scores_per_view': view_scores_per_view,
                'view_quality_mean': view_score_mean,
                'final_quality': final_score,

                # Primary stability metrics
                'final_stability': final_stability_primary,
                'per_view_stabilities': per_view_stabilities_primary,
                'mean_view_stability': mean_view_stability_primary,
                'min_view_stability': min_view_stability_primary,

                # All stability variants
                'final_stability_ari': full_final_stab_ari,
                'final_stability_jaccard': full_final_stab_jaccard,
                'per_view_stabilities_ari': full_views_stab_ari,
                'per_view_stabilities_jaccard': full_views_stab_jaccard,
                'final_stability_SUM_MAT_full': full_final_stab_SUM_MAT_full,
                'per_view_stabilities_SUM_MAT_full': full_v_stab_SUM_MAT_full,
                'mean_view_stability_MAT_CCC': full_mean_view_stab_MAT_CCC,
                'mean_view_stability_MAT_PAC': full_mean_view_stab_MAT_PAC,

                # --- SVM outputs (raw) ---
                'svm_results': results,
                'svm_final_model': final_model,
                'svm_results_modalities': results_modalities,
                'svm_final_models_modalities': final_models_modalities,

                # --- NEW: SVM metadata needed on laptop ---
                'svm_feature_names': svm_feature_names,
                'svm_train_index': svm_train_index,
                'svm_feature_names_modalities': svm_feature_names_modalities,
                'svm_train_index_modalities': svm_train_index_modalities,

                # --- NEW: packed/portable copies (optional but handy) ---
                'svm_results_packed': svm_results_packed,
                'svm_results_modalities_packed': svm_results_modalities_packed,
            }

            with open(output_final_metrics_path, 'wb') as f:
                dill.dump(metrics_merged, f)

            print(f"Outer metrics saved to {output_final_metrics_path}")
            return

    else:
        print("SVM classification not requested; skipping SVM step.")
        metrics_merged = {
            'data': dict_final,
            'ae_res': ae_res,
            'final_labels': final_labels,
            'individual_labels': indiv_labels,
            'final_params': final_params,
            'view_scores_per_view': view_scores_per_view,
            'view_quality_mean': view_score_mean,
            'final_quality': final_score,

            # Primary stability metrics
            'final_stability': final_stability_primary,
            'per_view_stabilities': per_view_stabilities_primary,
            'mean_view_stability': mean_view_stability_primary,
            'min_view_stability': min_view_stability_primary,

            # All stability variants
            'final_stability_ari': full_final_stab_ari,
            #'final_stability_coassoc': full_final_stab_coassoc,
            #'final_stability_CCC': full_final_stab_CCC,
            'final_stability_jaccard': full_final_stab_jaccard,
            'per_view_stabilities_ari': full_views_stab_ari,
            #'per_view_stabilities_coassoc': full_views_stab_coassoc,
            #'per_view_stabilities_CCC': full_views_stab_CCC,
            'per_view_stabilities_jaccard': full_views_stab_jaccard,
            # MATLAB-style consensus diagnostics (PAC/CCC)
            'final_stability_SUM_MAT_full': full_final_stab_SUM_MAT_full,
            'per_view_stabilities_SUM_MAT_full': full_v_stab_SUM_MAT_full,
            'mean_view_stability_MAT_CCC': full_mean_view_stab_MAT_CCC,
            'mean_view_stability_MAT_PAC': full_mean_view_stab_MAT_PAC
        }
        with open(output_final_metrics_path, 'wb') as f:
            dill.dump(metrics_merged, f)
        print(f"Outer metrics saved to {output_final_metrics_path}")
        return



# --- Mode: init population ---
def do_init(args):
    """
    Initialize GA population and save to args.population_file
    """
    base_dir = os.path.abspath(getattr(args, "base_dir", "."))
    ga_root = _ga_root(base_dir, args.fold_index if hasattr(args, "fold_index") else 0)
    population_file = _resolve_path(base_dir, args.population_file) or os.path.join(
        ga_root, f"population_init_fold{getattr(args, 'fold_index', 0)}.pkl"
    )
    # 2) Build toolbox with original gene definitions
    toolbox = base.Toolbox()
    n_views = len(args.modalities)
    fusion_methods = list(args.fusion_methods)
    if not fusion_methods:
        raise ValueError("At least one fusion method must be supplied via --fusion_methods.")
    #linkages = ['complete','average','weighted']
    linkages = list(args.linkages)
    k_min, k_max = args.k_min, args.k_max

         # Seed RNGs so each fold’s init is unique
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        if torch is not None:
            torch.manual_seed(args.seed)


    names = []

    for i in range(n_views):
        toolbox.register(f"c_{i+1}_k", random.randint, k_min, k_max)
        names.append(f"c_{i+1}_k")
        toolbox.register(f"c_{i+1}_method", random.choice, linkages)
        names.append(f"c_{i+1}_method")
    toolbox.register("pre_method", random.choice, linkages)
    names.append("pre_method")
    toolbox.register("k_final", random.randint, k_min, k_max)
    names.append("k_final")
    toolbox.register("fusion_method", random.choice, fusion_methods)
    names.append("fusion_method")

    # 3) Create individual & population initializers
    to_pass = tuple(getattr(toolbox, name) for name in names)
    toolbox.register("individual", tools.initCycle, creator.Individual, to_pass, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 4) Generate and save initial population
    pop = toolbox.population(n=args.n_population)
    # — Attach gene_names metadata to each individual so convert_to_parameters can use it —
    for ind in pop:
        ind.gene_names = names
    
    os.makedirs(os.path.dirname(population_file) or '.', exist_ok=True)
    with open(population_file, 'wb') as f:
        dill.dump(pop, f)
    print(f"Initial population ({len(pop)}) saved to {population_file}")
    return



def do_test1(args):
    """
    Test mode 1: Run only KMeans on preprocessed data, skipping AE and Parea.
    """
    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        # No CV split: use all rows for training to allow fast synthetic-data tests
        train_df = df.reset_index(drop=True)
    else:
        raise ValueError("TEST mode only supports n_folds=1 for fast testing on synthetic data")

    # Preprocess data
    ae_data, subject_id_list, dict_final = preprocessing(
        train_df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities
    )

    print("TEST mode 1: Running KMeans on preprocessed data...")
    # Extract preprocessed & scaled per-modality dataframes
    data_list = [
        dict_final[mod].drop(columns=[args.subject_id_column]).to_numpy(dtype=np.float32, copy=True)
        for mod in args.modalities
    ]

    # Keep subject order from preprocessing (should be identical across modalities)
    # Use the IDs from the first modality as reference order
    ref_ids = dict_final[args.modalities[0]][args.subject_id_column].to_numpy()

    # Run KMeans clustering independently on each modality and store labels
    labels_list = []
    for i, X in enumerate(data_list):
        # Fixed cluster counts for synthetic test (adjust as needed)
        if i == 0:
            k = 3
        elif i == 1:
            k = 4
        elif i == 2:
            k = 2
        km = KMeans(n_clusters=k, random_state=42)
        lab = km.fit_predict(X)
        labels_list.append(lab)
        counts = np.bincount(lab)
        print(f"Modality {args.modalities[i]}: KMeans found {len(np.unique(lab))} clusters with sizes {counts}")

    # If in TEST mode, compute ARI against ground truth labels with proper alignment
    if getattr(args, 'TEST', 'FALSE').upper() == "TRUE":
        print("TEST mode: computing Adjusted Rand Index against ground truth labels.")
        df_truth = pd.read_csv("/data/gpfs/projects/punim1993/students/Jente/multiclust/synthetic_multimodal_spartan.csv")
        # Align truth to the exact subject order used during preprocessing
        # Assumes the same subject ID column exists in df_truth
        truth_map = df_truth.set_index(args.subject_id_column)
        # Build truth arrays in the same order as ref_ids
        true_m1 = truth_map.loc[ref_ids, "subgroup_m1"].to_numpy()
        true_m2 = truth_map.loc[ref_ids, "subgroup_m2"].to_numpy()
        true_m3 = truth_map.loc[ref_ids, "subgroup_m3"].to_numpy()
        truth_cols = [true_m1, true_m2, true_m3]

        for i, pred in enumerate(labels_list):
            true_labels = truth_cols[i]
            # Map true labels (strings) to integers deterministically
            uniq = np.unique(true_labels)
            l2i = {name: idx for idx, name in enumerate(uniq)}
            true_ints = np.array([l2i[v] for v in true_labels], dtype=int)
            ari = adjusted_rand_score(true_ints, pred)
            print(f"Adjusted Rand Index modality {i} ({args.modalities[i]}): {ari:.3f}")
    return


def do_test2(args):
    """
    Test mode 2: Run only Spectral Clustering on preprocessed data, skipping AE and Parea.
    """
    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        # No CV split: use all rows for training to allow fast synthetic-data tests
        train_df = df.reset_index(drop=True)
    else:
        raise ValueError("TEST mode only supports n_folds=1 for fast testing on synthetic data")

    # Preprocess data
    ae_data, subject_id_list, dict_final = preprocessing(
        train_df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities
    )

    print("TEST mode 2: Running Spectral clustering on preprocessed data...")
    # Extract preprocessed & scaled per-modality dataframes
    data_list = [
        dict_final[mod].drop(columns=[args.subject_id_column]).to_numpy(dtype=np.float32, copy=True)
        for mod in args.modalities
    ]

    # Keep subject order from preprocessing (should be identical across modalities)
    # Use the IDs from the first modality as reference order
    ref_ids = dict_final[args.modalities[0]][args.subject_id_column].to_numpy()

    # Run KMeans clustering independently on each modality and store labels
    labels_list = []
    for i, X in enumerate(data_list):
        # Fixed cluster counts for synthetic test (adjust as needed)
        if i == 0:
            k = 3
        elif i == 1:
            k = 4
        elif i == 2:
            k = 2
        km = SpectralClustering(n_clusters=k, n_init=10, gamma=1.0, n_neighbors=10, eigen_tol=0.0, degree=3, coef0=1, verbose=False, assign_labels='kmeans', affinity='nearest_neighbors')
        lab = km.fit_predict(X)
        labels_list.append(lab)
        counts = np.bincount(lab)
        print(f"Modality {args.modalities[i]}: Spectral Clustering found {len(np.unique(lab))} clusters with sizes {counts}")

    # If in TEST mode, compute ARI against ground truth labels with proper alignment
    if getattr(args, 'TEST', 'FALSE').upper() == "TRUE":
        print("TEST mode: computing Adjusted Rand Index against ground truth labels.")
        df_truth = pd.read_csv("/data/gpfs/projects/punim1993/students/Jente/multiclust/synthetic_multimodal_spartan.csv")
        # Align truth to the exact subject order used during preprocessing
        # Assumes the same subject ID column exists in df_truth
        truth_map = df_truth.set_index(args.subject_id_column)
        # Build truth arrays in the same order as ref_ids
        true_m1 = truth_map.loc[ref_ids, "subgroup_m1"].to_numpy()
        true_m2 = truth_map.loc[ref_ids, "subgroup_m2"].to_numpy()
        true_m3 = truth_map.loc[ref_ids, "subgroup_m3"].to_numpy()
        truth_cols = [true_m1, true_m2, true_m3]

        for i, pred in enumerate(labels_list):
            true_labels = truth_cols[i]
            # Map true labels (strings) to integers deterministically
            uniq = np.unique(true_labels)
            l2i = {name: idx for idx, name in enumerate(uniq)}
            true_ints = np.array([l2i[v] for v in true_labels], dtype=int)
            ari = adjusted_rand_score(true_ints, pred)
            print(f"Adjusted Rand Index modality {i} ({args.modalities[i]}): {ari:.3f}")
    return


def do_test3(args):
    """
    Test mode 3: Test fusion matrix construction from individual labels.
    """

    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        # No CV split: use all rows for training to allow fast synthetic-data tests
        train_df = df.reset_index(drop=True)
    else:
        raise ValueError("TEST mode only supports n_folds=1 for fast testing on synthetic data")

    # Preprocess data
    ae_data, subject_id_list, dict_final = preprocessing(
        train_df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities
    )

    # Extract preprocessed & scaled per-modality dataframes
    data_list = [
        dict_final[mod].drop(columns=[args.subject_id_column]).to_numpy(dtype=np.float32, copy=True)
        for mod in args.modalities
    ]

    # Keep subject order from preprocessing (should be identical across modalities)
    # Use the IDs from the first modality as reference order
    ref_ids = dict_final[args.modalities[0]][args.subject_id_column].to_numpy()

    k_s=[3,4,2]

    n_views = len(data_list)

    clustering_algorithms = [None] * n_views

    for i in range(n_views):
        clustering_algorithms[i] = clusterer(
            'ensemble',
            n_clusters=k_s[i],
            precomputed=False,
            linkage_method='average',
            random_state=42,
            final=False
        )

    # Create the views. Initiates views as a list of view objects. Each view links one dataset and one clustering algorithm.
    views = [view(data_list[i], clustering_algorithms[i]) for i in range(n_views)]

    fusion_methods = list(args.fusion_methods) or DEFAULT_FUSION_METHODS

    for fusion_method in fusion_methods:
        print(f"Testing fusion method: {fusion_method}")

        # Create fusion algorithm
        f = fuser(fusion_method)

        # Compute fusion matrix by executing the ensemble of views directly
        fusion_matrix, individual_labels = execute_ensemble(views, f)

        print(f"Fusion matrix shape: {fusion_matrix.shape}")
        print(f"Shape of individual_labels: {len(individual_labels)} modalities, each with {len(individual_labels[0])} samples")

        # Inspect the fusion distance matrix on symmetry
        if np.allclose(fusion_matrix, fusion_matrix.T):
            print("Fusion matrix is symmetric.")
        else:
            print("Warning: Fusion matrix is not symmetric.")
        
        # Inspect the fusion matrix on zeros on diagonal
        if np.all(np.diag(fusion_matrix) == 0):
            print("Fusion matrix has zeros on its diagonal.")
        else:
            print("Warning: Fusion matrix diagonal has non-zero entries.")
        
        # Inspect values in the fusion matrix
        print(f"Fusion matrix values range from {np.min(fusion_matrix)} to {np.max(fusion_matrix)}")

        # If consensus, count singletons
        if fusion_method == 'consensus':
            # Reconstruct strict-intersection consensus labels (mirror of Consensus.execute)
            labs = [np.asarray(x) for x in individual_labels]
            n_samp = len(labs[0])
            n_cl = len(labs)
            cl_cons = np.zeros(n_samp, dtype=int)
            k = 1
            for i in range(n_samp):
                ids = np.where(labs[0] == labs[0][i])[0]
                for j in range(1, n_cl):
                    m = np.where(labs[j] == labs[j][i])[0]
                    ids = np.intersect1d(ids, m)
                if np.sum(cl_cons[ids]) == 0:
                    cl_cons[ids] = k
                    k += 1
            # Count true singletons from cl_cons
            _, counts = np.unique(cl_cons, return_counts=True)
            singleton_count = int(np.sum(counts == 1))
            print(f"Consensus produced {len(counts)} consensus clusters; true singletons = {singleton_count}.")


        # Compare clusters to true labels
        # If in TEST mode, compute ARI against ground truth labels with proper alignment
        if getattr(args, 'TEST', 'FALSE').upper() == "TRUE":
            print("TEST mode: computing Adjusted Rand Index against ground truth labels.")
            df_truth = pd.read_csv("/data/gpfs/projects/punim1993/students/Jente/multiclust/synthetic_multimodal_spartan.csv")
            # Align truth to the exact subject order used during preprocessing
            # Assumes the same subject ID column exists in df_truth
            truth_map = df_truth.set_index(args.subject_id_column)
            # Build truth arrays in the same order as ref_ids
            true_m1 = truth_map.loc[ref_ids, "subgroup_m1"].to_numpy()
            true_m2 = truth_map.loc[ref_ids, "subgroup_m2"].to_numpy()
            true_m3 = truth_map.loc[ref_ids, "subgroup_m3"].to_numpy()
            truth_cols = [true_m1, true_m2, true_m3]

            for i, pred in enumerate(individual_labels):
                true_labels = truth_cols[i]
                # Map true labels (strings) to integers deterministically
                uniq = np.unique(true_labels)
                l2i = {name: idx for idx, name in enumerate(uniq)}
                true_ints = np.array([l2i[v] for v in true_labels], dtype=int)
                ari = adjusted_rand_score(true_ints, pred)
                print(f"Adjusted Rand Index modality {i} ({args.modalities[i]}): {ari:.3f}")


    sys.exit(0)
        

def do_test4(args):
    """
    Test mode 4: Test full clustering pipeline (without VAE, without genetic algorithm etc)
    """
    df = pd.read_csv(args.input_csv)
    meta = pd.read_csv(args.meta_csv)
    if args.n_folds == 1:
        # No CV split: use all rows for training to allow fast synthetic-data tests
        train_df = df.reset_index(drop=True)
    else:
        raise ValueError("TEST mode only supports n_folds=1 for fast testing on synthetic data")

    # Preprocess data
    ae_data, subject_id_list, dict_final = preprocessing(
        train_df, meta,
        subject_id_column=args.subject_id_column,
        col_threshold=args.col_threshold,
        row_threshold=args.row_threshold,
        skew_threshold=args.skew_threshold,
        scaler_type=args.scaler_type,
        modalities=args.modalities
    )

    # Extract preprocessed & scaled per-modality dataframes
    data_list = [
        dict_final[mod].drop(columns=[args.subject_id_column]).to_numpy(dtype=np.float32, copy=True)
        for mod in args.modalities
    ]

    # Keep subject order from preprocessing (should be identical across modalities)
    # Use the IDs from the first modality as reference order
    ref_ids = dict_final[args.modalities[0]][args.subject_id_column].to_numpy()

    k_s=[3,4,2]

    n_views = len(data_list)

    clustering_algorithms = [None] * n_views

    for i in range(n_views):
        clustering_algorithms[i] = clusterer(
            'ensemble',
            n_clusters=k_s[i],
            precomputed=False,
            linkage_method='average',
            random_state=42,
            final=False
        )

    # Create the views. Initiates views as a list of view objects. Each view links one dataset and one clustering algorithm.
    views = [view(data_list[i], clustering_algorithms[i]) for i in range(n_views)]

    fusion_methods = list(args.fusion_methods) or DEFAULT_FUSION_METHODS

    for fusion_method in fusion_methods:
        print(f"Testing fusion method: {fusion_method}")

        # Create fusion algorithm
        f = fuser(fusion_method)

        # Compute fusion matrix by executing the ensemble of views directly
        fusion_matrix, individual_labels = execute_ensemble(views, f)



        # Compare clusters to true labels
        # If in TEST mode, compute ARI against ground truth labels with proper alignment
        if getattr(args, 'TEST', 'FALSE').upper() == "TRUE":
            print("TEST mode: computing Adjusted Rand Index against ground truth labels.")
            df_truth = pd.read_csv("/data/gpfs/projects/punim1993/students/Jente/multiclust/synthetic_multimodal_spartan.csv")
            # Align truth to the exact subject order used during preprocessing
            # Assumes the same subject ID column exists in df_truth
            truth_map = df_truth.set_index(args.subject_id_column)
            # Build truth arrays in the same order as ref_ids
            true_m1 = truth_map.loc[ref_ids, "subgroup_m1"].to_numpy()
            true_m2 = truth_map.loc[ref_ids, "subgroup_m2"].to_numpy()
            true_m3 = truth_map.loc[ref_ids, "subgroup_m3"].to_numpy()
            truth_cols = [true_m1, true_m2, true_m3]

            for i, pred in enumerate(individual_labels):
                true_labels = truth_cols[i]
                # Map true labels (strings) to integers deterministically
                uniq = np.unique(true_labels)
                l2i = {name: idx for idx, name in enumerate(uniq)}
                true_ints = np.array([l2i[v] for v in true_labels], dtype=int)
                ari = adjusted_rand_score(true_ints, pred)
                print(f"Adjusted Rand Index modality {i} ({args.modalities[i]}): {ari:.3f}")


        # Final clustering on the fused distance matrix
        k_final=8
        v_res = fusion_matrix 

        if not k_final:
            raise ValueError(
                "k_final must be provided for the second-step ensemble. "
                "Pass an explicit number of clusters to apply the predefined ensemble."
            )

        # Final clustering: use ensemble on the fused **distance** matrix
        c_final = clusterer(
            'ensemble',
            precomputed=True,
            n_clusters=k_final,
            linkage='average'
        )

        v_res_final = view(v_res, c_final)
        final_labels = v_res_final.execute()

        # Normalize label shape to 1D robustly
        final_labels = np.asarray(final_labels)
        if final_labels.ndim == 2:
            if final_labels.shape[1] == 1:
                final_labels = final_labels.ravel()
            else:
                # If multiple cuts/columns are returned, take the last column
                final_labels = final_labels[:, -1]
        elif final_labels.ndim != 1:
            final_labels = final_labels.reshape(-1)

        # Quality metric: Silhouette only (normalized to [0,1]).
        # For precomputed distances we use metric='precomputed'; for feature matrices we use the standard silhouette.
        def compute_quality(mat, labels, precomputed=False):
            labels = np.asarray(labels)
            if len(np.unique(labels)) <= 1:
                return 0.0
            if precomputed:
                sil = silhouette_score(mat, labels, metric='precomputed')  # raw in [-1, 1]
            else:
                sil = silhouette_score(mat, labels)  # raw in [-1, 1]
            sil_n = (sil + 1.0) / 2.0  # normalize to [0,1]
            return float(sil_n)

        # View-level quality averaged across all views
        view_scores_per_view = [
            compute_quality(data_list[v], individual_labels[v], precomputed=False)
            for v in range(n_views)
        ]
        view_score = float(np.mean(view_scores_per_view))
        # Final consensus quality on the fused (precomputed distance) matrix
        final_score = compute_quality(v_res, final_labels, precomputed=True)

        print(f"Final clustering produced {len(np.unique(final_labels))} clusters with sizes {np.bincount(final_labels)}")
        print(f"View-level quality scores for fusion method {fusion_method}: {view_scores_per_view}")
        print(f"Mean view-level quality: {view_score:.4f}, final clustering quality: {final_score:.4f} for fusion method: {fusion_method}")
    
    sys.exit(0)



# --- Command-line entry point ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Original args
    parser.add_argument('--input_csv', default='cleaned_discovery_data.csv')
    parser.add_argument('--meta_csv', default='merged_meta.csv')
    parser.add_argument('--base_dir', default='/data/gpfs/projects/punim1993/students/Jente/multiclust')
    parser.add_argument('--subject_id_column', default='src_subject_id')
    parser.add_argument('--col_threshold', type=float, default=0.5)
    parser.add_argument('--row_threshold', type=float, default=0.5)
    parser.add_argument('--skew_threshold', type=float, default=0.75)
    parser.add_argument('--scaler_type', default='robust')
    parser.add_argument('--modalities', nargs='+', default=['Internalising', 'Functioning', 'Cognition', 'Detachment', 'Psychoticism'])
    parser.add_argument('--dim_reduction', choices=[None, "None", 'VAE', 'AE', 'PCA'], default='VAE', help='Dimensionality reduction method to use (VAE, AE, or None)')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128,256,512])
    parser.add_argument('--activation_functions', nargs='+', default=['ReLU','LeakyReLU','selu','swish'])
    parser.add_argument('--learning_rates', nargs='+', type=float, default=[0.001,0.0001])
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[32,64,128])
    parser.add_argument('--latent_dims', nargs='+', type=int, default=[2,5,10])
    parser.add_argument('--k_min', type=int, default=2)
    parser.add_argument('--k_max', type=int, default=10)
    parser.add_argument('--linkages', type=str, nargs='+', default=['complete','average','weighted'])
    parser.add_argument('--n_population', type=int, default=100)
    parser.add_argument('--n_generations', type=int, default=10)
    parser.add_argument('--optimisation', choices=['single','multi'], default='multi')
    parser.add_argument('--ga_objectives', nargs='+', default=None,
                        help='Objectives optimised by GA (tokens such as mean_view_stability, mean_view_quality, '
                             'final_stability, final_quality, min_view_stability, min_view_quality).')
    parser.add_argument('--fusion_methods', nargs='+', default=DEFAULT_FUSION_METHODS,
                        help='Fusion methods available to the GA (e.g., agreement consensus disagreement).')
    parser.add_argument('--n_bootstrap', type=int, default=100)
    parser.add_argument('--bootstrap_mode', choices=['bootstrap','subsample'], default='subsample')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--output_pkl', default='pipeline_results.pkl')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel workers for bootstrap clustering')
    parser.add_argument('--TEST', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='Skip VAE and use preprocessed features as latent embeddings when TRUE')
    parser.add_argument('--max_missing_bootstraps', type=int, default=5,
                        help='Maximum number of missing bootstrap label files allowed before gather aborts')
    parser.add_argument('--mincluster', default="TRUE", help='Enforce minimum cluster size of 10 in final clustering (True/False)', choices=['TRUE','FALSE'])
    parser.add_argument('--mincluster_n', type=int, default=10, help='Minimum cluster size to enforce in final clustering')
    # New mode args
    parser.add_argument('--mode', choices=['bootstrap','gather','outer','init', 'merge', 'test1', 'test2', 'test3', 'test4', 'test5'], default='init')
    parser.add_argument('--generation', type=int, help='GA generation index')
    parser.add_argument('--population_file', type=str)
    parser.add_argument('--seed',            type=int, default=None,
                    help='Random seed for GA init (only used with --mode init)')
    parser.add_argument('--population_dir', type=str, help='Directory where population files are stored')
    parser.add_argument('--population_initial_file', type=str, help='File to load initial population from in bootstrap mode')
    parser.add_argument('--bootstrap_index', type=int)
    parser.add_argument('--bootstrap_dir', type=str)
    parser.add_argument('--output_labels', type=str, help='Where to save bootstrap labels for stability computation')
    parser.add_argument('--output_population', type=str)
    parser.add_argument('--fold_index', type=int)
    parser.add_argument('--output_metrics', type=str)
    parser.add_argument('--output_final_metrics', type=str)
    parser.add_argument('--TEST-phase', choices=[0,1,2,3,4], type=int, default=0,
                        help='For TEST mode: which phase to run (0=Full pipeline, 1=Only Kmeans, 2=Only Spectral, 3=Test fusion matrix, 4=Test final clustering, 5=Test only individual labels but full ensemble.)')
    parser.add_argument('--ga_cxpb', type=float, default=0.7,
                        help='Crossover probability used during GA gather stage.')
    parser.add_argument('--ga_mutpb', type=float, default=0.2,
                        help='Mutation probability used during GA gather stage.')
    parser.add_argument('--ga_elitism', type=int, default=2,
                        help='Number of elite individuals to carry over each generation.')
    parser.add_argument('--DO_SVM', choices=['TRUE', 'FALSE'], default='FALSE',
                        help='In OUTER mode, whether to run SVM classification on the final clustering labels (TRUE/FALSE).')
    args = parser.parse_args()

    # Map activation strings to actual nn modules
    act_map = {
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(),
        "selu": nn.SELU(),
        "swish": nn.SiLU()
    }
    activation_functions = {
        name: act_map[name]
        for name in args.activation_functions
        if name in act_map
    }

    args.ga_objectives = _normalize_objective_tokens(getattr(args, "ga_objectives", []), args.optimisation)
    args.fusion_methods = _normalize_method_list(getattr(args, "fusion_methods", DEFAULT_FUSION_METHODS))
    if not args.fusion_methods:
        args.fusion_methods = DEFAULT_FUSION_METHODS.copy()

    # --- Clean out any old classes so we can rebuild them fresh ---
    for attr in list(vars(creator).keys()):
        if attr.startswith("FitnessMulti"):
            delattr(creator, attr)
    for cls in ("FitnessMax", "Individual"):
        if hasattr(creator, cls):
            delattr(creator, cls)

    # Ensure DEAP classes exist before loading pickled populations
    _ensure_multi_fitness_class(args)

    if args.mode == 'init':
        do_init(args)
    elif args.mode == 'bootstrap':
        do_bootstrap(args)
    elif args.mode == 'gather':
        do_gather(args)
    elif args.mode == 'outer':
        do_outer(args)
    elif args.mode == 'merge':
        do_merge(args)
    elif args.mode == 'test1':
        do_test1(args)
    elif args.mode == 'test2':
        do_test2(args)
    elif args.mode == 'test3':
        do_test3(args)
    elif args.mode == 'test4':
        do_test4(args)
    elif args.mode == 'test5':
        do_test5(args)
    else:
        parser.error(f"Unknown mode: {args.mode}")

   

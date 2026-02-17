def enforce_min_cluster_size(distance_matrix, final_labels, min_size=20):
    """
    Reassign points from clusters smaller than `min_size` to the closest
    large clusters, where closeness is defined as the mean pairwise
    distance to all members of a candidate large cluster.

    Parameters
    ----------
    distance_matrix : (n_samples, n_samples) array-like
        Pairwise distance matrix for the samples in this view.
    final_labels : (n_samples,) array-like
        Cluster labels for this view.
    min_size : int
        Minimum allowed cluster size.

    Returns
    -------
    labels : (n_samples,) ndarray
        Labels after enforcing the minimum cluster size.
    """
    labels = np.asarray(final_labels).copy()
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    # Identify big and small clusters
    big_clusters = [cl for cl, sz in cluster_sizes.items() if sz >= min_size]
    small_clusters = [cl for cl, sz in cluster_sizes.items() if sz < min_size]

    # If everything is small, we can't do anything sensible here
    if len(big_clusters) == 0:
        return labels

    # Boolean masks for big clusters (fixed, based on original labels)
    masks_big = {cl: (labels == cl) for cl in big_clusters}

    n = len(labels)

    # Precompute, for every point, the mean distance to each big cluster
    # mean_dists shape: (n_samples, n_big_clusters)
    mean_dists = np.stack(
        [
            distance_matrix[:, mask].mean(axis=1) if mask.sum() > 0 else np.full(n, np.inf)
            for cl, mask in masks_big.items()
        ],
        axis=1,
    )
    big_clusters_arr = np.array(big_clusters)

    # Reassign all points from small clusters
    for cl in small_clusters:
        ids = np.where(labels == cl)[0]
        if len(ids) == 0:
            continue
        # For each point in this small cluster, pick closest big cluster
        best_cols = np.argmin(mean_dists[ids], axis=1)
        labels[ids] = big_clusters_arr[best_cols]

    return labels
## Parea classes


import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN, OPTICS
from typing import List, Union, Any
from scipy.cluster import hierarchy
from scipy import spatial
from sklearn.cluster import KMeans, MeanShift, Birch
from sklearn.mixture import GaussianMixture
import inspect
from scipy.spatial.distance import squareform

class Clusterer(object):
    """
    :class:`Clusterer` is the Abstract Base Class for all clustering algorithms.
    All clustering algorithms must be a subclass of this class in order to
    accepted by functions such as :func:`~pyrea.core.execute_ensemble()`.
    To extend Pyrea with a custom clustering algorithm, create a new
    class that is a subclass of :class:`Clusterer`, and implement the
    :func:`Clusterer.execute` function.
    """
    def __init__(self) -> None:
        pass

    def execute(self) -> list:
        """
        Execute the clustering algorithm with the given :attr:`data`.
        """
        pass


class NullClusteringPyrea(Clusterer):
    """
    Dummy clusterer that assigns all samples to a single cluster (label 0).
    Useful when a view is deemed uninformative / unimodal and we want to
    include it without forcing spurious splits.
    """
    def __init__(self, n_clusters: int = 1, **kwargs) -> None:
        super().__init__()
        self.n_clusters = 1  # always 1 for null model

    def execute(self, data) -> list:
        import numpy as _np
        n = data.shape[0] if hasattr(data, 'shape') else len(data)
        return _np.zeros(n, dtype=int)


class HierarchicalClusteringPyrea(Clusterer):
    def __init__(self, precomputed,
                       method='average',
                       linkage_method=None,
                       metric='euclidean',
                       distance_metric=None,
                       optimal_ordering=False,
                       out=None,
                       n_clusters=None,
                       height=None) -> None:
        """
        Hierarchical clustering that supports both raw-feature inputs and
        precomputed distance matrices.  
        - Use Ward **only** with raw observations (not precomputed distances).
        - For precomputed distances, use linkages in {'single','complete','average','weighted','centroid','median'}.

        Parameters
        ----------
        precomputed : bool
            If True, `data` passed to execute() is a distance matrix (condensed or square).
        method / linkage_method : str
            Linkage method. `method` is accepted for compatibility; if both are provided,
            `linkage_method` takes precedence. Supports 'single', 'complete', 'average',
            'weighted', 'centroid', 'median', 'ward', 'ward2' (alias of 'ward').
        metric / distance_metric : str
            Distance metric used when computing pdist on raw observations. If both are
            given, `distance_metric` takes precedence.
        """
        super().__init__()
        # Accept either name; prefer linkage_method when provided
        self.linkage_method = linkage_method or method or 'average'

        # Keep both names for backward compatibility; prefer distance_metric
        self.distance_metric = distance_metric or metric or 'euclidean'
        self.metric = metric  # retained for compatibility with external callers

        self.optimal_ordering = optimal_ordering
        self.precomputed = precomputed
        self.out = out
        self.n_clusters = n_clusters
        self.height = height

        # Normalize 'ward2' to 'ward' (SciPy implements Ward.D2 internally for observation input)
        if self.linkage_method == 'ward2':
            self.linkage_method = 'ward'

        # Early validation: forbid ward with precomputed distances
        if self.precomputed and self.linkage_method == 'ward':
            self.linkage_method = 'average'
            print("Warning: 'ward' linkage is not valid for precomputed distances. Using 'average' instead.")

    def execute(self, data) -> list:
        super().execute()

        # --- Build linkage matrix correctly depending on input type ---
        if self.precomputed:
            # Accept square (n x n) or condensed (n*(n-1)/2,) distance inputs
            if isinstance(data, np.ndarray) and data.ndim == 2:
                # Sanitize data: replace NaN and inf, fill diagonal with 0.0
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                np.fill_diagonal(data, 0.0)
                y = squareform(data, checks=True)  # condensed
            else:
                y = data  # assume already condensed

            # Ward is invalid for precomputed distances (already validated in __init__)
            Z = hierarchy.linkage(y, method=self.linkage_method,
                                  optimal_ordering=self.optimal_ordering)
        else:
            X = data  # raw observations
            # Validate Ward input for degenerate or NaN features
            if self.linkage_method == 'ward' and not self.precomputed:
                X = np.asarray(data)
                if np.allclose(np.var(X, axis=0), 0) or np.isnan(X).any():
                    raise ValueError("Invalid input for Ward linkage: degenerate or NaN features.")
            if self.linkage_method == 'ward':
                # For Ward, pass the observation matrix directly
                Z = hierarchy.linkage(X, method='ward',
                                      optimal_ordering=self.optimal_ordering)
            else:
                # For other linkages, compute condensed distances with pdist
                y = spatial.distance.pdist(X, metric=self.distance_metric, out=self.out)
                Z = hierarchy.linkage(y, method=self.linkage_method,
                                      optimal_ordering=self.optimal_ordering)

        # --- Cut the tree to obtain labels ---
        labels_2d = hierarchy.cut_tree(Z, n_clusters=self.n_clusters, height=self.height)

        # Ensure 1D output and sanitize
        if labels_2d.shape[1] > 1:
            labels_1d = labels_2d[:, -1]
        else:
            labels_1d = labels_2d.ravel()
        labels_1d = np.nan_to_num(labels_1d).astype(int)

        return labels_1d

class SpectralClusteringPyrea(Clusterer):
    def __init__(self, n_clusters=8,
                       eigen_solver=None,
                       n_components=None,
                       random_state=None,
                       n_init=10,
                       gamma=1.0,
                       affinity='nearest_neighbors',
                       n_neighbors=10,
                       eigen_tol=0.0,
                       assign_labels='kmeans',
                       degree=3,
                       coef0=1,
                       kernel_params=None,
                       n_jobs=None,
                       verbose=False,
                       method=None) -> None:  # method is not used, but is here
                                              # for compatibility with other
                                              # clustering algorithms
        """
        Perform spectral clustering.

        See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_components
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    def execute(self, data: list) -> list:
        super().execute()
        X = np.asarray(data)
        n = X.shape[0]

        # If n < n_clusters, return unique labels (each sample its own cluster)
        if n < self.n_clusters:
            return np.arange(n)

        # Adaptive neighbors: max(5, sqrt(n)), but never >= n
        if self.affinity == 'nearest_neighbors':
            nn = self.n_neighbors if self.n_neighbors is not None else int(np.sqrt(max(n, 1)))
            nn = max(5, nn)
            nn = min(nn, max(1, n - 1))
        else:
            nn = self.n_neighbors

        # Build model with (possibly) adapted n_neighbors
        model = SpectralClustering(
            n_clusters=self.n_clusters,
            eigen_solver=self.eigen_solver,
            n_components=self.n_components,
            random_state=self.random_state,
            n_init=self.n_init,
            gamma=self.gamma,
            affinity=self.affinity,
            n_neighbors=nn,
            eigen_tol=self.eigen_tol,
            assign_labels=self.assign_labels,
            degree=self.degree,
            coef0=self.coef0,
            kernel_params=self.kernel_params,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        # If using kNN affinity, check connectivity and fall back to RBF if disconnected
        if self.affinity == 'nearest_neighbors' and n > 1:
            try:
                from sklearn.neighbors import kneighbors_graph
                from scipy.sparse.csgraph import connected_components
                G = kneighbors_graph(X, nn, include_self=False)
                n_comp, _ = connected_components(G)
                if n_comp > 1:
                    # Fall back to RBF with median heuristic for gamma
                    dists = spatial.distance.pdist(X, metric='euclidean')
                    med = np.median(dists) if dists.size > 0 else 1.0
                    gamma = 1.0 / (2.0 * (med ** 2) + 1e-12)
                    model = SpectralClustering(
                        n_clusters=self.n_clusters,
                        random_state=self.random_state,
                        n_init=self.n_init,
                        affinity='rbf',
                        gamma=gamma,
                        assign_labels=self.assign_labels,
                        eigen_tol=self.eigen_tol,
                        verbose=self.verbose,
                    )
            except Exception:
                # If any check fails, just proceed with the configured model
                pass

        return model.fit(X).labels_






class ModelBasedClusteringPyrea(Clusterer):
    def __init__(self, n_components=3, covariance_type='full', random_state=None,
                       n_clusters=None,
                       metric = None,
                       method = None,
                       optimal_ordering=None,
                       distance_metric = None,
                       precomputed = None,
                       out = None,
                       height = None, **kwargs):


        """
        Model-Based Clustering using Gaussian Mixture Models (GMM).
        """
        super().__init__()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state

        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.optimal_ordering = optimal_ordering
        self.distance_metric = distance_metric
        self.precomputed = precomputed
        self.out = out
        self.height = height
        
        # Dynamically filter out invalid kwargs
        valid_params = inspect.signature(GaussianMixture.__init__).parameters
        self.kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    def execute(self, data) -> list:
        super().execute()
        model = GaussianMixture(n_components=self.n_components, 
                                covariance_type=self.covariance_type, 
                                random_state=self.random_state, 
                                **self.kwargs)  # Now only valid arguments are passed
        return model.fit_predict(data)


class KMeansPyrea(Clusterer):
    def __init__(self, n_clusters=3, init='k-means++', max_iter=300, random_state=None, **kwargs):
        """
        K-Means Clustering.
        
        Parameters:
            - n_clusters: Number of clusters (default=3).
            - init: Initialization method ('k-means++' or 'random').
            - max_iter: Maximum number of iterations (default=300).
            - random_state: Random seed for reproducibility.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs

        # Dynamically filter out invalid kwargs
        valid_params = inspect.signature(KMeans.__init__).parameters
        self.kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    def execute(self, data) -> list:
        super().execute()
        model = KMeans(n_clusters=self.n_clusters, 
                       init=self.init, 
                       max_iter=self.max_iter, 
                       random_state=self.random_state, 
                       **self.kwargs)
        return model.fit_predict(data)


class MeanShiftPyrea(Clusterer):
    def __init__(self, bandwidth=None, cluster_all=True, **kwargs):
        """
        Mean-Shift Clustering.
        
        Parameters:
            - bandwidth: Window size for clustering (default=None, automatically estimated).
            - cluster_all: Whether to assign all points to clusters.
        """
        super().__init__()
        self.bandwidth = bandwidth
        self.cluster_all = cluster_all
        self.kwargs = kwargs

        # Dynamically filter out invalid kwargs
        valid_params = inspect.signature(MeanShift.__init__).parameters
        self.kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    def execute(self, data) -> list:
        super().execute()
        model = MeanShift(bandwidth=self.bandwidth, 
                          cluster_all=self.cluster_all, 
                          **self.kwargs)
        return model.fit_predict(data)




class EnsembleClusteringPyrea(Clusterer):
    def __init__(self,
                 clusterers: List[Clusterer] = None,
                 n_clusters=3,
                 precomputed=False,
                 linkage_method: str = 'average',
                 method=None,  # <-- For compatibility; not actually used
                 **kwargs):
        """
        Ensemble Clustering using a co-association matrix.
        
        Parameters:
            - clusterers: A list of Clusterer objects (e.g., HierarchicalClusteringPyrea, 
                          SpectralClusteringPyrea, ModelBasedClusteringPyrea, KMeansPyrea , etc.).
            - n_clusters: Desired number of final clusters.
            - linkage_method: Linkage method for the final hierarchical clustering 
                              (e.g., 'single', 'complete', 'average', 'ward', etc.).
            - method: An unused parameter for compatibility (like other classes).
            - **kwargs: Catch-all for any other arguments that might be passed.
        """
        super().__init__()
        
        self.clusterers = clusterers
        self.n_clusters = n_clusters
        self.precomputed = precomputed
        self.linkage_method = linkage_method
        self.method = method  # Not used, but kept for compatibility
        self.kwargs = kwargs  # In case something else is passed

        # If no specific clusterers are passed, create a default set (optional).
        if self.clusterers is None:
            self.clusterers = [
                HierarchicalClusteringPyrea(precomputed=False, linkage_method=linkage_method, n_clusters=self.n_clusters),
                SpectralClusteringPyrea(n_clusters=self.n_clusters),
                ModelBasedClusteringPyrea(n_components=self.n_clusters),
                KMeansPyrea(n_clusters=self.n_clusters),
                MeanShiftPyrea()
            ]

    def execute(self, data) -> list:
        super().execute()
        import warnings
        if self.precomputed == False:
            # 1. Run each sub-clusterer
            num_samples = len(data)
            num_clusterers = len(self.clusterers)
            all_labels = np.zeros((num_clusterers, num_samples), dtype=int)
            
            for i, clstr in enumerate(self.clusterers):
                labels = clstr.execute(data)
                all_labels[i, :] = labels

            # 2. Build co-association matrix
            co_assoc = np.zeros((num_samples, num_samples), dtype=float)
            for i in range(num_clusterers):
                for s in range(num_samples):
                    for o in range(s + 1, num_samples):
                        if all_labels[i, s] == all_labels[i, o]:
                            co_assoc[s, o] += 1
                            co_assoc[o, s] += 1
            
            co_assoc /= float(num_clusterers)  # fraction in same cluster
            # Sanitize co_assoc
            if np.any(np.isnan(co_assoc)):
                co_assoc = np.nan_to_num(co_assoc, nan=0.0)
            
            # 3. Use (1 - co_assoc) as distance, do hierarchical clustering
            dist_matrix = 1.0 - co_assoc
            dist_matrix = np.nan_to_num(dist_matrix, nan=0.0, posinf=1.0, neginf=0.0)
            np.fill_diagonal(dist_matrix, 0.0)
            dist_matrix = np.clip(dist_matrix, 0.0, 1.0)
            condensed = squareform(dist_matrix, checks=True)
            method = self.linkage_method
            Z = hierarchy.linkage(condensed, method=method)
            
            # 4. Cut the dendrogram to produce final labels
            try:
                final_labels = hierarchy.cut_tree(Z, n_clusters=self.n_clusters).reshape(-1)
            except ValueError:
                # Sometimes Z is invalid for cut_tree; fall back to flat clustering
                final_labels = hierarchy.fcluster(Z, t=self.n_clusters, criterion='maxclust')
            
            return final_labels

        elif self.precomputed == True:
            # === Final-stage ensemble over a fused matrix ===
            # Expect an NxN pairwise matrix (distance or similarity)
            import warnings
            X = np.asarray(data)
            if X.ndim != 2 or X.shape[0] != X.shape[1]:
                raise ValueError("Ensemble (precomputed=True) expects a square NxN matrix.")

            # Detect type and construct both D (distance) and A (similarity)
            is_sim = np.allclose(np.diag(X), 1.0, atol=1e-3) and X.min() >= -1e-6 and X.max() <= 1.0 + 1e-6
            if is_sim:
                A = 0.5 * (X + X.T)
                A = np.clip(A, 0.0, 1.0)
                np.fill_diagonal(A, 1.0)
                D = 1.0 - A
                np.fill_diagonal(D, 0.0)
            else:
                D = 0.5 * (X + X.T)
                np.fill_diagonal(D, 0.0)
                D[D < 0.0] = 0.0
                A = 1.0 - D
                A[A < 0.0] = 0.0
                np.fill_diagonal(A, 1.0)

            n = D.shape[0]

            # ---- Build a set of compatible final-stage clusterers ----
            final_clusterers = [HierarchicalClusteringPyrea(precomputed=True, linkage_method=self.linkage_method),
                                SpectralClusteringPyrea(n_clusters=self.n_clusters, affinity='precomputed')
                                ]

            if len(final_clusterers) == 0:
                raise ValueError("No compatible final-stage clusterers configured for precomputed ensemble.")

            # ---- Run each compatible clusterer and collect labels ----
            labels_list = []
            for clf in final_clusterers:
                try:
                    if isinstance(clf, HierarchicalClusteringPyrea):
                        lab = np.asarray(clf.execute(D)).ravel()
                    elif isinstance(clf, SpectralClusteringPyrea):
                        lab = np.asarray(clf.execute(A)).ravel()
                    else:
                        # Unknown type; skip defensively
                        continue
                    if lab.shape[0] == n:
                        labels_list.append(lab.astype(int))
                except Exception:
                    # Ignore failing clusterers in the ensemble; continue with others
                    continue

            if len(labels_list) == 0:
                warnings.warn("All final-stage clusterers failed; returning single cluster.")
                return np.zeros(X.shape[0], dtype=int)

            # ---- Co-association over final-stage labelings ----
            num_models = len(labels_list)
            co_assoc = np.zeros((n, n), dtype=float)
            for lab in labels_list:
                co_assoc += (lab[:, None] == lab[None, :]).astype(float)
            co_assoc /= float(num_models)
            co_assoc = 0.5 * (co_assoc + co_assoc.T)
            np.fill_diagonal(co_assoc, 1.0)
            # Sanitize co_assoc
            if np.any(np.isnan(co_assoc)):
                co_assoc = np.nan_to_num(co_assoc, nan=0.0)

            # Final distance and hierarchical cut to get the single output labeling
            dist_matrix = 1.0 - co_assoc
            dist_matrix = np.nan_to_num(dist_matrix, nan=0.0, posinf=1.0, neginf=0.0)
            np.fill_diagonal(dist_matrix, 0.0)
            dist_matrix = np.clip(dist_matrix, 0.0, 1.0)
            method = self.linkage_method or 'average'
            if method == 'ward':
                method = 'average'
                print("Warning: 'ward' linkage is invalid for precomputed distances; using 'average'.")
            Z = hierarchy.linkage(squareform(dist_matrix, checks=True), method=method)
            try:
                final_labels = hierarchy.cut_tree(Z, n_clusters=self.n_clusters).reshape(-1)
                return final_labels
            except ValueError:
                return hierarchy.fcluster(Z, t=self.n_clusters, criterion='maxclust')








class Fusion(object):
    def __init__(self) -> None:
        """
        :class:`Fusion` is the Abstract Base Class for all fusion algorithms.
        All fusion algorithms must be a subclass of this class in order to
        accepted by functions such as :func:`~pyrea.core.execute_ensemble()`.
        To extend Pyrea with a custom fusion algorithm, create a new
        class that is a subclass of :class:`Fusion`, and implement the
        :func:`Fusion.execute` function.
        """
        pass

    def execute(self, views: list) -> list:
        """
        Execute the fusion algorithm on the provided :attr:`views`.
        """
        # TODO: Fix views type to List[View] (requires reshuffle of class order)
        pass


class Disagreement(Fusion):
    """
    Disagreement fusion function.

    Creates the disagreement of two clusterings.
    
    # The input is a list of clusterings. Each view is presumably an array of cluster labels for n data points for each view.
    # Finds how many data points are in the first set of labels.
    # Initializes an nxn matrix of zeros. This will accumulate the disagreements.
    # Loops over each cluster:
        # Counts the views
        # Compute res: For each pair of data points (x,y), it checks if the labels are different. If yes, int (x !=y) is 1, otherwise 0. 
        # This gives an nxn matrix with res[i,j]=1 if the i-th point and j-th point have different labels in that view, or 0 if they share the same label.
        # Accumulate: labels=labels+res adds the pairwise disagreements from this view to the running total. 
    # It returns the final nxn matrix, where labels[i,j] indicates how many of the clusterings (out of all views) classified data point i and point j differently.
    """
    def __init__(self) -> None:
        super().__init__()

    def execute(self, views: list) -> list:
        """
        Executes the disagreement fusion algorithm on the provided clusterings,
        :attr:`views`.
        """
        n = len(views[0])
        labels = np.zeros((n, n), dtype=int)

        for i in range(0, len(views)):
            l = np.asarray(views[i])
            res = (l[:, None] != l[None, :]).astype(int)
            labels += res

        return labels


class Agreement(Fusion):
    """
    Agreement fusion function.

    Creates the agreement of multiple clusterings, then converts to a distance matrix.
    """
    def __init__(self) -> None:
        super().__init__()

    def execute(self, views: list) -> np.ndarray:
        """
        Executes the agreement fusion algorithm on the provided clusterings,
        then inverts the count into a distance.
        """
        n_samp = len(views[0])
        n_views = len(views)
        # Count how many views agree for each pair
        labels = np.zeros((n_samp, n_samp), dtype=float)
        for l in views:
            l = np.asarray(l)
            res = (l[:, None] == l[None, :]).astype(float)
            labels += res

        # Zero out self-matches and enforce symmetry
        np.fill_diagonal(labels, 0.0)
        labels = 0.5 * (labels + labels.T)

        # Convert agreement counts (0…V) into normalized distances [0,1]
        distances = (n_views - labels) / float(n_views)
        np.fill_diagonal(distances, 0.0)
        distances = np.clip(distances, 0.0, 1.0)
        return distances

class Consensus(Fusion):
    """
    Consensus fusion function.

    Builds a strict-intersection consensus clustering, then converts to distances.
    """
    def __init__(self) -> None:
        super().__init__()

    def execute(self, views: list) -> np.ndarray:
        # Ensure we have NumPy arrays (critical for elementwise ==)
        views = [np.asarray(v) for v in views]

        n_samp  = len(views[0])
        n_cl    = len(views)
        cl_cons = np.zeros(n_samp, dtype=int)
        k = 1
        for i in range(n_samp):
            ids = np.where(views[0] == views[0][i])[0]
            for j in range(1, n_cl):
                m = np.where(views[j] == views[j][i])[0]
                ids = np.intersect1d(ids, m)
            if np.sum(cl_cons[ids]) == 0:
                cl_cons[ids] = k
                k += 1

        # (rest of your code unchanged)
        mat_bin = np.zeros((n_samp, n_samp), dtype=int)
        for i in range(n_samp):
            ids = np.where(cl_cons == cl_cons[i])
            mat_bin[i, ids] = 1
            mat_bin[ids, i] = 1

        distances = 1 - mat_bin.astype(float)
        np.fill_diagonal(distances, 0)
        return distances
    


class View(object):
    """
    Represents a :class:`View`, which consists of some :attr:`data` and a
    clustering algorithm, :attr:`clusterer`.

    Requires a data source, :attr:`data`, which is used to create the
    view (the data source can be a Python matrix (a list of lists), a
    NumPy 2D array, or a Pandas DataFrame) and a clustering method
    :attr:`clusterer`.

    Some examples follow (using a list of lists)::

        import pyrea

        data = [[1, 5, 3, 7],
                [4, 2, 9, 4],
                [8, 6, 1, 9],
                [7, 1, 8, 1]]

        v = pyrea.view(data, pyrea.cluster('ward'))

    Or by passing a Pandas DataFrame (``pandas.core.frame.DataFrame``)::

        import pyrea
        import pandas

        data = pandas.read_csv('iris.csv')

        v = pyrea.view(data, pyrea.cluster('ward'))

    Or (passing a numpy 2d array or matrix (``numpy.matrix`` or ``numpy.ndarray``))::

        import pyrea
        import numpy

        data = numpy.random.randint(0, 10, (4,4))

        v = pyrea.view(data, pyrea.cluster('ward'))


    .. seealso:: The :class:`Clusterer` class.

    :param data: The data from which to create your :class:`View`.
    :param clusterer: The clustering algorithm to use to cluster your
     :attr:`data`
    :ivar labels: Contains the calculated labels when the :attr:`clusterer`
     is run on the :attr:`data`.
    """
    def __init__(self, data, clusterer: List[Clusterer]) -> None:

        self.data = np.asarray(data)
        self.clusterer = clusterer
        self.labels = None

        if data.ndim != 2:
            raise Exception("Number of dimensions is not 2: you supplied a data structure with %s dimensions." % data.ndim)

    def execute(self) -> list:
        """
        Clusters the :attr:`data` using the :attr:`clusterer` specified at
        initialisation.
        """
        # TODO: If a list is passed, then we need to execute them all.
        import warnings
        try:
            self.labels = self.clusterer.execute(self.data)
        except Exception as e:
            warnings.warn(f"Clusterer failed: {e}")
            self.labels = np.zeros(self.data.shape[0], dtype=int)

        return self.labels


class Ward(Clusterer):
    """
    Implements the 'Ward' clustering algorithm.
    """
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Deprecated.")

    def execute(self, data):
        """
        Perform the clustering and return the results.
        """
        return AgglomerativeClustering().fit(data).labels_


class Complete(Clusterer):
    """
    Implements the 'complete' clustering algorithm.
    """
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Deprecated.")

    def execute(self, data):
        """
        Perform the clustering and return the results.
        """
        return AgglomerativeClustering(linkage='complete').fit(data).labels_


class Average(Clusterer):
    """
    Implements the 'average' clustering algorithm.
    """
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Deprecated.")

    def execute(self, data):
        """
        Perform the clustering and return the results.
        """
        return AgglomerativeClustering(linkage='average').fit(data).labels_


class Single(Clusterer):
    """
    Implements the 'single' clustering algorithm.
    """
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Deprecated.")

    def execute(self, data):
        """
        Perform the clustering and return the results.
        """
        return AgglomerativeClustering(linkage='single').fit(data).labels_


from typing import List

class Ensemble(object):
    """
    The Ensemble class encapsulates the views, fusion algorithm
    and clustering methods required to perform a multi-view clustering.

    :param views: The views that constitute the ensemble's multi-view data.
    :param fuser: The fusion algorithm to use.
    :param mincluster: Whether to enforce a minimum cluster size per view.
    :param mincluster_n: Minimum allowed cluster size.
    """
    def __init__(self,
                 views: List["View"],
                 fuser: "Fusion",
                 mincluster: bool = False,
                 mincluster_n: int = 10) -> None:

        if isinstance(views, View):
            self.views = [views]
        elif isinstance(views, list):
            self.views = views
        else:
            raise TypeError("views must be a View or list of Views")

        self.fuser = fuser
        self.labels: List[np.ndarray] = []
        self.mincluster = mincluster
        self.mincluster_n = mincluster_n

    def execute(self):
        """
        Executes the ensemble, returning a fused matrix and the
        (optionally min-size-adjusted) individual labels for each view.
        """

        # 1. Execute each view's clustering algorithm on its data
        self.labels = [np.asarray(v.execute()) for v in self.views]

        # Make a working copy we can modify per view
        individual_labels = [lbl.copy() for lbl in self.labels]

        # 2. Optionally enforce minimum cluster size per view
        if self.mincluster == "TRUE" or self.mincluster is True:
            from sklearn.metrics import pairwise_distances
            for idx, v in enumerate(self.views):
                X = np.asarray(v.data)
                labels = np.asarray(individual_labels[idx])

                # Skip degenerate or single-cluster views
                if X.shape[0] == 0 or len(np.unique(labels)) <= 1:
                    continue

                # Build a pairwise distance matrix for this view
                D = pairwise_distances(X)

                # Enforce minimum cluster size using the distance-based helper
                new_labels = enforce_min_cluster_size(D, labels, min_size=self.mincluster_n)
                individual_labels[idx] = new_labels

        # Keep self.labels in sync with any modifications
        self.labels = individual_labels

        # 3. Fuse the clusterings to a single fused matrix
        fusion_matrix = self.fuser.execute(self.labels)

        return fusion_matrix, individual_labels
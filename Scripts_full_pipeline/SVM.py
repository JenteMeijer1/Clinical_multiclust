# SVM classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.inspection import permutation_importance


# --- SVM Uncertainty and Feature Importance Utilities ---
def svm_predict_with_uncertainty(model, X):
    """Return predictions along with simple uncertainty diagnostics.

    Outputs
    -------
    y_pred : np.ndarray
    proba  : np.ndarray or None
        Class probabilities if available.
    confidence : np.ndarray
        Max class probability per sample (NaN if proba unavailable).
    entropy : np.ndarray
        Predictive entropy in nats (NaN if proba unavailable).
    margin : np.ndarray
        |decision_function| for binary classification (NaN otherwise/unavailable).
    """
    y_pred = model.predict(X)

    proba = None
    confidence = np.full(len(y_pred), np.nan, dtype=float)
    entropy = np.full(len(y_pred), np.nan, dtype=float)

    # Probability-based uncertainty (works for binary + multiclass when probability=True)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            # confidence: max probability per row
            confidence = np.max(proba, axis=1)
            # entropy: -sum(p log p)
            p = np.clip(proba, 1e-12, 1.0)
            entropy = -np.sum(p * np.log(p), axis=1)
        except Exception:
            proba = None

    # Margin-based uncertainty (binary only)
    margin = np.full(len(y_pred), np.nan, dtype=float)
    if hasattr(model, "decision_function"):
        try:
            df = model.decision_function(X)
            df = np.asarray(df)
            if df.ndim == 1:
                margin = np.abs(df)
        except Exception:
            pass

    return y_pred, proba, confidence, entropy, margin


def _get_feature_names(X):
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    return [f"x{i}" for i in range(X.shape[1])]


def svm_feature_contributions(model, X, y, scoring="balanced_accuracy", n_repeats=10, random_state=42):
    """Compute per-feature contributions.

    - For linear SVM: uses |coef_| (model weights).
    - For non-linear kernels: uses permutation importance on (X,y).

    Returns
    -------
    importance : pd.Series
        Indexed by feature name, sorted descending.
    meta : dict
        Describes method and any kernel details.
    """
    feature_names = _get_feature_names(X)

    # Linear kernel: coefficient magnitudes
    if getattr(model, "kernel", None) == "linear" and hasattr(model, "coef_"):
        coefs = np.asarray(model.coef_)
        # binary: (1, n_features), multiclass: (n_classes, n_features)
        w = np.mean(np.abs(coefs), axis=0)
        s = pd.Series(w, index=feature_names).sort_values(ascending=False)
        return s, {"method": "linear_coef", "kernel": "linear"}

    # Otherwise: permutation importance (model-agnostic)
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    s = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
    return s, {"method": "permutation_importance", "kernel": getattr(model, "kernel", None)}


def SVM_nested_cv(x, y, outer_splits=5, inner_splits=3, n_jobs=-1):
    def _counts_and_min(y_vec):
        y_series = pd.Series(y_vec)
        counts = y_series.value_counts()
        return counts, int(counts.min()) if not counts.empty else 0

    class_counts, min_class_count = _counts_and_min(y)
    usable_outer = min(outer_splits, min_class_count)
    if usable_outer < 2:
        raise ValueError(
            f"Not enough samples per class for outer CV. Minimum class size is {min_class_count}; "
            "need at least 2 per class."
        )

    outer_cv = StratifiedKFold(n_splits=usable_outer, shuffle=True, random_state=42)

    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }

    metrics = {
        'accuracy': [],
        'balanced_accuracy': [],
        'precision': [],
        'recall': [],
        'roc_auc': []
    }

    best_params = []

    # Store out-of-fold predictions and uncertainty diagnostics
    oof_records = []

    # Store feature importances per fold (Series)
    fold_importances = []
    fold_importance_meta = []

    for train_idx, test_idx in outer_cv.split(x, y):
        # Handle both pandas DataFrame/Series and NumPy arrays
        if isinstance(x, pd.DataFrame):
            X_train, X_test = x.iloc[train_idx], x.iloc[test_idx]
        else:
            X_train, X_test = x[train_idx], x[test_idx]

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV may need to shrink if the outer split leaves a singleton class
        _, inner_min_count = _counts_and_min(y_train)
        usable_inner = min(inner_splits, inner_min_count)
        inner_cv = StratifiedKFold(n_splits=usable_inner, shuffle=True, random_state=42) if usable_inner >= 2 else None

        svc = SVC(class_weight="balanced", probability=True, random_state=42)
        if inner_cv is None:
            best_model = svc.fit(X_train, y_train)
            best_params.append({'C': best_model.C, 'kernel': best_model.kernel, 'gamma': best_model.gamma})
        else:
            grid_search = GridSearchCV(
                estimator=svc,
                param_grid=param_grid,
                scoring='balanced_accuracy',
                cv=inner_cv,
                n_jobs=n_jobs
            )
            grid_search.fit(X_train, y_train)
            best_params.append(grid_search.best_params_)
            best_model = grid_search.best_estimator_

        y_pred, y_pred_proba, conf, ent, marg = svm_predict_with_uncertainty(best_model, X_test)

        # Save per-sample out-of-fold uncertainty diagnostics
        # Keep indices to align back to original dataframe if needed
        test_index = (
            X_test.index.to_numpy() if isinstance(X_test, pd.DataFrame) else np.asarray(test_idx)
        )
        oof_fold = pd.DataFrame({
            "index": test_index,
            "y_true": np.asarray(y_test).ravel(),
            "y_pred": np.asarray(y_pred).ravel(),
            "confidence": np.asarray(conf).ravel(),
            "entropy": np.asarray(ent).ravel(),
            "margin": np.asarray(marg).ravel(),
        })
        oof_records.append(oof_fold)

        # Feature contributions for this fold (computed on held-out fold for less bias)
        try:
            imp, imp_meta = svm_feature_contributions(best_model, X_test, y_test, scoring="balanced_accuracy")
            fold_importances.append(imp)
            fold_importance_meta.append(imp_meta)
        except Exception:
            pass

        # Determine target type to handle binary or multiclass
        from sklearn.utils.multiclass import type_of_target
        target_type = type_of_target(y)
        is_multiclass = target_type in ['multiclass', 'multilabel-indicator']

        if is_multiclass:
            avg_type = 'macro'
            roc_mode = {'multi_class': 'ovr', 'average': 'macro'}
        else:
            avg_type = 'macro'
            roc_mode = {}

        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['balanced_accuracy'].append(balanced_accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, average=avg_type, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, average=avg_type, zero_division=0))

        unique_test_classes = np.unique(y_test)
        if len(unique_test_classes) < 2:
            roc_auc_value = np.nan
        elif is_multiclass:
            roc_auc_value = roc_auc_score(y_test, y_pred_proba, **roc_mode)
        else:
            # Probability column for the positive class. For binary, sklearn orders classes_.
            pos_index = 1 if y_pred_proba is not None and y_pred_proba.shape[1] == 2 else None
            if y_pred_proba is None or pos_index is None:
                roc_auc_value = np.nan
            else:
                roc_auc_value = roc_auc_score(y_test, y_pred_proba[:, pos_index])
        metrics['roc_auc'].append(roc_auc_value)

    # Combine OOF uncertainty diagnostics
    oof_df = pd.concat(oof_records, ignore_index=True) if len(oof_records) else pd.DataFrame()

    # Aggregate feature importances across folds (align on union of feature names)
    feat_imp_mean = None
    feat_imp_std = None
    feat_imp_meta = fold_importance_meta[0] if len(fold_importance_meta) else {}
    if len(fold_importances) > 0:
        imp_df = pd.concat(fold_importances, axis=1).fillna(0.0)
        imp_df.columns = [f"fold{i}" for i in range(imp_df.shape[1])]
        feat_imp_mean = imp_df.mean(axis=1).sort_values(ascending=False)
        feat_imp_std = imp_df.std(axis=1).reindex(feat_imp_mean.index)

    results = {
        'best_params': best_params,
        'mean_metrics': {m: np.mean(v) for m, v in metrics.items()},
        'std_metrics': {m: np.std(v) for m, v in metrics.items()},
        'roc_auc_scores': metrics['roc_auc'],
        'oof_uncertainty': oof_df,
        'feature_importance_mean': feat_imp_mean,
        'feature_importance_std': feat_imp_std,
        'feature_importance_meta': feat_imp_meta,
    }

    print("Nested Cross-Validation Results:")
    for m, v in results['mean_metrics'].items():
        print(f"{m.capitalize()}: {v:.3f} ± {results['std_metrics'][m]:.3f}")

    if results.get('feature_importance_mean') is not None:
        print("\nTop feature contributions (mean across outer folds):")
        top = results['feature_importance_mean'].head(10)
        for name, val in top.items():
            print(f"  {name}: {val:.6f}")

    # --- Train final model on all data ---
    # Find most frequent best parameter combination
    from collections import Counter
    most_common_params = Counter(tuple(sorted(p.items())) for p in best_params).most_common(1)[0][0]
    best_param_dict = dict(most_common_params)
    print("\nTraining final model on full dataset with parameters:", best_param_dict)

    final_model = SVC(
        **best_param_dict,
        class_weight="balanced",
        probability=True,
        random_state=42
    )
    final_model.fit(x, y)

    return results, final_model




def plot_SVM_curves(y, y_pred_cv,recall, precision):
    # Plot the precision-recall curve
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    plt.show()

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y, y_pred_cv)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

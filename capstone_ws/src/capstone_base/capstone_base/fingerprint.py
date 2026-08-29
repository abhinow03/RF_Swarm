"""Eigengap K-estimation + spectral partition + cohesion confidence. Pure
numpy/sklearn/scipy, NO ROS. Matches the canonical RFF base station
(gen_rff/demo_router/base_station.py) eigengap_k / _partition verbatim, extended
to also return the Laplacian spectrum + winning gap for FingerprintSet, and a
real cohesion-based per-cluster confidence.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_samples
from scipy.sparse.csgraph import laplacian


def _nn(n: int) -> int:
    return int(min(15, max(3, n // 4)))


def eigengap(X: np.ndarray, kmax: int = 10):
    """Canonical eigengap: normalized-Laplacian spectrum of a symmetrized kNN
    connectivity graph; K = argmax consecutive eigenvalue gap over k in [2, kmax].
    Returns (k_hat, eigenvalues_sorted, eigengap_value)."""
    n = len(X)
    K = min(kmax, n - 1)
    if K < 2:
        return 1, np.zeros(1, float), 0.0
    A = kneighbors_graph(X, n_neighbors=min(_nn(n), n - 1),
                         mode="connectivity", include_self=False)
    A = 0.5 * (A + A.T)
    vals = np.sort(np.linalg.eigvalsh(laplacian(A, normed=True).toarray()))
    gaps = {k: vals[k] - vals[k - 1] for k in range(2, K + 1)}
    khat = int(max(gaps, key=gaps.get))
    return khat, vals, float(gaps[khat])


def partition(X: np.ndarray, K: int) -> np.ndarray:
    if K >= len(X):
        return np.arange(len(X))
    try:
        return SpectralClustering(K, affinity="nearest_neighbors", random_state=0,
                                  n_neighbors=min(_nn(len(X)), len(X) - 1)).fit_predict(X)
    except Exception:
        return KMeans(K, n_init=10, random_state=0).fit_predict(X)


def cluster_cohesion(X: np.ndarray, labels: np.ndarray, lb) -> float:
    """Mean cosine of cluster `lb`'s members to their L2 centroid (tightness).
    NOTE: cohesion alone is fooled by low DIVERSITY — a unit whose accumulated
    tracks haven't spanned enough condition variety is internally tight (high
    cohesion) yet not identity-separable. Use `separability` for confidence."""
    idx = np.where(labels == lb)[0]
    if len(idx) == 0:
        return 0.0
    c = X[idx].mean(0)
    c /= np.linalg.norm(c) + 1e-8
    return float(np.clip((X[idx] @ c).mean(), 0.0, 1.0))


def separability(X: np.ndarray, labels: np.ndarray) -> dict:
    """Per-cluster mean silhouette (cosine), clamped [0,1] -> the identity
    confidence. Unlike cohesion, this is LOW when a cluster is cohesive but not
    yet SEPARATED from its neighbours — exactly the channel-dominated /
    low-track-diversity failure mode (finding 2): tracks that haven't averaged
    across conditions stay close to other units, so silhouette (not tightness)
    reflects whether the unit is truly distinct yet. Returns {label: confidence}."""
    labs = np.asarray(labels)
    uniq = sorted(set(int(x) for x in labs))
    if len(uniq) < 2:
        return {c: 0.0 for c in uniq}      # a single cluster has no separation signal
    sil = silhouette_samples(X, labs, metric="cosine")
    return {c: float(np.clip(sil[labs == c].mean(), 0.0, 1.0)) for c in uniq}


def _merge_count(X: np.ndarray, labels: np.ndarray, tau: float) -> int:
    """Number of clusters after union-merging any two whose L2 centroids have
    cosine > tau. Recovers K when raw eigengap OVER-splits a unit into fragments
    (fragments of one unit have centroid cosine ~1; distinct provisioned units are
    < tau apart by construction). A genuine near-duplicate unit-pair (cosine > tau)
    would correctly merge to one ID — an honest 'indistinguishable' outcome."""
    ids = sorted(set(labels))
    cent = {}
    for c in ids:
        m = X[labels == c].mean(0)
        cent[c] = m / (np.linalg.norm(m) + 1e-8)
    parent = {c: c for c in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if cent[a] @ cent[b] > tau:
                parent[find(a)] = find(b)
    return len({find(c) for c in ids})


# Raw eigengap has a Goldilocks per-cluster sample count (~15-30 tracks/unit); with
# many more tracks the normalized-Laplacian spectrum smooths and argmax-gap drifts
# HIGH (over-splits units), so K is unknowable via any fixed track budget (K=2 wants
# few tracks, K=8 wants many). `discover` therefore keeps eigengap as the first pass
# for the published spectrum, but refines K by merging over-split fragments.
# tau sits ABOVE the provisioned between-unit worst centroid cosine so distinct units
# never merge, and BELOW same-unit fragment similarity so over-split fragments do.
# Raised 0.85->0.90 (2026-08-28, demo2 multi-protocol extension): the generic 512-D
# deep-tier encoders (wifi/drone_ocusync) aren't identity-metric-trained like BLE's
# 128-D native head, so real devices sit closer together in that space. Exhausted the
# real DRFF drone pool (15 -> 11 well-provisioned -> 13 with moderate inclusion) and
# confirmed 0.879 is the hard ceiling for any 6-unit separable subset — no combination
# of real units clears 0.85. Verified this doesn't regress BLE (worst-pair 0.761) or
# WiFi (worst-pair 0.689) — both keep a comfortable margin under 0.90.
MERGE_TAU = 0.90
COMPUTE_CAP = 300     # subsample this many tracks for the eigvalsh/partition (compute + density)


def discover(X: np.ndarray, kmax: int = 10, cap: int = COMPUTE_CAP,
             tau: float = MERGE_TAU, seed: int = 0):
    """Estimate K and label all tracks. Returns
    (k_final, eigenvalues, eigengap_value, labels_all, k_raw):
      k_raw   = raw eigengap estimate on a density-controlled subsample (over-splits)
      k_final = after merging fragment clusters (the trustworthy estimate)
      labels_all = k_final-way partition of the FULL X (for per-track assignment)
    """
    n = len(X)
    if n < 2:
        return 1, np.zeros(1, float), 0.0, np.zeros(n, int), 1
    rng = np.random.default_rng(seed)
    Xs = X if n <= cap else X[rng.choice(n, cap, replace=False)]
    k_raw, vals, gap = eigengap(Xs, kmax)
    k_final = _merge_count(Xs, partition(Xs, k_raw), tau)
    labels_all = partition(X, k_final) if k_final >= 2 else np.zeros(n, int)
    return int(k_final), vals, float(gap), labels_all, int(k_raw)

import numpy as np

from spred import evaluation


def test_hamming_clusters_group_near_duplicates_only():
    seqs = ["AAAAAAAA", "AAAAAAAC", "AAAAAACC", "CCCCCCCC", "CCCCCCCA"]
    g = evaluation.hamming_clusters(seqs, 2)
    assert g[0] == g[1] == g[2] and g[3] == g[4] and g[0] != g[3]


def test_hamming_clusters_match_bruteforce():
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACDE"), 8)) for _ in range(150)]
    g = evaluation.hamming_clusters(seqs, 2)
    from spred.features import index_matrix
    A = index_matrix(seqs)
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            if (A[i] != A[j]).sum() <= 2:
                assert g[i] == g[j]


def test_cross_validation_keeps_clusters_within_one_fold():
    import pandas as pd
    from spred import data
    pos = data.trainable_positives()
    fr = data.training_frame("NE", pos)
    cv = evaluation.cross_validate(fr)
    for grp in set(cv["groups"]):
        assert len(set(cv["fold"][cv["groups"] == grp])) == 1


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    s = y + rng.normal(0, 1, 400)
    from sklearn.metrics import roc_auc_score
    lo, hi = evaluation.cluster_bootstrap_auroc(y, s, np.arange(400), 300)
    assert lo < roc_auc_score(y, s) < hi

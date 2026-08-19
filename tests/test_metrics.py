from clinical_rag.eval.metrics import (
    hit_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    assert precision_at_k(retrieved, relevant, 1) == 0.0
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert precision_at_k([], relevant, 3) == 0.0
    assert precision_at_k(retrieved, relevant, 0) == 0.0


def test_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 1 / 3
    assert recall_at_k(retrieved, relevant, 4) == 2 / 3
    assert recall_at_k(retrieved, set(), 4) == 0.0


def test_hit_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"b", "z"}
    assert hit_at_k(retrieved, relevant, 1) == 0.0
    assert hit_at_k(retrieved, relevant, 2) == 1.0
    assert hit_at_k(retrieved, relevant, 10) == 1.0


def test_mrr():
    retrieved = ["a", "b", "c"]
    relevant = {"b", "c"}
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert reciprocal_rank(["x", "y"], relevant) == 0.0
    assert reciprocal_rank(["b"], relevant) == 1.0


def test_ndcg_at_k():
    relevant = {"a", "b"}
    assert ndcg_at_k(["a", "b", "c"], relevant, 2) == 1.0
    assert ndcg_at_k(["a", "b", "c"], relevant, 1) == 1.0
    # Relevant only at rank 2 with k=2: DCG=1/log2(3), IDCG=1+1/log2(3)
    import math

    dcg = 1.0 / math.log2(3)
    idcg = 1.0 + 1.0 / math.log2(3)
    assert abs(ndcg_at_k(["x", "a"], relevant, 2) - (dcg / idcg)) < 1e-9
    assert ndcg_at_k(["x", "a"], relevant, 1) == 0.0
    assert ndcg_at_k([], relevant, 5) == 0.0
    assert ndcg_at_k(["x", "y"], set(), 5) == 0.0


def test_percentile():
    assert percentile([10, 20, 30, 40], 50) == 25.0
    assert percentile([], 95) == 0.0

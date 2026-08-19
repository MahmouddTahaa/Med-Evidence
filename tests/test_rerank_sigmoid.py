from clinical_rag.retrieval.rerank import sigmoid


def test_sigmoid_bounds_and_midpoint():
    assert 0.0 < sigmoid(-20.0) < 0.01
    assert abs(sigmoid(0.0) - 0.5) < 1e-12
    assert 0.99 < sigmoid(20.0) <= 1.0
    assert sigmoid(2.0) > sigmoid(1.0) > sigmoid(0.0) > sigmoid(-1.0)

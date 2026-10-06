import numpy as np

from sbahn.models.uncertainty import bootstrap_f1_interval, two_proportion_ztest

LABELS = np.array([0, 0, 0, 1, 1, 1, 1, 0])
PREDICTED = np.array([0, 1, 0, 1, 1, 0, 1, 0])


def test_bootstrap_interval_contains_the_point_f1_and_repeats():
    first = bootstrap_f1_interval(LABELS, PREDICTED, n_resamples=200, seed=42)
    second = bootstrap_f1_interval(LABELS, PREDICTED, n_resamples=200, seed=42)
    assert first["low"] <= first["f1"] <= first["high"]
    assert first["low"] < first["high"]
    assert first == second


def test_two_proportion_ztest_matches_the_hand_calculation():
    tested = two_proportion_ztest(8, 10, 2, 10)
    np.testing.assert_allclose(tested["rate_difference"], 0.6)
    np.testing.assert_allclose(tested["z"], 2.683281573, atol=1e-6)
    np.testing.assert_allclose(tested["p_value"], 0.007290358, atol=1e-6)

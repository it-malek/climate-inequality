"""Scientific regressions for cross-group redundancy, not preferred outcomes."""
import numpy as np
import pandas as pd
import pytest

from research.model_v2 import cv
from research.model_v2.m0 import design_matrix
from src.decomposition import group_lmg_shares


def orthogonal_example():
    p = np.array([-1., -1., 1., 1.]) + 7
    c = np.array([-1., 1., -1., 1.]) + 2
    return pd.DataFrame(dict(
        Country=list('ABCD'), warming_trend=p,
        population=10**p, cum_co2_per_capita=10**c,
        cum_co2_total=10**(c+p-6),
    ))


def test_audit_detects_exact_identity_and_preserves_predictions():
    from research.model_v2.rank_audit import audit_identity, parameterizations

    d = orthogonal_example()
    report = audit_identity(d)
    assert report['identity_constant'] == pytest.approx(6)
    assert report['max_abs_identity_error'] < 1e-12
    full = cv.V1Design().fit(d).predict(d)
    for name, frame in parameterizations(d).items():
        x, _, _ = design_matrix(frame)
        assert np.allclose(cv.V1Design().fit(frame).predict(frame), full)
        if name != 'M0_legacy_representation':
            assert np.linalg.matrix_rank(x) == x.shape[1]


def test_known_shapley_redistribution_with_unchanged_column_space():
    from research.model_v2.rank_audit import parameterizations

    variants = parameterizations(orthogonal_example())
    baseline = group_lmg_shares(variants['M0_legacy_representation'])
    total = group_lmg_shares(variants['drop_per_capita'])
    percap = group_lmg_shares(variants['drop_total'])
    assert baseline.shares['emissions'] == pytest.approx(0.5)
    assert total.shares['emissions'] == pytest.approx(0.25)
    assert percap.shares['emissions'] == pytest.approx(0.0, abs=1e-12)
    assert baseline.total_r2 == total.total_r2 == percap.total_r2 == 1


def test_identity_audit_refuses_near_but_not_structural_relation():
    from research.model_v2.rank_audit import audit_identity

    d = orthogonal_example()
    d.loc[0, 'cum_co2_per_capita'] *= 1.001
    with pytest.raises(ValueError, match='identity'):
        audit_identity(d)


def test_null_direction_produces_nonidentifiable_coefficients():
    from research.model_v2.rank_audit import audit_identity

    d = orthogonal_example()
    r = audit_identity(d)
    x, columns, _ = design_matrix(d)
    null = np.array([r['null_vector'].get(name, 0.) for name in columns])
    beta = np.linalg.lstsq(x, d.warming_trend, rcond=None)[0]
    assert np.allclose(x @ beta, x @ (beta + 100 * null))
    assert np.max(np.abs(100 * null)) >= 600


def test_saved_real_results_reproduce_from_pinned_inputs_and_folds():
    """The committed rank-audit artifacts reproduce from the pinned inputs and the primary folds."""
    import json
    from research.model_v2 import m0, territory
    from research.model_v2.rank_audit import parameterizations

    if not (m0.DEFAULT_INEQUALITY_PATH.exists() and m0.DEFAULT_FEATURES_PATH.exists()):
        pytest.skip('requires the local pinned inputs')
    design = m0.m0_complete_design(*m0.load_inputs())
    saved = json.loads((m0.OUTPUT_DIR / 'rank_audit.json').read_text())
    countries = pd.read_csv(m0.OUTPUT_DIR / 'm0_countries.csv')
    predictions = pd.read_csv(m0.OUTPUT_DIR / 'rank_audit_country_predictions.csv')
    distance = territory.load_bounds(countries.iso3.tolist())
    for name, frame in parameterizations(design).items():
        prior = predictions[predictions.variant == name]
        assert prior.Country.tolist() == frame.Country.tolist()
        fit = cv.V1Design().fit(frame).predict(frame)
        result = cv.cross_validate(frame, np.arange(len(frame)), dist=distance, buffer_km=500)
        np.testing.assert_allclose(prior.fitted, fit, rtol=0, atol=1e-12)
        np.testing.assert_allclose(prior.cv_prediction, result.yhat, rtol=0, atol=1e-12)
        decomposition = group_lmg_shares(frame)
        for group, share in decomposition.shares.items():
            assert saved['cards'][name][f'share_{group}'] == pytest.approx(share, abs=1e-12)
        assert np.linalg.matrix_rank(design_matrix(frame)[0]) == saved['cards'][name]['matrix_rank']

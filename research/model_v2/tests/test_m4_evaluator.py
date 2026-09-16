"""M4 evaluator: V1-identical resampling, bootstrap summaries and materiality, sensitivities, inventory status.

Synthetic data only, apart from digest checks of the committed ERA5 files. No test runs M4 on the warming
outcome.
"""
import numpy as np
import pandas as pd
import pytest

from research.model_v2 import m2_evaluate as ev
from research.model_v2 import m4_evaluate as m4
from research.model_v2 import v2_provenance as prov
from research.model_v2.tests.test_m2_evaluator import synthetic_frozen
from src import stability


@pytest.fixture(scope='module')
def frozen():
    return synthetic_frozen(n=60, seed=5)


def test_country_draws_are_the_v1_resampling_sequence(frozen):
    frame = frozen.frames['primary_total_co2']
    ours = np.random.default_rng(0)
    v1 = np.random.default_rng(0)
    for _ in range(5):
        idx = ours.integers(0, len(frame), len(frame))
        expected = stability._resample_countries(frame, v1)
        assert frame.iloc[idx].reset_index(drop=True).equals(expected)


def test_block_draws_are_the_v1_resampling_sequence(frozen):
    frame = frozen.frames['primary_total_co2']
    ours = np.random.default_rng(1)
    v1 = np.random.default_rng(1)
    blocks = pd.unique(frame['spatial_block'])
    for _ in range(5):
        chosen = ours.choice(blocks, size=len(blocks), replace=True)
        draw = pd.concat([frame[frame['spatial_block'] == c] for c in chosen], ignore_index=True)
        assert draw.equals(stability._resample_blocks(frame, v1))


@pytest.mark.parametrize('model', [ev.COMPARATOR, ev.CANDIDATE])
def test_bootstrap_draws_record_every_draw_and_account(frozen, monkeypatch, model):
    monkeypatch.setattr(m4, 'N_BOOT', 12)
    frame = frozen.frames['primary_total_co2']
    draws = m4.bootstrap(model, frame, 'country')
    assert len(draws) == 12 and draws.draw.tolist() == list(range(12))
    usable = draws[draws.usable]
    total = usable[list(m4.SHARE_ORDER)].sum(axis=1) + usable['residual']
    assert np.allclose(total, 1.0, atol=1e-9)


def test_a_degenerate_draw_is_counted_not_replaced(frozen, monkeypatch):
    monkeypatch.setattr(m4, 'N_BOOT', 3)
    frame = frozen.frames['primary_total_co2'].assign(abs_latitude=5.0)
    draws = m4.bootstrap(ev.CANDIDATE, frame, 'country')
    assert (~draws.usable).all() and draws.failure.str.contains('DegenerateLatitudeBasis').all()


def _draws(geo, other, n=200):
    rows = []
    for kind in ('country', 'block'):
        for b in range(n):
            rows.append({'bootstrap': kind, 'draw': b, 'n_rows': 60, 'usable': True, 'failure': '',
                         'rank_deficient': False, 'emissions': 0.02, 'geography': geo[b % len(geo)],
                         'socioeconomic': other[b % len(other)], 'population': 0.03, 'residual': 0.3})
    return pd.DataFrame(rows)


def test_material_change_flags():
    point = {'emissions': 0.02, 'geography': 0.5, 'socioeconomic': 0.06, 'population': 0.03, 'residual': 0.39}
    stable = m4.summarize_bootstrap(point, _draws([0.5, 0.45, 0.55], [0.06, 0.07]))
    assert stable['material_change']['material_change_to_v1_conclusion'] is False
    overlapping = m4.summarize_bootstrap(point, _draws([0.5, 0.05, 0.55], [0.06, 0.2]))
    assert overlapping['material_change']['interval_geography_not_established_country'] is True
    assert overlapping['material_change']['material_change_to_v1_conclusion'] is True
    heavy = m4.summarize_bootstrap({**point, 'emissions': 0.11}, _draws([0.5], [0.06]))
    assert heavy['material_change']['responsibility_exceeds_0_10'] is True


def test_largest_named_ties_go_to_the_earlier_schema_group():
    row = {'emissions': 0.3, 'geography': 0.3, 'socioeconomic': 0.1, 'population': 0.1}
    assert m4.largest_named(row) == 'emissions'


def test_sensitivity_protocols_and_distances(frozen):
    territory = m4.sensitivity_frozen(frozen, 'buffer_territory_1000km')
    assert np.array_equal(territory.distance, frozen.distance)
    assert territory.folds['buffer_territory_1000km'][1] == 1000.0


def test_cv_scores():
    y, yhat = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0])
    scores = m4.cv_scores(y, yhat)
    assert scores['cv_rmse'] == pytest.approx(np.sqrt(1 / 3)) and scores['cv_r2'] == pytest.approx(0.5)


def test_the_committed_era5_files_match_their_pins():
    assert prov.sha256(m4.ROOT / m4.LEGACY_ERA5) == m4.LEGACY_ERA5_SHA256
    summary = __import__('json').loads((m4.ROOT / m4.ALIGNED_SUMMARY).read_text())
    assert prov.sha256(m4.ROOT / m4.ALIGNED_ERA5) == summary['aligned_outcome_sha256']


def test_inventory_status_covers_every_row_and_flags_missing_bootstrap():
    uncertainty = {r: {'country': {'usable': 1}, 'block': {'usable': 1}} for r in ev.REPRESENTATIONS}
    product = {'aligned_era5/primary_total_co2': {}, 'legacy_era5/primary_total_co2': {}}
    status = m4.inventory_status('M0*', [], {}, uncertainty, {}, product)
    assert [item['id'] for item in status['items']] == [f'I{k}' for k in range(1, 28)]
    assert all(item['status'] in ('verified evidence', 'permitted disposition') for item in status['items'])
    missing = {r: {'country': {'usable': 0}, 'block': {'usable': 1}} for r in ev.REPRESENTATIONS}
    assert m4.inventory_status('M0*', [], {}, missing, {}, product)['items'][9]['status'] == 'missing'


def test_a_group_without_columns_in_a_draw_keeps_a_zero_share(frozen):
    frame = frozen.frames['primary_total_co2'].assign(income_group='High-income countries')
    shares, failure, _ = m4.draw_shares(ev.COMPARATOR, frame)
    assert failure is None and shares['socioeconomic'] == 0.0
    total = sum(shares[g] for g in m4.SHARE_ORDER) + shares['residual']
    assert total == pytest.approx(1.0, abs=1e-12)

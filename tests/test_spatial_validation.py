"""Synthetic contracts for the prospectively fixed spatial validation."""
import numpy as np
import pandas as pd
import pytest

from src import spatial_validation as sv


def frame(n=40):
    return pd.DataFrame({"Country": [f"C{i % 4}" for i in range(n)], "City": [f"City{i:03}" for i in range(n)], "Latitude": np.linspace(-60, 60, n), "Longitude": np.linspace(-170, 170, n), "slope_c_per_decade": np.linspace(.1, .4, n)})


def test_coordinate_groups_and_buffer_are_disjoint():
    cities = frame()
    cities.loc[1, ["Latitude", "Longitude"]] = cities.loc[0, ["Latitude", "Longitude"]].to_numpy()
    ordered, folds = sv.interpolation_folds(cities)
    for name, labels in folds.items():
        assert ordered.assign(fold=labels).groupby(sv.COORDINATES).fold.nunique().max() == 1
        for fold in np.unique(labels):
            train, test = sv.interpolation_masks(ordered, labels, fold, 250 if "spatial" in name else 0)
            assert not (train & test).any()
            assert not set(map(tuple, ordered.loc[train, sv.COORDINATES].to_numpy())) & set(map(tuple, ordered.loc[test, sv.COORDINATES].to_numpy()))
    changed = cities.assign(slope_c_per_decade=-cities.slope_c_per_decade)
    _, again = sv.interpolation_folds(changed)
    for name in folds:
        np.testing.assert_array_equal(folds[name], again[name])


def test_variogram_gets_only_training_responses(monkeypatch):
    cities = frame()
    train = np.arange(40) < 32
    captured = []
    monkeypatch.setattr(sv, "fit_variogram_parameters", lambda lon, lat, value, **kw: captured.append(value.copy()) or [1., 30., 0.])
    def kriging(*args, **kw):
        assert kw["variogram_parameters"] == {"psill": 1., "range": 30., "nugget": 0.}
        return np.ones(8), np.ones(8)
    monkeypatch.setattr(sv, "ordinary_kriging_interpolate", kriging)
    sv.predict_interpolation_fold(cities, train, ~train)
    np.testing.assert_array_equal(captured[0], cities.loc[train, "slope_c_per_decade"])


def test_group_weighting_does_not_weight_colocated_cities_twice():
    predictions = pd.DataFrame({"Latitude": [0, 0, 1], "Longitude": [0, 0, 1], "slope_c_per_decade": [0, 0, 0], "idw": [1, 1, 3], "kriging": [1, 1, 3]})
    scores = sv.interpolation_scores(predictions)["idw"]
    assert scores["coordinate_group"]["rmse"] == pytest.approx(np.sqrt(5))
    assert scores["city_row"]["rmse"] == pytest.approx(np.sqrt(11 / 3))


def test_tied_zero_distance_graph_retains_records_and_is_order_invariant():
    cities = frame(16)
    cities[["Latitude", "Longitude"]] = 0.
    result, graphs, acceptance = sv.neighbor_order_validation(cities, np.arange(16.) ** 2)
    assert acceptance["accepted"]
    assert len(result) == 14 * 2 * 2
    canonical = sv.canonical_frame(cities)
    graph = sv.neighbor_graph(canonical, "identity-distance-ties-v1")
    assert graph.shape == (16, 8)
    for i, row in enumerate(graph):
        assert i not in row
        np.testing.assert_array_equal(row, [j for j in range(16) if j != i][:8])
    assert all(g.shape == (16, 8) for g in graphs.values())


def test_duplicate_identity_rejected():
    cities = pd.concat([frame(10), frame(10).iloc[:1]])
    with pytest.raises(ValueError, match="duplicate persistent"):
        sv.canonical_frame(cities)


def test_pykrige_list_input_changes_partial_sill_but_dict_preserves_it():
    from pykrige.ok import OrdinaryKriging
    coordinates = dict(x=np.array([0., 1., 0., 1.]), y=np.array([0., 0., 1., 1.]), z=np.array([0., 1., 2., 3.]), variogram_model="spherical", coordinates_type="geographic", pseudo_inv=True)
    correct = OrdinaryKriging(**coordinates, variogram_parameters={"psill": 1., "range": 30., "nugget": .2})
    wrong = OrdinaryKriging(**coordinates, variogram_parameters=[1., 30., .2])
    np.testing.assert_allclose(correct.variogram_model_parameters, [1., 30., .2])
    np.testing.assert_allclose(wrong.variogram_model_parameters, [.8, 30., .2])
    _, correct_variance = correct.execute("points", np.array([.3]), np.array([.4]))
    _, wrong_variance = wrong.execute("points", np.array([.3]), np.array([.4]))
    assert not np.isclose(correct_variance[0], wrong_variance[0], atol=1e-12, rtol=0)


def test_holdout_response_perturbation_cannot_change_training_fit_or_predictions(monkeypatch):
    cities = frame()
    train = np.arange(40) < 32
    fitted_responses = []
    def variogram(lon, lat, response, **kwargs):
        fitted_responses.append(response.copy())
        return [float(response.mean()), 30., .01]
    def kriging(*args, **kwargs):
        # Training response mean deliberately makes any accidentally supplied held-out values visible.
        return np.full(len(args[3]), args[2].mean()), np.ones(len(args[3]))
    monkeypatch.setattr(sv, "fit_variogram_parameters", variogram)
    monkeypatch.setattr(sv, "ordinary_kriging_interpolate", kriging)
    first, parameters, _ = sv.predict_interpolation_fold(cities, train, ~train)
    altered = cities.copy()
    altered.loc[~train, "slope_c_per_decade"] = np.arange(8) * 1e12
    second, changed_parameters, _ = sv.predict_interpolation_fold(altered, train, ~train)
    assert parameters == changed_parameters
    np.testing.assert_array_equal(fitted_responses[0], fitted_responses[1])
    for method in first:
        np.testing.assert_array_equal(first[method], second[method])


def test_buffer_removes_nearby_training_coordinates_without_changing_test_rows():
    cities = frame(4)
    cities.Latitude = 0.
    cities.Longitude = [0., 1., 3., 10.]
    train, test = sv.interpolation_masks(cities, np.array([0, 1, 1, 1]), 0, 250)
    np.testing.assert_array_equal(test, [True, False, False, False])
    np.testing.assert_array_equal(train, [False, False, True, True])

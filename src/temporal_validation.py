"""Fixed prospective rolling-origin evaluation and descriptive rank uncertainty.

These routines do not select a new model or assert independent coverage trials.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import physical_model as pm


def fit_with_trace(x, y, mask):
    """Mirror the existing fixed estimator while exposing convergence evidence."""
    xt, yt = x[mask], y[mask]
    rho = 0.0
    trace = []
    for iteration in range(pm.MAX_ITERS):
        xw, yw = pm.prais_winsten_whiten(xt, rho), pm.prais_winsten_whiten(yt, rho)
        lam = pm.select_lambda(xw, yw)
        mean, _, _, _ = pm.nig_posterior(xw, yw, lam)
        next_rho = pm.estimate_rho(yt - xt @ mean)
        delta = abs(next_rho - rho)
        trace.append({'iteration': iteration + 1, 'rho_in': rho, 'rho_out': next_rho,
                      'lambda': lam, 'delta': delta})
        rho = next_rho
        if delta < pm.RHO_TOL:
            break
    xw, yw = pm.prais_winsten_whiten(xt, rho), pm.prais_winsten_whiten(yt, rho)
    lam = pm.select_lambda(xw, yw)
    mean, covariance, alpha, beta = pm.nig_posterior(xw, yw, lam)
    fit = pm.FitState(mean, covariance, alpha, beta, rho, lam,
                      np.zeros(0), np.ones(0), int(mask.sum()))
    return fit, {'converged': bool(trace[-1]['delta'] < pm.RHO_TOL),
                 'iterations': trace, 'final_rho': rho, 'final_lambda': lam}


def rolling_origin(forcings, origins=range(1990, 2014), max_horizon=11):
    """Retrospective conditional hindcasts with every fit restricted to its origin."""
    years, raw, y, _ = pm.build_design(forcings)
    rows, fits = [], []
    for origin in origins:
        mask = years <= origin
        if not np.any(years == origin) or not np.any(years == origin + max_horizon):
            raise ValueError('Incomplete fixed origin/horizon support')
        x, mu, scale = pm.standardize(raw, mask)
        fit, record = fit_with_trace(x, y, mask)
        trajectory = pm.predict_trajectory(fit, x, y, years, mask, train_end=origin)
        future = trajectory.loc[(years > origin) & (years <= origin + max_horizon)].copy()
        future['origin'] = origin
        future['horizon'] = future.year - origin
        future['error'] = future.predicted_mean - future.observed
        future['width'] = future.upper95 - future.lower95
        future['covered'] = future.observed.between(future.lower95, future.upper95)
        future['interval_score'] = (future.width + 40 * (future.lower95 - future.observed).clip(lower=0)
                                    + 40 * (future.observed - future.upper95).clip(lower=0))
        rows.append(future)
        fits.append({'origin': origin, 'train_start': int(years[mask][0]),
                     'train_end': int(years[mask][-1]), 'n_train': int(mask.sum()),
                     'training_mean': mu.tolist(), 'training_sd': scale.tolist(), **record})
    predictions = pd.concat(rows, ignore_index=True)
    summary = []
    for horizon, part in predictions.groupby('horizon'):
        summary.append({'horizon': int(horizon), 'n': len(part),
                        'rmse': float(np.sqrt(np.mean(part.error**2))),
                        'mae': float(np.mean(abs(part.error))), 'bias': float(part.error.mean()),
                        'coverage': float(part.covered.mean()), 'covered_count': int(part.covered.sum()),
                        'mean_width': float(part.width.mean()), 'min_width': float(part.width.min()),
                        'max_width': float(part.width.max()), 'interval_score': float(part.interval_score.mean())})
    return predictions, fits, summary


def rank_uncertainty(frame, x, y, *, block='continent', n=2000):
    """Country and whole-continent percentile bootstrap; undefined draws retained."""
    pair = frame.loc[np.isfinite(frame[x]) & np.isfinite(frame[y])].copy()
    if pair[block].isna().any():
        raise ValueError('Missing block in finite-pair sample')
    if len(pair) < 3:
        raise ValueError('Too few pairs')
    groups = [np.flatnonzero(pair[block].to_numpy() == b) for b in sorted(pair[block].unique())]
    a, b = pair[x].to_numpy(), pair[y].to_numpy()
    draws, summaries = [], {}
    for kind, seed in [('country', 0), ('continent', 1)]:
        rng = np.random.default_rng(seed)
        values = []
        for draw in range(n):
            indices = (rng.integers(0, len(pair), len(pair)) if kind == 'country'
                       else np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
            valid = len(np.unique(a[indices])) > 1 and len(np.unique(b[indices])) > 1
            value = float(spearmanr(a[indices], b[indices]).statistic) if valid else None
            if value is not None:
                values.append(value)
            draws.append({'kind': kind, 'seed': seed, 'draw': draw, 'n': len(indices), 'rho': value})
        ci = np.percentile(values, [2.5, 97.5]).tolist() if values else [None, None]
        summaries[kind] = {'ci95': ci, 'valid': len(values), 'undefined': n-len(values), 'draws': n, 'seed': seed}
    return {'x': x, 'y': y, 'n': len(pair), 'blocks': len(groups),
            'countries': pair.Country.tolist(), 'rho': float(spearmanr(a, b).statistic),
            'bootstrap': summaries}, pd.DataFrame(draws)

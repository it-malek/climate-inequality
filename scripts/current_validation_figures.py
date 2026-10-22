"""Render current-validation summaries from a verified bundle; no model fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.snapshot import fingerprint


def render(bundle: Path, output: Path) -> None:
    identity = fingerprint(bundle, require_manifest=True)
    names = ['stats.json', 'physical_summary.json', 'vulnerability_summary.json',
             'stability_summary.json']
    stats, physical, vulnerability, stability = [json.loads((bundle / n).read_text()) for n in names]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    blue, orange = '#245c87', '#ad6226'
    validation = stats['interpolation']['validation']
    ax = axes[0, 0]
    for i, method in enumerate(['idw', 'kriging']):
        values = [v[method]['coordinate_group']['rmse'] for v in validation.values()]
        ax.bar(np.arange(2) + (i - .5) * .3, values, width=.3,
               label=method.upper() if i == 0 else 'Local kriging', color=[blue, orange][i])
    ax.set(xticks=[0, 1], xticklabels=['Grouped 5-fold', 'Spatial 10-fold, 250 km'],
           ylabel='Coordinate-group RMSE (°C/decade)', title='Training-only interpolation')
    ax.legend(frameon=False)
    rows = physical['rolling_origin_validation']['by_horizon']
    ax = axes[0, 1]
    ax.plot([r['horizon'] for r in rows], [r['rmse'] for r in rows], 'o-', color=blue)
    ax.set(xlabel='Horizon (years)', ylabel='RMSE (°C)', ylim=(0, .115),
           title='Conditional rolling-origin hindcast')
    ax.text(.03, .05, '24 dependent predictions per horizon\nObserved future drivers; coverage 23/24 or 24/24',
            transform=ax.transAxes, fontsize=9)
    selected = [r for r in vulnerability['rank_uncertainty']['evaluations']
                if r['sample'] == 'pair_specific' and r['y'] in
                ['cum_co2_t_per_capita', 'trend_c_per_decade_area_weighted']]
    ax = axes[1, 0]
    labels = []
    for i, row in enumerate(selected):
        lo, hi = row['bootstrap']['country']['ci95']
        ax.errorbar(row['rho'], i, xerr=[[row['rho']-lo], [hi-row['rho']]], fmt='o', color=blue, capsize=3)
        labels.append(('Income' if row['x'] == 'income_rank' else 'Vulnerability') + ' / ' +
                      ('responsibility' if row['y'] == 'cum_co2_t_per_capita' else 'area warming') + f" (n={row['n']})")
    ax.axvline(0, color='#666666', linewidth=1)
    ax.set(yticks=range(len(labels)), yticklabels=labels, xlim=(-1, 1),
           xlabel='Spearman ρ, country-bootstrap 95% interval', title='Income and updated composite vulnerability')
    ax.invert_yaxis()
    ax = axes[1, 1]
    values = [stability, stability['sensitivity']['station_weighted']]
    for i, summary in enumerate(values):
        g = summary['share_stability']['groups']['geography']
        ax.errorbar(g['point'], i, xerr=[[g['point']-g['ci_low']], [g['ci_high']-g['point']]],
                    fmt='o', capsize=3, color=[blue, orange][i])
    ax.set(yticks=[0, 1], yticklabels=['Area-weighted', 'City-location mean'], xlim=(0, .75),
           xlabel='Geography share of total variance', title='V1 complete-case bootstrap: 151 countries')
    ax.invert_yaxis()
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x' if ax in axes[1] else 'y', alpha=.15)
    fig.suptitle('Climate inequality · analysis 2.1.0', fontsize=15)
    fig.text(.03, .025, 'Descriptive validation on previously examined data. Country resampling assumes exchangeability; six-continent sensitivities are reported separately.\nND-GAIN: 2026 release_data_v2, observation year 2023. Warming: 1950–2013. Interval overlap is not an equivalence test.', fontsize=9)
    fig.tight_layout(rect=(.01, .08, 1, .96), w_pad=2.5, h_pad=3)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'current_validation.png', dpi=180)
    plt.close(fig)
    (output / 'current_validation.sources.json').write_text(json.dumps({
        'bundle_sha256': identity, 'generator': 'python -m scripts.current_validation_figures',
        'sources': {n: hashlib.sha256((bundle / n).read_bytes()).hexdigest() for n in names},
        'figure_sha256': hashlib.sha256((output / 'current_validation.png').read_bytes()).hexdigest(),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'matplotlib': matplotlib.__version__, 'scope': 'Presentation only; no estimates refitted.'
    }, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('app/data'))
    parser.add_argument('--output', type=Path, default=Path('reports/figures'))
    args = parser.parse_args()
    render(args.bundle, args.output)


if __name__ == '__main__':
    main()

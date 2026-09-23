"""Run the pre-outcome redundancy diagnostic on the accepted C2 package, instrumented.

Wraps ``m1b_redundancy.main()`` without changing it. During the run:
* every ``pandas.read_parquet`` call is recorded, and a call without a column allow-list raises;
* every ``pandas.read_csv`` call is recorded with its ``usecols``;
* file opens are recorded through an audit hook (Python-level opens of str/Path arguments under the
  repository root, excluding source files; native readers that bypass Python's ``open`` audit event would
  not appear, which is why the parquet spy is the primary guard);
* the outcome-bearing design loaders (``m0.load_inputs``, ``m0.m0_complete_design``,
  ``src.decomposition.build_country_design``) raise if called.

Writes ``outputs/m1b_redundancy_instrumentation.json`` next to the diagnostic record. The warming
outcome is never loaded; selected predictor columns are read from parquet containers that also hold
outcome columns.

    python -m research.model_v2.feasibility.m1b_redundancy_instrumented
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

import src.decomposition
from research.model_v2 import m0
from research.model_v2 import m1b_hydroclimate as h
from research.model_v2 import m1b_redundancy as r

INSTRUMENTATION = h.OUT / 'm1b_redundancy_instrumentation.json'
FORBIDDEN_WORDS = ('trend', 'slope', 'warming', 'fitted', 'resid', 'era5', 'score', 'coverage')


def main():
    calls = {'parquet': [], 'csv': [], 'opens': [], 'forbidden_loader_calls': []}
    active = {'on': True}

    def audit(event, args):
        if active['on'] and event == 'open' and args and isinstance(args[0], (str, Path)):
            path = str(args[0])
            if path.startswith(str(m0.ROOT)) and not path.endswith(('.py', '.pyc')) and '__pycache__' not in path:
                calls['opens'].append(str(Path(path).relative_to(m0.ROOT)))

    real_parquet, real_csv = pd.read_parquet, pd.read_csv

    def parquet(path, columns=None, **kwargs):
        calls['parquet'].append({'path': str(Path(path).relative_to(m0.ROOT)), 'columns': columns})
        if columns is None:
            raise AssertionError('parquet read without a column allow-list')
        return real_parquet(path, columns=columns, **kwargs)

    def csv(path, *args, **kwargs):
        name = '<verified C2 bytes>' if hasattr(path, 'read') else str(Path(path).relative_to(m0.ROOT))
        calls['csv'].append({'path': name, 'usecols': kwargs.get('usecols')})
        return real_csv(path, *args, **kwargs)

    def forbidden(name):
        def raiser(*args, **kwargs):
            calls['forbidden_loader_calls'].append(name)
            raise AssertionError(f'outcome-bearing loader {name} called')
        return raiser

    originals = (pd.read_parquet, pd.read_csv, m0.load_inputs, m0.m0_complete_design, src.decomposition.build_country_design)
    sys.addaudithook(audit)
    pd.read_parquet, pd.read_csv = parquet, csv
    m0.load_inputs, m0.m0_complete_design = forbidden('m0.load_inputs'), forbidden('m0.m0_complete_design')
    src.decomposition.build_country_design = forbidden('src.decomposition.build_country_design')
    try:
        result = r.main()
    finally:
        active['on'] = False
        (pd.read_parquet, pd.read_csv, m0.load_inputs, m0.m0_complete_design,
         src.decomposition.build_country_design) = originals

    expected_parquet = [{'path': str(Path(r.DEFAULT_INEQUALITY_PATH).relative_to(m0.ROOT)), 'columns': r.INEQUALITY_COLUMNS},
                        {'path': str(Path(r.DEFAULT_FEATURES_PATH).relative_to(m0.ROOT)), 'columns': r.CITY_COLUMNS}]
    expected_csv = [{'path': '<verified C2 bytes>', 'usecols': None},
                    {'path': str(Path(r.SUPPORT_RECORD).relative_to(m0.ROOT)), 'usecols': ['iso3', 'Country']},
                    {'path': str(Path(r.INCOME_PATH).relative_to(m0.ROOT)), 'usecols': None}]
    allowed_opens = {str(Path(p).relative_to(m0.ROOT)) for p in r.PREDICTOR_INPUT_SHA256} | {
        str((h.OUT / n).relative_to(m0.ROOT)) for n in (*h.DETERMINISTIC_OUTPUTS, r.RECORD)}
    checks = {
        'parquet_reads_exactly_the_allow_lists': calls['parquet'] == expected_parquet,
        'no_forbidden_column_requested': not any(w in c.lower() for p in calls['parquet'] for c in p['columns'] for w in FORBIDDEN_WORDS),
        'csv_reads_exactly_expected': calls['csv'] == expected_csv,
        'no_m0_countries_dependency': not any('m0_countries' in x for x in calls['opens'] + [c['path'] for c in calls['csv']]),
        'opens_within_allow_list': set(calls['opens']) <= allowed_opens,
        'no_outcome_bearing_loader_called': calls['forbidden_loader_calls'] == [],
        'outcome_column_absent_from_predictors': result['provenance']['outcome_column_present'] is False,
        'predictor_columns_exact': result['provenance']['predictor_columns'] == r.PREDICTOR_COLUMNS,
    }
    record_bytes = (h.OUT / r.RECORD).read_bytes()
    instrumentation = {'instrumentation_script_sha256': h.sha256(Path(__file__)),
                       'python_hash_seed': __import__('os').environ.get('PYTHONHASHSEED'),
                       'checks': checks, 'all_checks_pass': all(checks.values()), 'calls': calls,
                       'redundancy_record_sha256': hashlib.sha256(record_bytes).hexdigest(),
                       'redundancy_status': result['status'], 'hard_stops': result['hard_stops']}
    INSTRUMENTATION.write_text(json.dumps(instrumentation, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'checks': checks, 'status': result['status'], 'hard_stops': result['hard_stops']}, indent=2))
    if not instrumentation['all_checks_pass']:
        raise AssertionError('redundancy instrumentation check failed')
    return instrumentation


if __name__ == '__main__':
    sys.exit(0 if main()['redundancy_status'] == 'pass' else 2)

"""Initialize an exclusive private run from the portable frozen scientific spec.

Run with ``python -m scripts.init_current_validation --run-root PATH --raw-root PATH``.
This prepares evidence plumbing only; it does not execute scientific evaluators.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from src.validation_receipts import ValidationReceipts, dump, sha

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ADAPTER = '''"""Generated private adapter for a portable prospective validation run."""
import json
from pathlib import Path
import sys
RUN = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from src.validation_receipts import ValidationReceipts, dump, sha
_spec = json.loads((RUN / 'VALIDATION_SPEC.json').read_text())
RAW = Path(_spec['inputs']['raw_root'])
_previous = _spec['inputs'].get('previous_run')
OLD = Path(_previous) if _previous else None
SPEC_HASH = json.loads((RUN / 'SPEC_FREEZE.json').read_text())['sha256']
_receipts = ValidationReceipts(ROOT, RUN, RAW, SPEC_HASH)
stage = _receipts.stage
verified_stage = _receipts.verified_stage
'''


def initialize(run_root, raw_root, *, repository_root=REPOSITORY_ROOT, previous_run=None):
    root = Path(repository_root).resolve(strict=True)
    run = Path(run_root).absolute()
    raw = Path(raw_root).resolve(strict=True)
    template = root / 'docs/current_validation_spec.json'
    spec = json.loads(template.read_text())
    if spec['inputs']['raw_root'] != '${RAW_ROOT}' or spec['inputs']['previous_run'] != '${PREVIOUS_RUN}':
        raise ValueError('portable specification must retain symbolic machine paths')
    previous = Path(previous_run).resolve(strict=True) if previous_run is not None else None
    spec['inputs']['raw_root'] = str(raw)
    spec['inputs']['previous_run'] = str(previous) if previous else None
    # Validate confinement, raw availability and aliases before creating anything.
    ValidationReceipts(root, run, raw, 'not-yet-frozen')
    if run.exists():
        raise FileExistsError('run already exists; select a new run root')
    run.mkdir(parents=True, exist_ok=False)
    control = run / 'control'
    control.mkdir()
    dump(run / 'VALIDATION_SPEC.json', spec)
    dump(run / 'SPEC_FREEZE.json', {'sha256': sha(run / 'VALIDATION_SPEC.json'), 'status': 'frozen before any stage execution', 'initialized_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'portable_template_sha256': sha(template), 'original_frozen_spec_sha256': spec['provenance']['original_frozen_spec_sha256']})
    (control / 'runlib.py').write_text(ADAPTER)
    dump(control / 'interpolation-implementation-resolution.json', {'record_type': 'pre-execution implementation resolution', 'frozen_spec_changed': False, 'resolution': 'Pass fitted PyKrige internal partial sill via explicit psill/range/nugget dictionary; list input instead denotes full sill. New comparison includes parameter handoff correctness in addition to training-only fitting.', 'synthetic_regression': 'tests/test_spatial_validation.py::test_pykrige_list_input_changes_partial_sill_but_dict_preserves_it', 'original_frozen_spec_sha256': spec['provenance']['original_frozen_spec_sha256']})
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--previous-run', type=Path)
    args = parser.parse_args()
    print(initialize(args.run_root, args.raw_root, previous_run=args.previous_run))


if __name__ == '__main__':
    main()

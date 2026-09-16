"""Execution guard and deterministic serialization shared by the M2, M3 and M4 evaluators.

The guard implements the execution boundary of ``M2_EVALUATOR_SPEC.md`` §1 and
``DOWNSTREAM_COMPLETION_SPEC.md`` §4: an outcome-facing run starts only when ``PYTHONHASHSEED=0``, when
every file of the evaluator's code closure is tracked and identical to ``HEAD``, when ``HEAD`` is contained
in the *live* remote head of its upstream branch (``git ls-remote``, not only the cached tracking ref), and
when the output root is fresh.

The code closure is computed, not declared: the transitive set of local modules (``research.*``, ``src.*``)
imported by the entry modules, found by parsing their source, plus any listed documents and the lock files.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGES = ('research', 'src')
LOCK_FILES = ('pyproject.toml', 'uv.lock')


class ExecutionRefused(RuntimeError):
    """A pre-run refusal: nothing has been read from the outcome and nothing is written."""


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<f8').tobytes()).hexdigest()


def json_text(payload) -> str:
    return json.dumps(payload, indent=2, allow_nan=False, default=_json_default) + '\n'


def _json_default(value):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


def write_json(payload, path) -> None:
    Path(path).write_text(json_text(payload), encoding='utf-8')


def write_csv(frame: pd.DataFrame, path) -> None:
    """CSV with round-trip float text (``repr``), ``\\n`` line endings and no index."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
    Path(path).write_text(buffer.getvalue(), encoding='utf-8')


def require_hash_seed() -> None:
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ExecutionRefused('PYTHONHASHSEED=0 is required for byte-reproducible scoring')


def require_fresh(out_dir) -> Path:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ExecutionRefused(f'{out_dir} already exists; a run never overwrites a prior run')
    return out_dir


def _module_file(name: str) -> Path | None:
    parts = name.split('.')
    if parts[0] not in LOCAL_PACKAGES:
        return None
    base = ROOT.joinpath(*parts)
    if base.with_suffix('.py').is_file():
        return base.with_suffix('.py')
    if (base / '__init__.py').is_file():
        return base / '__init__.py'
    return None


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f'{node.module}.{alias.name}' for alias in node.names)
    return names


def import_closure(entry_files) -> list[str]:
    """Repository-relative paths of the entry files and every local module they import, transitively."""
    pending = [Path(ROOT / f).resolve() for f in entry_files]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in sorted(_imports(path)):
            parts = name.split('.')
            for k in range(1, len(parts) + 1):
                target = _module_file('.'.join(parts[:k]))
                if target is not None and target.resolve() not in seen:
                    pending.append(target.resolve())
    return sorted(str(p.relative_to(ROOT)) for p in seen)


def _git(*args, check=True) -> subprocess.CompletedProcess:
    done = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
    if check and done.returncode != 0:
        raise ExecutionRefused(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done


def live_remote_head() -> dict:
    """The upstream remote and branch of ``HEAD`` and the head commit the remote reports now."""
    upstream = _git('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{upstream}').stdout.strip()
    remote, _, branch = upstream.partition('/')
    if not remote or not branch:
        raise ExecutionRefused(f'HEAD has no usable upstream branch ({upstream!r})')
    listing = _git('ls-remote', remote, f'refs/heads/{branch}').stdout.split()
    if len(listing) != 2:
        raise ExecutionRefused(f'the live remote does not report exactly one head for {upstream}')
    return {'upstream': upstream, 'remote': remote, 'branch': branch, 'live_commit': listing[0]}


def frozen_code_state(files) -> dict:
    """Fail closed unless ``files`` are tracked, identical to ``HEAD`` and contained in the live remote head."""
    files = sorted(set(files))
    head = _git('rev-parse', 'HEAD').stdout.strip()
    missing = [p for p in files if not (ROOT / p).exists()]
    if missing:
        raise ExecutionRefused(f'code-closure files are missing: {missing}')
    untracked = [p for p in files if _git('ls-files', '--error-unmatch', p, check=False).returncode != 0]
    if untracked:
        raise ExecutionRefused(f'untracked files in the code closure: {untracked}')
    if _git('diff', '--quiet', 'HEAD', '--', *files, check=False).returncode != 0:
        changed = _git('diff', '--name-only', 'HEAD', '--', *files).stdout.split()
        raise ExecutionRefused(f'the code closure differs from HEAD; commit before running: {changed}')
    live = live_remote_head()
    if live['live_commit'] != head:
        if _git('cat-file', '-e', f"{live['live_commit']}^{{commit}}", check=False).returncode != 0:
            raise ExecutionRefused('the live remote head is not available locally; fetch before running')
        if _git('merge-base', '--is-ancestor', head, live['live_commit'], check=False).returncode != 0:
            raise ExecutionRefused(f"HEAD {head} is not contained in the live remote head "
                                   f"{live['live_commit']}; push before running")
    return {'commit': head, 'upstream': live['upstream'], 'live_remote_commit': live['live_commit'],
            'code_closure_clean_and_pushed': True,
            'code_closure_sha256': {p: sha256(ROOT / p) for p in files}}


def tracked_and_pushed(paths) -> dict:
    """Result artifacts a later stage depends on must be committed, unmodified and in the live remote."""
    return frozen_code_state(paths)


def verify_manifest(directory, manifest_name, key='artifact_sha256') -> dict:
    """Load a result manifest and check every listed artifact digest; returns the manifest."""
    directory = Path(directory)
    manifest = json.loads((directory / manifest_name).read_text(encoding='utf-8'))
    for name, digest in manifest[key].items():
        actual = sha256(directory / name)
        if actual != digest:
            raise ExecutionRefused(f'{directory / name}: sha256 {actual} differs from its manifest {digest}')
    return manifest

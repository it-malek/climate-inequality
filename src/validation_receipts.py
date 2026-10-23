"""Immutable, source-capturing receipts for explicitly configured prospective runs."""
from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

from src.correction_identity import CORRECTION_SET_ID


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + '\n')


class ValidationReceipts:
    """Exclusive stages confined to one named correction-namespace run."""

    def __init__(self, root, run, raw, spec_hash):
        self.root = Path(root).resolve(strict=True)
        namespace = self.root / 'outputs' / 'prospective' / CORRECTION_SET_ID
        self.run = Path(run).absolute()
        resolved = self.run.resolve()
        if not resolved.is_relative_to(namespace) or resolved == namespace:
            raise ValueError('run must be a named directory in the prospective namespace')
        if len(resolved.relative_to(namespace).parts) != 1 or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', resolved.name):
            raise ValueError('run must be a simple direct child of the prospective namespace')
        if resolved != self.run or any(p.is_symlink() for p in [self.run, *self.run.parents] if p != self.root.parent):
            raise ValueError('run path must not use symlinks')
        self.raw = Path(raw).resolve(strict=True)
        if not self.raw.is_dir():
            raise ValueError('raw root must be an existing read-only input directory')
        self.spec_hash = spec_hash

    def _stage_path(self, name):
        name = Path(name)
        if name.is_absolute() or not name.parts or any(part in ('.', '..') for part in name.parts):
            raise ValueError('stage name must be a relative confined path')
        path = self.run / name
        if path.resolve() != path or not path.resolve().is_relative_to(self.run):
            raise ValueError('stage path escapes run or uses symlinks')
        return path

    def verified_stage(self, name):
        path = self._stage_path(name)
        record = json.loads((path / 'completed.json').read_text())
        if record.get('status') != 'completed':
            raise ValueError('dependency is not completed')
        for relative, digest in record['outputs'].items():
            output = path / relative
            if not output.resolve().is_relative_to(path) or output.is_symlink() or sha(output) != digest:
                raise ValueError('Checkpoint drift: ' + str(output))
        return path

    def stage(self, name, inputs, action, *, runner, dependencies=(), sources=()):
        if sha(self.run / 'VALIDATION_SPEC.json') != self.spec_hash:
            raise ValueError('frozen validation specification drift')
        for dependency in dependencies:
            self.verified_stage(dependency)
        directory = self._stage_path(name)
        directory.mkdir(parents=True, exist_ok=False)
        identities = {str(Path(path).resolve()): sha(path) for path in inputs}
        code = {Path(runner).resolve(), Path(__file__).resolve(), self.root / 'uv.lock', self.root / 'pyproject.toml', *[Path(p).resolve() for p in sources]}
        for module in list(sys.modules.values()):
            file = getattr(module, '__file__', None)
            if file:
                path = Path(file).resolve()
                if path.is_relative_to(self.root) and '.venv' not in path.parts and path.suffix == '.py':
                    code.add(path)
        archive = directory / 'execution-source'
        archive.mkdir()
        code_hash = {}
        for i, path in enumerate(sorted(code)):
            if not path.is_file():
                raise ValueError('Missing execution source: ' + str(path))
            relative = path.relative_to(self.root) if path.is_relative_to(self.root) else Path('external') / f'{i}-{path.name}'
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            code_hash[str(path)] = sha(destination)
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=self.root, text=True, capture_output=True)
        packages = {}
        for package in ['numpy', 'scipy', 'pandas', 'statsmodels', 'duckdb', 'pykrige', 'xarray']:
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
        record = {'stage': str(name), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'spec_sha256': self.spec_hash, 'runner': str(Path(runner).resolve()), 'code_sha256': code_hash, 'input_sha256': identities, 'dependencies': list(dependencies), 'git_HEAD': revision.stdout.strip() if revision.returncode == 0 else None, 'python': sys.version, 'platform': platform.platform(), 'threads': {key: os.environ.get(key) for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']}, 'packages': packages}
        dump(directory / 'started.json', record)
        start = time.monotonic()
        print('START', name, flush=True)
        try:
            details = action(directory)
            for path, digest in identities.items():
                if sha(path) != digest:
                    raise ValueError('input changed: ' + path)
            for path, digest in code_hash.items():
                if sha(path) != digest:
                    raise ValueError('executed source changed during stage: ' + path)
            outputs = {}
            for path in sorted(directory.rglob('*')):
                if path.is_symlink():
                    raise ValueError('stage outputs must not contain symlinks')
                if path.is_file():
                    outputs[str(path.relative_to(directory))] = sha(path)
            dump(directory / 'completed.json', {'status': 'completed', 'seconds': time.monotonic() - start, 'details': details, 'outputs': outputs})
            print('DONE', name, round(time.monotonic() - start, 2), flush=True)
        except BaseException as exc:
            dump(directory / 'failed.json', {'status': 'failed', 'seconds': time.monotonic() - start, 'exception': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
            raise
        return directory

"""Explicit, confined output execution for the prospective correction set.

Pure numerical functions and historical readers remain usable without this
context. Writing entrypoints require it before reading inputs or computing.
No environment variable silently opts a process into prospective execution.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from pathlib import Path
import tempfile

from src.correction_identity import CORRECTION_SET_ID
from src.data_io import OUTPUTS_DIR

PROSPECTIVE_OUTPUT_ROOT = OUTPUTS_DIR / "prospective" / CORRECTION_SET_ID
_ACTIVE: ContextVar[ProspectiveExecution | None] = ContextVar("prospective_execution", default=None)


class ProspectiveExecutionRequired(RuntimeError):
    """A writer attempted to run without an explicit, confined execution scope."""


@dataclass(frozen=True)
class ProspectiveExecution:
    root: Path
    synthetic: bool = False

    def destination(self, path: Path | str) -> Path:
        path = Path(path).absolute()
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ProspectiveExecutionRequired(f"Output escapes prospective root {self.root}: {path}")
        # No symlink aliases (even within the namespace) or existing hard links.
        # This also validates the directory tree for builders with implicit children.
        for item in (path, *path.parents):
            if item == self.root.parent:
                break
            if item.is_symlink():
                raise ProspectiveExecutionRequired(f"Symlink output is forbidden: {item}")
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_symlink() or (child.is_file() and child.stat().st_nlink > 1):
                    raise ProspectiveExecutionRequired(f"Linked output is forbidden: {child}")
        elif path.exists() and path.stat().st_nlink > 1:
            raise ProspectiveExecutionRequired(f"Hard-linked output is forbidden: {path}")
        return path

    def path(self, relative: str) -> Path:
        return self.destination(self.root / relative)


def require_existing_inputs(*paths: Path) -> None:
    """Prospective builders consume prepared inputs; they never acquire/cache them."""
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Prepared read-only inputs required; no automatic acquisition: " + ", ".join(missing))


def require_prospective() -> ProspectiveExecution:
    execution = _ACTIVE.get()
    if execution is None:
        raise ProspectiveExecutionRequired(
            "Explicit prospective execution required; use --prospective or prospective_execution(). "
            "Historical outputs and the published bundle are read-only."
        )
    # Revalidate the root if a caller replaced a directory after entering scope.
    if execution.root.resolve() != execution.root or execution.root.is_symlink():
        raise ProspectiveExecutionRequired("Prospective root changed or became a symlink")
    return execution


@contextmanager
def prospective_execution(*, scratch: Path | None = None):
    """Opt in to the configured namespace, or an OS-temp synthetic test scope.

    Scratch is deliberately limited to temporary directories. It is not a way
    to redirect production output into another checkout or an archived bundle.
    The caller must still supply synthetic inputs; this API cannot infer data provenance.
    """
    if scratch is None:
        root = PROSPECTIVE_OUTPUT_ROOT.absolute()
        if root.resolve() != root:
            raise ProspectiveExecutionRequired("Configured prospective namespace contains a symlink")
    else:
        root = Path(scratch).absolute()
        allowed = (Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve())
        if root.resolve() != root or not any(root.is_relative_to(p) and root != p for p in allowed):
            raise ProspectiveExecutionRequired("Synthetic scratch must be an unaliased subdirectory of OS temp")
    execution = ProspectiveExecution(root, synthetic=scratch is not None)
    execution.destination(root)
    token = _ACTIVE.set(execution)
    try:
        yield execution
    finally:
        _ACTIVE.reset(token)


def prospective_outputs(**destinations: str):
    """Guard writer arguments and replace omitted legacy defaults inside a scope."""
    def decorate(function):
        sig = signature(function)

        @wraps(function)
        def guarded(*args, **kwargs):
            execution = require_prospective()
            bound = sig.bind(*args, **kwargs)
            for argument, relative in destinations.items():
                supplied = bound.arguments.get(argument)
                bound.arguments[argument] = execution.destination(supplied) if supplied is not None else execution.path(relative)
            return function(*bound.args, **bound.kwargs)
        return guarded
    return decorate


def require_synthetic_evaluator() -> ProspectiveExecution:
    execution = require_prospective()
    if not execution.synthetic:
        raise ProspectiveExecutionRequired(
            "Historical evaluator is not adapted for prospective scientific execution; "
            "approve its input/output and unavailable-diagnostic contract separately"
        )
    return execution


def unadapted_evaluator(*, output_argument: str | None = None):
    """Keep historical scoring writers closed until their prospective contract is approved.

    Synthetic injected-input tests may exercise a confined writer. A normal
    prospective execution is refused even after this branch has been committed.
    """
    def decorate(function):
        sig = signature(function)

        @wraps(function)
        def guarded(*args, **kwargs):
            execution = require_prospective()
            if not execution.synthetic or output_argument is None:
                raise ProspectiveExecutionRequired(
                    f"{function.__module__}.{function.__name__} is a historical evaluator; "
                    "prospective input/output and unavailable-diagnostic adaptations need a separate decision"
                )
            bound = sig.bind(*args, **kwargs)
            execution.destination(bound.arguments[output_argument])
            return function(*bound.args, **bound.kwargs)
        return guarded
    return decorate


@contextmanager
def prospective_cli(parser: argparse.ArgumentParser | None = None):
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prospective", action="store_true", required=True,
                        help="Explicitly execute into the correction-set output namespace")
    args = parser.parse_args()
    with prospective_execution() as execution:
        yield args, execution

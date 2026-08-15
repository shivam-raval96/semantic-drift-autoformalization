#!/usr/bin/env python3
"""Run directories: provenance, resume-safe records, and the shared CLI.

A run directory is written once and never edited afterwards:

    <dated folder>/runs/<experiment name>/
        run_meta.json              config and provenance
        records/<condition>.jsonl  one row per graded example
        summary.json               the headline numbers

Two properties matter more than the layout. First, a condition's records are
written under a temporary name and published only once that condition
finishes, so a crashed or disconnected run can never leave a short file that
looks complete and then gets treated as cached. Second, every run records what
produced it: model, seed, all parameters, and the versions of both this
repository and the vendored checkout.

Sweeps are declared as data — a list of named Conditions — rather than as
nested loops, so each cell has a name, its settings land in its records, and a
rerun skips whatever already finished.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .vendor import REPO_ROOT, VENDOR_DIR


class Condition:
    """One cell of a sweep: a name and the settings that define it.

    The name is used as a filename, so it has to be filesystem-safe and unique
    within a run; the settings are merged into every record the cell produces.
    """

    __slots__ = ("name", "settings")

    def __init__(self, name: str, **settings: Any):
        if not name or any(c in name for c in "/\\ \t\n"):
            raise ValueError("condition name {!r} is not filename-safe".format(name))
        self.name = name
        self.settings = settings

    def __getitem__(self, key: str) -> Any:
        return self.settings[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def as_dict(self) -> dict:
        return dict(self.settings, name=self.name)

    def __repr__(self) -> str:
        pairs = ", ".join(
            "{}={!r}".format(k, v) for k, v in sorted(self.settings.items())
        )
        return "Condition({!r}, {})".format(self.name, pairs)


class RunDirectory:
    """Everything one experiment run wrote."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.records_dir = self.path / "records"

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def open(
        cls,
        out_dir: Path,
        name: str,
        meta: Optional[dict] = None,
        force: bool = False,
    ) -> "RunDirectory":
        """Open (creating if needed) the run directory for `name`.

        On a first run the provenance file is written. On a resume it is read
        back and checked: if the model, the seed or any parameter has changed
        since the run started, the records already on disk were produced under
        different conditions and mixing them would be silent corruption, so
        this refuses unless `force` is set.
        """
        run = cls(Path(out_dir) / "runs" / name)
        run.records_dir.mkdir(parents=True, exist_ok=True)

        meta_path = run.path / "run_meta.json"
        fresh = dict(meta or {})
        fresh.update(provenance())
        fresh.setdefault("experiment", name)

        if meta_path.exists() and not force:
            previous = json.loads(meta_path.read_text())
            changed = _differences(previous, fresh)
            if changed:
                raise SystemExit(
                    "{} was started with different settings: {}\n"
                    "Pass --force to overwrite it, or use a different "
                    "--out-dir.".format(meta_path, "; ".join(changed))
                )
        else:
            meta_path.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n")
        return run

    # --------------------------------------------------------------- records

    def records_path(self, condition: str) -> Path:
        return self.records_dir / "{}.jsonl".format(condition)

    def has(self, condition: str) -> bool:
        """True once a condition's records have been published."""
        return self.records_path(condition).exists()

    def read(self, condition: str) -> List[dict]:
        path = self.records_path(condition)
        if not path.exists():
            raise FileNotFoundError(path)
        return [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    def read_all(self, conditions: Optional[Iterable[str]] = None) -> List[dict]:
        """Every published record, or those of the named conditions."""
        if conditions is None:
            names = sorted(p.stem for p in self.records_dir.glob("*.jsonl"))
        else:
            names = list(conditions)
        rows: List[dict] = []
        for name in names:
            rows.extend(self.read(name))
        return rows

    @contextmanager
    def writing(self, condition: str) -> Iterator["RecordWriter"]:
        """Write a condition's records, publishing only on clean completion.

        Rows land in a temporary file first. If the body raises — or the
        process dies — the temporary file is left behind for inspection and the
        real path stays absent, so `has()` keeps reporting the condition as
        unfinished and a rerun redoes it.
        """
        final = self.records_path(condition)
        partial = self.records_dir / ".{}.jsonl.partial".format(condition)
        handle = partial.open("w", encoding="utf-8")
        writer = RecordWriter(handle)
        try:
            yield writer
        except BaseException:
            handle.close()
            raise
        handle.close()
        os.replace(str(partial), str(final))

    def drop(self, condition: str) -> None:
        """Remove a condition's records so it will be run again."""
        for path in (
            self.records_path(condition),
            self.records_dir / ".{}.jsonl.partial".format(condition),
        ):
            if path.exists():
                path.unlink()

    # --------------------------------------------------------------- summary

    def write_summary(self, summary: dict) -> Path:
        path = self.path / "summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n")
        return path

    def read_summary(self) -> dict:
        return json.loads((self.path / "summary.json").read_text())

    def read_meta(self) -> dict:
        return json.loads((self.path / "run_meta.json").read_text())

    def figure_path(self, filename: str) -> Path:
        """A path under the dated folder's figures/ directory.

        Figures live beside the run rather than inside it, because a run
        directory is never edited after the fact and plots get redrawn.
        """
        figures = self.path.parent.parent / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        return figures / filename

    def __repr__(self) -> str:
        return "RunDirectory({!r})".format(str(self.path))


class RecordWriter:
    """Appends records to an open file, one JSON object per line."""

    __slots__ = ("_handle", "count")

    def __init__(self, handle):
        self._handle = handle
        self.count = 0

    def write(self, record: dict) -> None:
        self._handle.write(
            json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
        )
        self.count += 1

    def extend(self, records: Iterable[dict]) -> None:
        for record in records:
            self.write(record)


def _json_default(value):
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "tolist"):  # numpy scalars and arrays
        return value.tolist()
    raise TypeError("cannot serialize {!r}".format(type(value).__name__))


# ------------------------------------------------------------------ provenance

# Fields that describe the machine rather than the experiment. A resume on a
# different box is fine; a resume with a different seed is not.
_INCIDENTAL = frozenset({"created", "hostname", "platform", "python", "argv", "torch", "transformers"})


def provenance() -> dict:
    """What produced this run: code versions, interpreter, and machine."""
    return {
        "created": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "repo_commit": _git("rev-parse", "HEAD"),
        "repo_dirty": bool(_git("status", "--porcelain")),
        # The checkout is vendored as plain files rather than a submodule, so
        # its version is the git tree hash of that directory: it changes if and
        # only if something inside it changes.
        "informalizing_etp_tree": _git("rev-parse", "HEAD:informalizing-etp"),
        "informalizing_etp_path": str(VENDOR_DIR),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "argv": list(sys.argv),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
    }


def _git(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ("git",) + args,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.decode("utf-8", "replace").strip()


def _version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _differences(previous: dict, current: dict) -> List[str]:
    """Settings that changed between a run's start and this invocation."""
    changed = []
    for key in sorted(set(previous) | set(current)):
        if key in _INCIDENTAL:
            continue
        if previous.get(key) != current.get(key):
            changed.append(
                "{} was {!r}, now {!r}".format(key, previous.get(key), current.get(key))
            )
    return changed


# ------------------------------------------------------------------------ CLI


def base_parser(description: str) -> argparse.ArgumentParser:
    """The arguments every experiment takes.

    Nothing an experiment does should require editing a constant in its source:
    the model, the seed, the destination and the run size are all arguments, so
    a rerun is reproducible from the command line recorded in run_meta.json.
    """
    cli = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    cli.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where runs/ and figures/ go (default: the experiment file's own "
        "dated folder)",
    )
    cli.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model to run (default: %(default)s)",
    )
    cli.add_argument(
        "--seed",
        type=int,
        default=0,
        help="fixed at %(default)s unless the experiment is about sampling "
        "variance",
    )
    cli.add_argument(
        "--quick",
        action="store_true",
        help="tiny smoke run that exercises every step end to end, to check "
        "the plumbing before committing to a real run; writes to a separate "
        "run directory so it cannot overwrite results",
    )
    cli.add_argument(
        "--analyze-only",
        action="store_true",
        help="rebuild tables and figures from the existing run directory, "
        "with no model and no GPU",
    )
    cli.add_argument(
        "--force",
        action="store_true",
        help="rerun conditions that already have records, and overwrite "
        "provenance that no longer matches",
    )
    return cli


DEFAULT_MODEL = "Qwen/Qwen3-4B"


def resolve_out_dir(args: argparse.Namespace, experiment_file: str) -> Path:
    """Where this invocation should write.

    Defaults to the dated folder holding the experiment file. A smoke run gets
    its own subdirectory, because its eight-row conditions share the real
    conditions' names and would otherwise be picked up as finished work.
    """
    out_dir = args.out_dir or Path(experiment_file).resolve().parent
    if getattr(args, "quick", False):
        out_dir = out_dir / "quick"
    return out_dir


def pending(
    run: RunDirectory,
    conditions: Sequence[Condition],
    force: bool = False,
) -> List[Condition]:
    """The conditions still to run, dropping records first when forced."""
    todo = []
    for condition in conditions:
        if force:
            run.drop(condition.name)
        if not run.has(condition.name):
            todo.append(condition)
    return todo


def describe(conditions: Sequence[Condition], run: RunDirectory) -> str:
    """A short table of the sweep and what is already finished."""
    lines = ["{} conditions in {}".format(len(conditions), run.path)]
    for condition in conditions:
        state = "done" if run.has(condition.name) else "pending"
        settings = " ".join(
            "{}={}".format(k, v) for k, v in sorted(condition.settings.items())
        )
        lines.append("  [{:>7}] {:<34} {}".format(state, condition.name, settings))
    return "\n".join(lines)

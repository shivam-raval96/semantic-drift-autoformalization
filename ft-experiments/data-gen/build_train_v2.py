#!/usr/bin/env python3
"""Build train_v2: the story->grammar task corpus for FT v2.

The SAME equation pairs as train_v1 (v2 changes the surface and the
training objective, not the data distribution): each v1 row's pair is
re-rendered as a themed story with the repo renderer, paired with the
grammar-A reference completion — byte-identical to v1's `text` field.
Rows carry no prompt text; the prompt is built at train time from the
pinned template.

Theme holdout (DESIGN.md): of storyform's themes sorted alphabetically,
the LAST is held out of training entirely; each pair gets one of the
remaining themes by sha256 of its pair-class-key string mod 3 — no RNG.
The held-out theme's slice of eval_v1 measures input-side generalization.

Render failures (ParseError / palette exhaustion) drop the pair with a
per-tier count — no quota top-up, keeping exact pair-set correspondence
with train_v1 minus drops. Every emitted row is verified: completion
round-trips through the grader against its own pair, the story
back-parses to the source laws, and the canonical fields match the
parsed ASTs. Disjointness gates (SAIR pairs, eval_v1 laws) are
re-asserted from sair_index, plus paired provenance against train_v1.

Sources train_v1/{train,holdout}.jsonl locally when present (verified
against the committed manifest shas); otherwise regenerates them with
build_train.py's own machinery and refuses to proceed on any sha
mismatch. Deterministic: rerunning yields byte-identical files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent            # ft-experiments/v2/
sys.path.insert(0, str(HERE.parent / "data-gen"))

from ftlib import (  # noqa: E402  (also puts the repo on sys.path)
    Term,
    canonical,
    file_sha256,
    ftc,
    pair_class_key,
    pair_hash,
    rg_text,
    serialize_infix,
    verify_rg_round_trip,
)

from backparse import backparse  # noqa: E402  (repo modules via ftlib path)
from checkform import parse_prefix_equation  # noqa: E402
from storyform import THEME_ORDER, render_story  # noqa: E402

TRAIN_V1 = ftc.PATHS["train_v1"]
V1_FILES = ("train.jsonl", "holdout.jsonl", "synthetic-laws.txt")

ALL_THEMES = tuple(sorted(THEME_ORDER))
THEME_HOLDOUT = ALL_THEMES[-1]
TRAIN_THEMES = ALL_THEMES[:-1]

# v1 fields copied into every v2 row unchanged (paired provenance).
PROVENANCE_FIELDS = (
    "e_label", "f_label", "ops_e", "ops_f", "ops_total", "max_depth",
    "tier", "source", "pair_hash", "law_hash_e", "law_hash_f", "split",
)

STORY_PREVIEW_CHARS = 160


class RenderDrop(Exception):
    """A per-pair story-render failure; the pair is dropped and counted.

    Only render_story raises this. Verification failures (round-trip,
    back-parse, canonical mismatch) raise AssertionError and abort."""


def load_v1(scratch_root: Path) -> Tuple[Dict[str, List[dict]], dict, Path]:
    """train_v1 rows keyed by filename, sha-verified against its manifest.

    Regenerates into a scratch dir with build_train's own main() when the
    gitignored bulk JSONL is absent locally.
    """
    v1_manifest = json.loads((TRAIN_V1 / "manifest.json").read_text())
    src_dir = TRAIN_V1
    if not all((TRAIN_V1 / name).exists() for name in V1_FILES):
        import build_train  # heavy (genform corpus): only on the rebuild path

        src_dir = Path(tempfile.mkdtemp(prefix="train_v1_rebuild_", dir=scratch_root))
        print(f"train_v1 JSONL absent; regenerating into {src_dir}")
        build_train.main(["--out-dir", str(src_dir)])
    for name in V1_FILES:
        got, want = file_sha256(src_dir / name), v1_manifest["files"][name]
        if got != want:
            raise AssertionError(
                f"train_v1 {name} sha256 mismatch vs manifest: {got} != {want}"
            )
    rows = {
        name: [json.loads(line) for line in (src_dir / name).read_text().splitlines()]
        for name in V1_FILES if name.endswith(".jsonl")
    }
    return rows, v1_manifest, src_dir


def assign_theme(e_pair: Tuple[Term, Term], f_pair: Tuple[Term, Term]) -> str:
    """Deterministic training-theme choice: hash of the pair-class key."""
    ck = pair_class_key(e_pair, f_pair)
    digest = hashlib.sha256(f"{ck[0]} => {ck[1]}".encode("utf-8")).hexdigest()
    return TRAIN_THEMES[int(digest, 16) % len(TRAIN_THEMES)]


def parse_v1_text(text: str) -> Tuple[Tuple[Term, Term], Tuple[Term, Term]]:
    e_line, f_line = text.split("\n")
    if not e_line.startswith("ASSUME: ") or not f_line.startswith("ASK: "):
        raise AssertionError(f"malformed train_v1 text: {text!r}")
    e_pair = parse_prefix_equation(e_line[len("ASSUME: "):])
    f_pair = parse_prefix_equation(f_line[len("ASK: "):])
    return e_pair, f_pair


def make_row(v1_row: dict) -> dict:
    """One v2 row from one v1 row; RenderDrop on story-render failure,
    AssertionError (fatal) on any integrity mismatch."""
    e_pair, f_pair = parse_v1_text(v1_row["text"])
    completion = rg_text(e_pair, f_pair)
    if completion != v1_row["text"]:
        raise AssertionError(f"completion drifted from v1 text: {completion!r}")
    if pair_hash(e_pair, f_pair) != v1_row["pair_hash"]:
        raise AssertionError(f"pair_hash drifted from v1: {v1_row['pair_hash']}")
    verify_rg_round_trip(completion, e_pair, f_pair)

    theme = assign_theme(e_pair, f_pair)
    try:
        story, meta = render_story(
            serialize_infix(*e_pair), serialize_infix(*f_pair), theme_key=theme
        )
    except (ValueError, IndexError) as exc:
        # ParseError/ThemeError are ValueError; palette exhaustion
        # surfaces as IndexError.
        raise RenderDrop(f"{type(exc).__name__}: {exc}") from exc

    canonical_e, canonical_f = canonical(*e_pair), canonical(*f_pair)
    if meta["canonical_e"] != canonical_e or meta["canonical_f"] != canonical_f:
        raise AssertionError(f"renderer canonical mismatch for {completion!r}")
    try:
        got = backparse(story)
    except ValueError as exc:
        raise AssertionError(f"story back-parse failed: {exc}") from exc
    if canonical(*got["habit_law"]) != canonical_e or canonical(
        *got["question_law"]
    ) != canonical_f:
        raise AssertionError("story back-parse does not recover the source laws")

    row = {
        "story": story,
        "completion": completion,
        "canonical_e": canonical_e,
        "canonical_f": canonical_f,
        "theme": theme,
    }
    row.update({k: v1_row[k] for k in PROVENANCE_FIELDS})
    return row


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="Generate the story->grammar v2 corpus.")
    cli.add_argument("--out-dir", type=Path, default=ftc.PATHS["train_v2"])
    args = cli.parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_rows, v1_manifest, v1_dir = load_v1(out_dir.parent)
    print(f"train_v1 verified against manifest ({v1_dir}): "
          + ", ".join(f"{k} n={len(v)}" for k, v in v1_rows.items()))
    print(f"themes: {ALL_THEMES}; held out: {THEME_HOLDOUT!r}; "
          f"training: {TRAIN_THEMES}")

    sair_index = json.loads(Path(ftc.PATHS["sair_index"]).read_text())
    eval_laws = set(sair_index["eval_law_hashes"])
    sair_pairs = set(sair_index["pair_hashes"])

    tiers_in_order = list(dict.fromkeys(
        r["tier"] for rows in v1_rows.values() for r in rows
    ))
    drops = {tier: 0 for tier in tiers_in_order}
    out_rows: Dict[str, List[dict]] = {}
    for name, rows in v1_rows.items():
        kept: List[dict] = []
        for v1_row in rows:
            try:
                kept.append(make_row(v1_row))
            except RenderDrop as exc:
                # Drop and count; never top up (exact pair-set
                # correspondence with train_v1 minus drops).
                drops[v1_row["tier"]] += 1
                print(f"render drop [{v1_row['tier']}] "
                      f"{v1_row['pair_hash']}: {exc}")
        out_rows[name] = kept

    all_rows = [row for rows in out_rows.values() for row in rows]

    # Hard gates, re-asserted for v2 independently of v1's own audit.
    v1_hashes = {
        name: {r["pair_hash"] for r in rows} for name, rows in v1_rows.items()
    }
    for name, rows in out_rows.items():
        for row in rows:
            assert row["pair_hash"] not in sair_pairs, f"SAIR pair collision: {row}"
            assert row["law_hash_e"] not in eval_laws, f"eval_v1 law collision: {row}"
            assert row["law_hash_f"] not in eval_laws, f"eval_v1 law collision: {row}"
            assert row["pair_hash"] in v1_hashes[name], f"not a train_v1 pair: {row}"
    assert len({row["pair_hash"] for row in all_rows}) == len(all_rows)

    tier_stats = {
        tier: {
            "n": sum(1 for r in all_rows if r["tier"] == tier),
            "render_drops": drops[tier],
        }
        for tier in tiers_in_order
    }
    theme_counts = {
        theme: sum(1 for r in all_rows if r["theme"] == theme)
        for theme in TRAIN_THEMES
    }
    for tier, stats in tier_stats.items():
        print(f"{tier:8s} {stats['n']:5d} rows  render drops {stats['render_drops']}")
    print(f"theme distribution: {theme_counts}")
    for row in (all_rows[0], all_rows[len(all_rows) // 2], all_rows[-1]):
        preview = row["story"][:STORY_PREVIEW_CHARS].replace("\n", " ")
        print(f"sample [{row['tier']}/{row['theme']}] {row['e_label']}=>"
              f"{row['f_label']}  {row['completion']!r}\n  story: {preview}...")

    files = {}
    for name, rows in out_rows.items():
        path = out_dir / name
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[name] = file_sha256(path)
    synthetic_path = out_dir / "synthetic-laws.txt"
    synthetic_path.write_bytes((v1_dir / "synthetic-laws.txt").read_bytes())
    files["synthetic-laws.txt"] = file_sha256(synthetic_path)

    manifest = {
        "corpus_version": "train_v2",
        "generator": "ft-experiments/v2/build_train_v2.py",
        "repo_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE
        ).stdout.strip(),
        "source": "train_v1 pairs re-rendered",
        "train_v1_files_sha256": {n: v1_manifest["files"][n] for n in V1_FILES},
        "theme_holdout": THEME_HOLDOUT,
        "train_themes": list(TRAIN_THEMES),
        "theme_distribution": theme_counts,
        "disjointness_gate": {
            "pair_hashes_vs": "all 9 SAIR subsets (2,669 rows)",
            "law_hashes_vs": "eval_v1 evaluation subsets only (decision 2026-07-24)",
            "pair_provenance": "every pair_hash present in its train_v1 file",
        },
        "render_drops_per_tier": drops,
        "tiers": tier_stats,
        "n_train": len(out_rows["train.jsonl"]),
        "n_holdout": len(out_rows["holdout.jsonl"]),
        "files": files,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{manifest['n_train']} train + {manifest['n_holdout']} holdout rows "
          f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

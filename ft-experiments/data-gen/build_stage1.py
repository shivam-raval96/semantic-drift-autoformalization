#!/usr/bin/env python3
"""Build the stage-1 recognition corpus: opaque-label familiarity, no
translation and no grammar production.

Two task types over the three RGs (RG-1=a, RG-2=b_near, RG-3=b_far):

  identify  statement -> its label            (completion RG-1/RG-2/RG-3)
  validate  statement + label -> yes/no       (completion Yes/No)

Equation pairs are reused from train_v2 (already pair-disjoint from SAIR
and law-disjoint from eval_v1); we do NOT resample the pool. Pairs are
split into a train and a held-out eval pool by LAW CLASS (a pair joins a
pool only if BOTH its laws fall on that pool's side of a deterministic
hash threshold), so the eval measures recognition on laws never seen in
training. Serialization and parsing come from eval/grammars.py; every row
is verified as it is built (identification statements parse under their
own grammar and no other; validation negatives fail the asked parser).

Deterministic: one seeded RNG in fixed order; rerunning is byte-identical.
Writes stage1/{train,eval}.jsonl + manifest.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import stage1lib as s1
from ftlib import ftc, file_sha256, short_hash

PATHS = ftc.PATHS
TRAIN_V2 = PATHS["train_v2"]

SEED = 0
EVAL_LAW_THRESHOLD = 3        # laws with int(hash,16) % 10 < 3 are held out (~30%)
NEG_TYPES = ("wrong_grammar", "malformed", "mismatched_labels")

# per-grammar quotas (x3 grammars); validate splits 50/50 pos/neg
QUOTAS = {
    "train": {"identify": 300, "validate_pos": 150, "validate_neg": 150},
    "eval": {"identify": 81, "validate_pos": 42, "validate_neg": 42},
}

PROVENANCE = ("pair_hash", "law_hash_e", "law_hash_f")


def load_pairs() -> List[dict]:
    """train_v2 rows (train + holdout), deduped by pair class, with the AST
    pair reconstructed from the grammar-A completion."""
    rows: List[dict] = []
    for name in ("train.jsonl", "holdout.jsonl"):
        for line in (TRAIN_V2 / name).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    seen, pairs = set(), []
    for r in rows:
        if r["pair_hash"] in seen:
            continue
        seen.add(r["pair_hash"])
        e, f = s1.pair_from_completion(r["completion"])
        pairs.append({
            "e": e, "f": f,
            "pair_hash": r["pair_hash"],
            "law_hash_e": r["law_hash_e"],
            "law_hash_f": r["law_hash_f"],
            "source_tier": r["tier"],
        })
    return pairs


def split_by_law(pairs: List[dict]) -> Tuple[List[dict], List[dict], set]:
    """Law-disjoint train/eval pools. A pair is eval-eligible iff BOTH laws
    are held out, train-eligible iff NEITHER is; straddlers are dropped so
    no law class appears on both sides."""
    laws = {p["law_hash_e"] for p in pairs} | {p["law_hash_f"] for p in pairs}
    eval_laws = {h for h in laws if int(h, 16) % 10 < EVAL_LAW_THRESHOLD}

    def side(p: dict) -> str:
        in_e = p["law_hash_e"] in eval_laws
        in_f = p["law_hash_f"] in eval_laws
        if in_e and in_f:
            return "eval"
        if not in_e and not in_f:
            return "train"
        return "drop"

    train_pool = [p for p in pairs if side(p) == "train"]
    eval_pool = [p for p in pairs if side(p) == "eval"]
    return train_pool, eval_pool, eval_laws


def provenance(pair: dict, split: str) -> dict:
    row = {k: pair[k] for k in PROVENANCE}
    row["source_tier"] = pair["source_tier"]
    row["split"] = split
    return row


def identify_row(pair: dict, grammar_key: str, split: str) -> dict:
    st = s1.serialize(grammar_key, pair["e"], pair["f"])
    assert s1.parses_under(grammar_key, st), (grammar_key, st)
    for other in s1.GRAMMAR_KEYS:
        if other != grammar_key:
            assert not s1.parses_under(other, st), ("id ambiguous", grammar_key, other, st)
    label = s1.GRAMMAR_TO_LABEL[grammar_key]
    return {
        "task": "identify",
        "prompt": s1.identify_prompt(st),
        "completion": label,
        "grammar": grammar_key,
        "family": s1.GRAMMAR_FAMILY[grammar_key],
        "label": label,
        "polarity": None,
        "corruption": None,
        **provenance(pair, split),
    }


def validate_pos_row(pair: dict, grammar_key: str, split: str) -> dict:
    st = s1.serialize(grammar_key, pair["e"], pair["f"])
    assert s1.parses_under(grammar_key, st), ("pos must parse", grammar_key, st)
    label = s1.GRAMMAR_TO_LABEL[grammar_key]
    return {
        "task": "validate",
        "prompt": s1.validate_prompt(label, st),
        "completion": "Yes",
        "grammar": grammar_key,
        "family": s1.GRAMMAR_FAMILY[grammar_key],
        "label": label,
        "polarity": "yes",
        "corruption": None,
        **provenance(pair, split),
    }


def validate_neg_row(pair: dict, grammar_key: str, corruption: str,
                     other_key: str, split: str) -> dict:
    e, f = pair["e"], pair["f"]
    if corruption == "wrong_grammar":
        st = s1.neg_wrong_grammar(other_key, e, f)
    elif corruption == "malformed":
        st = s1.neg_malformed(grammar_key, e, f)
    elif corruption == "mismatched_labels":
        st = s1.neg_mismatched_labels(grammar_key, other_key, e, f)
    else:
        raise ValueError(corruption)
    assert not s1.parses_under(grammar_key, st), ("neg must fail", corruption, grammar_key, st)
    label = s1.GRAMMAR_TO_LABEL[grammar_key]
    return {
        "task": "validate",
        "prompt": s1.validate_prompt(label, st),
        "completion": "No",
        "grammar": grammar_key,
        "family": s1.GRAMMAR_FAMILY[grammar_key],
        "label": label,
        "polarity": "no",
        "corruption": corruption,
        **provenance(pair, split),
    }


def other_key(grammar_key: str) -> str:
    keys = s1.GRAMMAR_KEYS
    return keys[(keys.index(grammar_key) + 1) % len(keys)]


def generate_split(pool: List[dict], split: str, rng: random.Random) -> List[dict]:
    quotas = QUOTAS[split]
    need = max(quotas.values())
    assert len(pool) >= need, f"{split} pool {len(pool)} < largest quota {need}"
    rows: List[dict] = []
    for grammar_key in s1.GRAMMAR_KEYS:
        for pair in rng.sample(pool, quotas["identify"]):
            rows.append(identify_row(pair, grammar_key, split))
        for pair in rng.sample(pool, quotas["validate_pos"]):
            rows.append(validate_pos_row(pair, grammar_key, split))
        for i, pair in enumerate(rng.sample(pool, quotas["validate_neg"])):
            corruption = NEG_TYPES[i % len(NEG_TYPES)]
            rows.append(validate_neg_row(pair, grammar_key, corruption,
                                         other_key(grammar_key), split))
    return rows


def counts(rows: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    out["by_task"] = _tally(rows, "task")
    out["by_label"] = _tally([r for r in rows if r["task"] == "identify"], "label")
    out["validate_by_polarity"] = _tally(
        [r for r in rows if r["task"] == "validate"], "polarity")
    out["validate_neg_by_corruption"] = _tally(
        [r for r in rows if r["corruption"]], "corruption")
    out["by_grammar"] = _tally(rows, "grammar")
    return out


def _tally(rows: List[dict], key: str) -> Dict[str, int]:
    tally: Dict[str, int] = {}
    for r in rows:
        tally[str(r[key])] = tally.get(str(r[key]), 0) + 1
    return tally


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description="Generate the stage-1 recognition corpus.")
    cli.add_argument("--out-dir", type=Path, default=PATHS["stage1"])
    args = cli.parse_args(argv)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs()
    train_pool, eval_pool, eval_laws = split_by_law(pairs)
    print(f"pairs {len(pairs)} -> train pool {len(train_pool)}, eval pool {len(eval_pool)}, "
          f"dropped straddlers {len(pairs) - len(train_pool) - len(eval_pool)}")

    # Hard gate: the two pools share no law class.
    train_laws = {p["law_hash_e"] for p in train_pool} | {p["law_hash_f"] for p in train_pool}
    eval_pool_laws = {p["law_hash_e"] for p in eval_pool} | {p["law_hash_f"] for p in eval_pool}
    assert not (train_laws & eval_pool_laws), "train/eval law overlap"
    train_pairhashes = {p["pair_hash"] for p in train_pool}
    eval_pairhashes = {p["pair_hash"] for p in eval_pool}
    assert not (train_pairhashes & eval_pairhashes), "train/eval pair overlap"

    rng = random.Random(SEED)
    split_rows = {
        "train": generate_split(train_pool, "train", rng),
        "eval": generate_split(eval_pool, "eval", rng),
    }

    files, split_stats = {}, {}
    for split, name in (("train", "train.jsonl"), ("eval", "eval.jsonl")):
        rows = split_rows[split]
        path = out_dir / name
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        files[name] = file_sha256(path)
        split_stats[split] = {"n": len(rows), **counts(rows)}
        print(f"{split:5s} {len(rows):5d} rows  {counts(rows)['by_task']}")

    train_v2_manifest = json.loads((TRAIN_V2 / "manifest.json").read_text())
    prompt_sha = short_hash(s1.IDENTIFY_PROMPT + "\x00" + s1.VALIDATE_PROMPT)
    manifest = {
        "corpus_version": "stage1",
        "generator": "ft-experiments/data-gen/build_stage1.py",
        "purpose": "opaque-label rigid-grammar familiarity (recognition only)",
        "task_types": ["identify", "validate"],
        "labels": s1.RG_LABELS,
        "grammar_family": s1.GRAMMAR_FAMILY,
        "prompt_sha16": prompt_sha,
        "source": "train_v2 pairs (train+holdout), deduped by pair class",
        "train_v2_manifest_sha256": file_sha256(TRAIN_V2 / "manifest.json"),
        "train_v2_files_sha256": train_v2_manifest["files"],
        "seed": SEED,
        "eval_law_threshold_mod10": EVAL_LAW_THRESHOLD,
        "quotas_per_grammar": QUOTAS,
        "neg_types": list(NEG_TYPES),
        "pools": {
            "unique_pairs": len(pairs),
            "train_pool": len(train_pool),
            "eval_pool": len(eval_pool),
            "held_out_law_classes": len(eval_laws),
        },
        "disjointness": {
            "train_eval_law_overlap": 0,
            "train_eval_pair_overlap": 0,
        },
        "splits": split_stats,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest -> {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

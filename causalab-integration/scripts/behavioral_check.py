#!/usr/bin/env python3
"""V3 behavioral pre-check: are models above chance on the implication task?

Runs the etp_implication prompts (all 14 True items + a matched sample of
False items, both default registers) against OpenRouter models at
temperature 0 and reports accuracy, response bias, and per-register
accuracy. If models sit at chance, verdict-side interp analyses
(subspace/manifold on `implication`) are not worth GPU time yet; the
law-identity analyses are unaffected.

Usage:
    export OPENROUTER_API_KEY=...
    python3 behavioral_check.py --causalab /path/to/causalab \
        [--models openai/gpt-4o-mini anthropic/claude-haiku-4.5] [--n-per-level 20]
    python3 behavioral_check.py --causalab /path/to/causalab --dry-run
"""

import argparse
import json
import os
import random
import sys
import ssl
import urllib.request

URL = "https://openrouter.ai/api/v1/chat/completions"

def _ssl_context():
    """macOS framework Python ships without root certs; require certifi there."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
        if not ctx.cert_store_stats().get("x509_ca", 0):
            raise SystemExit(
                "No CA certificates available in this Python. Fix with ONE of:\n"
                "  python3 -m pip install certifi     (same interpreter as this script)\n"
                "  open '/Applications/Python 3.12/Install Certificates.command'"
            )
        return ctx


def build_items(n_per_level: int, seed: int):
    from causalab.tasks.etp_implication.config import (
        CERTIFIED_FALSE,
        CERTIFIED_TRUE,
        DEFAULT_REGISTERS,
        ops_bin,
        registers_for,
    )
    from causalab.tasks.etp_implication.templates import fill_template

    rng = random.Random(seed)
    by_label = {True: [], False: []}
    for label, pool in ((True, list(CERTIFIED_TRUE)), (False, list(CERTIFIED_FALSE))):
        for p, c in pool:
            for r in DEFAULT_REGISTERS:
                if r in registers_for(p) and r in registers_for(c):
                    by_label[label].append(
                        {"p": p, "c": c, "register": r, "label": label,
                         "bin": ops_bin(p, c),
                         "prompt": fill_template(r, p, c)}
                    )
    # stratified: balanced per (complexity bin, label) cell
    cells: dict = {}
    for label in (True, False):
        for it in by_label[label]:
            cells.setdefault((it["bin"], label), []).append(it)
    bins = sorted({b for b, _ in cells})
    per_cell = max(1, n_per_level // 2)
    items = []
    for b in bins:
        k = min(per_cell, len(cells.get((b, True), [])), len(cells.get((b, False), [])))
        items += rng.sample(cells[(b, True)], k) + rng.sample(cells[(b, False)], k)
    rng.shuffle(items)
    return items


SYSTEM = ("Think step by step if needed, then END your reply with a final "
          "line containing exactly one word: True or False.")


def _call(model: str, messages: list, api_key: str, max_tokens: int) -> str:
    payload = {"model": model, "messages": messages,
               "temperature": 0, "max_tokens": max_tokens}
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180, context=_ssl_context()) as r:
        out = json.load(r)
    return (out["choices"][0]["message"]["content"] or "").strip()


def ask(model: str, prompt: str, api_key: str) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]
    text = _call(model, messages, api_key, max_tokens=3000)
    if parse_answer(text) is None:
        # did not conclude: nudge once for the bare verdict
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "Final answer, one word: True or False?"}]
        text = _call(model, messages, api_key, max_tokens=8)
    return text


import re


def parse_answer(text: str):
    # verdict = the last NON-EMPTY LINE, which must be exactly true/false
    # (mid-reasoning mentions of the words never count)
    for line in reversed(text.strip().splitlines()):
        line = line.strip().strip("*_\"'`#. ").lower()
        if not line:
            continue
        if line in ("true", "false"):
            return line == "true"
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--causalab", required=True)
    ap.add_argument("--models", nargs="+",
                    default=["openai/gpt-4o-mini", "anthropic/claude-haiku-4.5"])
    ap.add_argument("--n-per-level", type=int, default=20,
                    help="items per complexity level, half True half False")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.causalab)
    items = build_items(args.n_per_level, args.seed)
    n_true = sum(1 for i in items if i["label"])
    print(f"items: {len(items)} ({n_true} True / {len(items) - n_true} False)")

    if args.dry_run:
        print("dry run; first prompt:\n" + items[0]["prompt"])
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        # fall back to a .env next to the repo root (gitignored)
        for env_path in (".env", os.path.join(os.path.dirname(__file__), "../../.env")):
            if os.path.exists(env_path):
                for line in open(env_path):
                    if line.strip().startswith("OPENROUTER_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip().strip('"')
                        break
            if api_key:
                break
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set and no .env found (use --dry-run to test offline)")

    for model in args.models:
        results = []
        raw_unparsed = []
        for it in items:
            raw = ask(model, it["prompt"], api_key)
            ans = parse_answer(raw)
            if ans is None and len(raw_unparsed) < 3:
                raw_unparsed.append(repr(raw))
            results.append({**it, "answer": ans})
        scored = [r for r in results if r["answer"] is not None]
        acc = sum(r["answer"] == r["label"] for r in scored) / max(len(scored), 1)
        bias = sum(1 for r in scored if r["answer"]) / max(len(scored), 1)
        tpr = (sum(r["answer"] for r in scored if r["label"])
               / max(sum(1 for r in scored if r["label"]), 1))
        tnr = (sum(not r["answer"] for r in scored if not r["label"])
               / max(sum(1 for r in scored if not r["label"]), 1))
        print(f"\n{model}: acc {acc:.2f} | answers-True share {bias:.2f} | "
              f"TPR {tpr:.2f} TNR {tnr:.2f} | unparsed {len(results) - len(scored)}")
        if raw_unparsed:
            print("  unparsed samples:", "; ".join(raw_unparsed))
        for reg in sorted({r['register'] for r in scored}):
            sub = [r for r in scored if r["register"] == reg]
            a = sum(r["answer"] == r["label"] for r in sub) / len(sub)
            print(f"  {reg}: acc {a:.2f} (n={len(sub)})")
        for b in sorted({r['bin'] for r in scored}):
            sub = [r for r in scored if r["bin"] == b]
            a = sum(r["answer"] == r["label"] for r in sub) / len(sub)
            print(f"  ops-bin {b}: acc {a:.2f} (n={len(sub)})")


if __name__ == "__main__":
    main()

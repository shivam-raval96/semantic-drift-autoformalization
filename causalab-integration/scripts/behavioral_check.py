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
        [--models openai/gpt-4o-mini anthropic/claude-haiku-4.5] [--n-false 46]
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


def build_items(n_per_class: int, seed: int):
    from causalab.tasks.etp_implication.config import (
        CERTIFIED_FALSE,
        CERTIFIED_TRUE,
        DEFAULT_REGISTERS,
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
                         "prompt": fill_template(r, p, c)}
                    )
    k = min(n_per_class, len(by_label[True]), len(by_label[False]))
    items = rng.sample(by_label[True], k) + rng.sample(by_label[False], k)
    rng.shuffle(items)
    return items


def ask(model: str, prompt: str, api_key: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 4,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as r:
        out = json.load(r)
    return (out["choices"][0]["message"]["content"] or "").strip()


def parse_answer(text: str):
    t = text.strip().lower()
    if t.startswith("true"):
        return True
    if t.startswith("false"):
        return False
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--causalab", required=True)
    ap.add_argument("--models", nargs="+",
                    default=["openai/gpt-4o-mini", "anthropic/claude-haiku-4.5"])
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.causalab)
    items = build_items(args.n_per_class, args.seed)
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
        for it in items:
            ans = parse_answer(ask(model, it["prompt"], api_key))
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
        for reg in sorted({r['register'] for r in scored}):
            sub = [r for r in scored if r["register"] == reg]
            a = sum(r["answer"] == r["label"] for r in sub) / len(sub)
            print(f"  {reg}: acc {a:.2f} (n={len(sub)})")


if __name__ == "__main__":
    main()

"""Causal test of the certified-fingerprint chart (fp3).

The isometry result is correlational: activations at premise_last fit the
certified chart better than shuffled charts. This script asks whether the
chart is causally load-bearing:

  1a. Chart-restricted patching. For an item (A -> B judged; counterfactual
      premise A' with the opposite certified verdict vs B), patch the
      premise_last hidden state at layer L:
        full   : h += mu(A') - mu(A)          (per-law centroid difference)
        chart  : h += W @ (fp3(A') - fp3(A))  (move ONLY the 3 certified coords)
        rot    : h += R @ W @ (fp3(A')-fp3(A))  (random rotation of the same
                                                 delta -- matched norm control)
        shuf   : h += W_perm @ (fp3_perm(A') - fp3_perm(A))  (chart fitted
                  under a permuted law->fp3 assignment -- the isometry null)
      Prediction: chart moves the verdict toward cert(A'->B); rot and shuf
      do not; full is the ceiling.

  1b. Boundary-crossing steering. Walk c(t) = fp3(A) + t*(fp3(A')-fp3(A)),
      t in [0, 1]; steer h += W @ (c(t) - fp3(A)). The certified boundary
      t* is where the nearest law in fp3 space flips its certified verdict
      vs B. Prediction: P(True) flips near t*, not uniformly in t.

  1c. Inertness. Run the same protocol on a model that shows the geometry
      but is at chance on the task (Llama-3.2-1B): if the chart is present
      but unread, chart patches should NOT move verdicts coherently.

Leakage control: centroids/W are estimated from prompts whose conclusions
come from even-indexed laws; item conclusions B come from odd-indexed laws.

Run (from repo root, causalab venv):
  python causalab-integration/scripts/causal_patch.py --model qwen --layer 14
  python causalab-integration/scripts/causal_patch.py --model llama1b --layer 4
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tasks"))

DATA = os.path.join(HERE, "..", "tasks", "etp_implication", "data", "etp_pairs.json")
OUT_BASE = os.path.join(HERE, "..", "analysis", "causal")

MODELS = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
}

PROMPT = (
    "Consider a set with one binary operation. An equation that always "
    "holds is stated, then a question is asked.\n"
    "Rule: {premise}\n"
    "Does it follow that the next statement also always holds?\n"
    "Statement: {conclusion}\n"
    "Answer with True or False.\nAnswer:"
)

REGISTERS = ["formal", "instance"]
RNG = np.random.default_rng(2026)


def load_data():
    with open(DATA) as f:
        d = json.load(f)
    laws = d["laws"]
    pairs = d["pairs"]
    fp3 = {k: np.array(v["fp3"], dtype=np.float64) for k, v in laws.items()}
    return laws, pairs, fp3


class Runner:
    def __init__(self, model_name, layer, device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        # fp32 on MPS/CPU for stable logit diffs; bf16 on CUDA (8B must fit 24GB)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device).eval()
        self.layer = layer
        self.layers = self.model.model.layers
        assert 0 <= layer < len(self.layers), f"layer {layer} out of range"
        tt = [self.tok.encode(s, add_special_tokens=False) for s in (" True", "True")]
        ff = [self.tok.encode(s, add_special_tokens=False) for s in (" False", "False")]
        self.true_ids = sorted({t[0] for t in tt})
        self.false_ids = sorted({f[0] for f in ff})

    def _prompt_and_pos(self, laws, premise_lid, conclusion_lid, register):
        p_surf = laws[premise_lid][register]
        c_surf = laws[conclusion_lid][register]
        text = PROMPT.format(premise=p_surf, conclusion=c_surf)
        enc = self.tok(text, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        p_start = text.find(p_surf)
        p_end = p_start + len(p_surf)
        toks = [i for i, (a, b) in enumerate(offsets) if a < p_end and b > p_start]
        if not toks:
            raise ValueError("premise span not found in offsets")
        return enc.to(self.device), toks[-1]  # premise_last token index

    @torch.no_grad()
    def forward(self, enc, patch_pos=None, patch_delta=None):
        """Run; optionally add patch_delta (np array, hidden dim) to the
        residual stream at layer output self.layer, position patch_pos.
        Returns (logit_true - logit_false, hidden state at premise_last)."""
        handle = None
        captured = {}

        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured["h"] = hs[0, patch_pos_holder[0], :].detach().float().cpu().numpy()
            if patch_delta is not None:
                delta = torch.tensor(
                    patch_delta, dtype=hs.dtype, device=hs.device
                )
                hs = hs.clone()
                hs[0, patch_pos_holder[0], :] += delta
                if isinstance(output, tuple):
                    return (hs,) + tuple(output[1:])
                return hs
            return output

        patch_pos_holder = [patch_pos]
        handle = self.layers[self.layer].register_forward_hook(hook)
        try:
            out = self.model(**enc)
        finally:
            handle.remove()
        logits = out.logits[0, -1, :].float()
        lt = torch.logsumexp(logits[self.true_ids], dim=0).item()
        lf = torch.logsumexp(logits[self.false_ids], dim=0).item()
        return lt - lf, captured.get("h")


def build_centroids(runner, laws, pairs, fp3, centroid_conclusions, k_prompts=6):
    """Per-law mean activation at (layer, premise_last), pooled over
    registers and a fixed pool of conclusions (leakage-controlled)."""
    law_ids = sorted(fp3.keys())
    centroids = {}
    for i, lid in enumerate(law_ids):
        acts = []
        concs = [c for c in centroid_conclusions if c != lid]
        picks = RNG.choice(len(concs), size=min(k_prompts, len(concs)), replace=False)
        for j in picks:
            for reg in REGISTERS:
                if reg not in laws[lid] or reg not in laws[concs[j]]:
                    continue
                enc, pos = runner._prompt_and_pos(laws, lid, concs[j], reg)
                _, h = runner.forward(enc, patch_pos=pos)
                acts.append(h)
        centroids[lid] = np.mean(acts, axis=0)
        if (i + 1) % 50 == 0:
            print(f"  centroids {i+1}/{len(law_ids)}", flush=True)
    return centroids


def fit_chart(centroids, fp3, perm=None):
    """Least-squares W: fp3 -> centroid (affine). perm optionally remaps
    law->fp3 (the shuffled-chart null)."""
    law_ids = sorted(centroids.keys())
    X = np.stack([fp3[perm[l] if perm else l] for l in law_ids])
    Y = np.stack([centroids[l] for l in law_ids])
    Xa = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    W = coef[:3].T  # hidden x 3
    return W


def sample_items(pairs, fp3, laws, item_conclusions, n_items, ops_of):
    """Items: (A, B, A') with cert(A->B) != cert(A'->B), both certified,
    B in the held-out conclusion pool. Balanced True->False / False->True."""
    by_b = {}
    for k, v in pairs.items():
        a, b = k.split("|")
        if b in item_conclusions and a in fp3 and b in fp3:
            by_b.setdefault(b, {"t": [], "f": []})["t" if v else "f"].append(a)
    items = []
    bs = [b for b in by_b if by_b[b]["t"] and by_b[b]["f"]]
    RNG.shuffle(bs)
    want_dir = True  # alternate direction for balance
    guard = 0
    while len(items) < n_items and guard < 20 * n_items:
        guard += 1
        b = bs[guard % len(bs)]
        t_pool, f_pool = by_b[b]["t"], by_b[b]["f"]
        a_true = t_pool[int(RNG.integers(len(t_pool)))]
        a_false = f_pool[int(RNG.integers(len(f_pool)))]
        if want_dir:
            a, a_prime, direction = a_false, a_true, "F->T"
        else:
            a, a_prime, direction = a_true, a_false, "T->F"
        if a == a_prime:
            continue
        items.append(dict(a=a, b=b, a_prime=a_prime, direction=direction,
                          ops=ops_of(a, b)))
        want_dir = not want_dir
    return items


def certified_boundary_t(fp3, pairs, a, a_prime, b, n_grid=41):
    """t* where the nearest law (in fp3, among laws certified vs b) flips
    its certified verdict vs b along the segment fp3(a)->fp3(a')."""
    cand = [l for l in fp3 if f"{l}|{b}" in pairs and l != b]
    P = np.stack([fp3[l] for l in cand])
    verd = np.array([1 if pairs[f"{l}|{b}"] else 0 for l in cand])
    c0, c1 = fp3[a], fp3[a_prime]
    ts = np.linspace(0, 1, n_grid)
    v_prev, t_star = None, None
    curve = []
    for t in ts:
        c = c0 + t * (c1 - c0)
        j = int(np.argmin(((P - c) ** 2).sum(1)))
        v = int(verd[j])
        curve.append(v)
        if v_prev is not None and v != v_prev and t_star is None:
            t_star = float(t)
        v_prev = v
    return t_star, ts.tolist(), curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-items", type=int, default=60)
    ap.add_argument("--n-steer", type=int, default=16)
    ap.add_argument("--k-centroid-prompts", type=int, default=6)
    ap.add_argument("--register", default="formal")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    global RNG
    RNG = np.random.default_rng(args.seed)

    out_dir = os.path.join(OUT_BASE, f"{args.model}_L{args.layer}")
    os.makedirs(out_dir, exist_ok=True)

    laws, pairs, fp3 = load_data()
    law_ids = sorted(fp3.keys())

    # leakage split: even-index laws -> centroid-conclusion pool, odd -> item Bs
    centroid_conclusions = [l for i, l in enumerate(law_ids) if i % 2 == 0]
    item_conclusions = set(l for i, l in enumerate(law_ids) if i % 2 == 1)

    print(f"loading {MODELS[args.model]} (layer {args.layer})", flush=True)
    runner = Runner(MODELS[args.model], args.layer)

    print("building per-law centroids (leakage-controlled pool)", flush=True)
    centroids = build_centroids(runner, laws, pairs, fp3,
                                centroid_conclusions, args.k_centroid_prompts)

    W = fit_chart(centroids, fp3)
    perm_map = dict(zip(law_ids, RNG.permutation(law_ids).tolist()))
    W_shuf = fit_chart(centroids, fp3, perm=perm_map)
    # residual variance explained by the 3D chart (linear CKA-ish sanity)
    Y = np.stack([centroids[l] for l in law_ids])
    X = np.stack([fp3[l] for l in law_ids])
    Xa = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    resid = Y - Xa @ coef
    r2 = 1 - (resid**2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    print(f"chart linear R^2 on centroids: {r2:.4f}", flush=True)

    ops_of = lambda a, b: laws[a]["n_ops"] + laws[b]["n_ops"]

    items = sample_items(pairs, fp3, laws, item_conclusions, args.n_items, ops_of)
    print(f"{len(items)} intervention items "
          f"({sum(1 for i in items if i['direction']=='F->T')} F->T)", flush=True)

    # fixed random rotation (activation space) for the matched-norm control
    hdim = W.shape[0]
    G = np.linalg.qr(RNG.standard_normal((hdim, hdim)))[0]

    results = []
    for n, it in enumerate(items):
        a, b, ap_ = it["a"], it["b"], it["a_prime"]
        reg = args.register if args.register in laws[a] and args.register in laws[ap_] and args.register in laws[b] else "instance"
        enc, pos = runner._prompt_and_pos(laws, a, b, reg)
        clean, _ = runner.forward(enc, patch_pos=pos)
        enc2, pos2 = runner._prompt_and_pos(laws, ap_, b, reg)
        donor, _ = runner.forward(enc2, patch_pos=pos2)

        d_full = centroids[ap_] - centroids[a]
        d_chart = W @ (fp3[ap_] - fp3[a])
        d_rot = G @ d_chart
        d_shuf = W_shuf @ (fp3[perm_map[ap_]] - fp3[perm_map[a]])

        row = dict(**it, register=reg, clean=clean, donor=donor,
                   norm_full=float(np.linalg.norm(d_full)),
                   norm_chart=float(np.linalg.norm(d_chart)))
        for name, delta in [("full", d_full), ("chart", d_chart),
                            ("rot", d_rot), ("shuf", d_shuf)]:
            ld, _ = runner.forward(enc, patch_pos=pos, patch_delta=delta)
            row[name] = ld
        results.append(row)
        if (n + 1) % 10 == 0:
            print(f"  items {n+1}/{len(items)}", flush=True)

    # summarize: effect = movement toward cert(A'->B), normalized by donor gap
    def summarize(rows):
        out = {}
        for cond in ("full", "chart", "rot", "shuf"):
            moved, frac = [], []
            flips = 0
            for r in rows:
                target = 1.0 if r["direction"] == "F->T" else -1.0
                gap = r["donor"] - r["clean"]
                m = (r[cond] - r["clean"]) * target
                moved.append(m)
                if abs(gap) > 1e-6:
                    frac.append((r[cond] - r["clean"]) / gap)
                want_true = r["direction"] == "F->T"
                if (r[cond] > 0) == want_true and (r["clean"] > 0) != want_true:
                    flips += 1
            out[cond] = dict(
                mean_directed_delta=float(np.mean(moved)),
                median_frac_of_donor_gap=float(np.median(frac)),
                flip_rate=flips / len(rows),
            )
        out["n"] = len(rows)
        out["clean_acc"] = float(np.mean([
            (r["clean"] > 0) == (r["direction"] == "T->F") for r in rows
        ]))  # clean verdict correct iff matches cert(A->B): T->F items start True
        return out

    summary = summarize(results)
    print(json.dumps(summary, indent=2), flush=True)

    # ---- 1b: boundary-crossing steering ----
    steer_items = [it for it in items if it["direction"] == "F->T"][: args.n_steer]
    steer_out = []
    ts = np.linspace(0, 1.25, 11)  # overshoot past A' a bit
    for n, it in enumerate(steer_items):
        a, b, ap_ = it["a"], it["b"], it["a_prime"]
        reg = args.register if all(args.register in laws[x] for x in (a, ap_, b)) else "instance"
        t_star, tgrid, vcurve = certified_boundary_t(fp3, pairs, a, ap_, b)
        enc, pos = runner._prompt_and_pos(laws, a, b, reg)
        curve = []
        for t in ts:
            delta = W @ (t * (fp3[ap_] - fp3[a]))
            ld, _ = runner.forward(enc, patch_pos=pos, patch_delta=delta)
            curve.append(float(ld))
        steer_out.append(dict(**it, t_star=t_star, ts=ts.tolist(), logit_diff=curve,
                              boundary_grid=tgrid, boundary_verdicts=vcurve))
        if (n + 1) % 5 == 0:
            print(f"  steer {n+1}/{len(steer_items)}", flush=True)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(dict(
            model=MODELS[args.model], layer=args.layer, seed=args.seed,
            register=args.register, chart_r2=float(r2),
            summary=summary, items=results, steering=steer_out,
        ), f, indent=1)
    print("wrote", os.path.join(out_dir, "results.json"), flush=True)


if __name__ == "__main__":
    main()

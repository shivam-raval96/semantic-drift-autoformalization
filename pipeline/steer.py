"""Steering along the drift direction (runs on the GPU instance).

Direction: difference of class means (drifted - faithful) in the residual stream at
the gauntlet's train-CV layer, computed from run-9 acts+labels (pushed alongside).
Intervention: activation addition of alpha * sigma * v_hat at every position of that
layer during generation (sigma = mean hidden-state norm, so alpha is in 'typical
activation' units). Endpoint: CERTIFIED drift/faithful/L-fail rates per alpha —
the steering curve with theorem-grade outcomes.

    ~/venv/bin/python steer.py [layer] [alphas...]   # defaults: 16, -6..6
Writes steer-results.jsonl + steer-summary.json.
"""
import json, sys
import numpy as np
from hf_backend import HFBackend
from pilot import score

ACTS = 'acts-hf-meta-llama-Llama-3.1-8B-Instruct.npz'
RES = 'results-hf-meta-llama-Llama-3.1-8B-Instruct.jsonl'
MODEL = 'meta-llama/Llama-3.1-8B-Instruct'

def drift_direction(layer):
    acts = np.load(ACTS)
    rows = [json.loads(l) for l in open(RES, encoding='utf-8')]
    d, f = [], []
    for r in rows:
        if r['item_id'] not in acts.files or r['verdict'].startswith('syntax'):
            continue
        (d if r['verdict'].startswith('drift') else f).append(
            acts[r['item_id']][layer].astype(np.float32))
    v = np.mean(d, axis=0) - np.mean(f, axis=0)
    return v / np.linalg.norm(v)

def main():
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    rest = sys.argv[2:]
    randdir = 'random' in rest  # control: random unit vector, same norms/protocol
    saveacts = 'saveacts' in rest  # dose-response: save per-alpha acts (layers 8/16/24)
    alphas = [float(a) for a in rest if a not in ('random', 'saveacts')] or [-6, -3, -1.5, 0, 1.5, 3, 6]
    items = [json.loads(l) for p in ('data.jsonl', 'data-etp.jsonl')
             for l in open(p, encoding='utf-8')]
    seen, uniq = set(), []
    for it in items:
        if it['task'] in ('translation', 'transcription') and it['surface'] not in seen:
            seen.add(it['surface'])
            uniq.append(it)
    if randdir:
        rng = np.random.default_rng(0)
        v = rng.standard_normal(len(drift_direction(layer)))
        v = (v / np.linalg.norm(v)).astype(np.float32)
        print('CONTROL: random direction', flush=True)
    else:
        v = drift_direction(layer)
    b = HFBackend(MODEL)
    import torch
    vt = torch.tensor(v, dtype=b.model.dtype, device=b.device)
    state = {'coef': 0.0}
    target = b.model.model.layers[layer - 1]  # hidden_states[layer] = output of block layer-1

    def hook(module, inputs, output):
        if state['coef'] == 0.0:
            return output
        tup = isinstance(output, tuple)  # transformers 4.x layers return a bare Tensor
        h = output[0] if tup else output
        sigma = h.norm(dim=-1, keepdim=True).mean()
        h2 = h + state['coef'] * sigma * vt
        return (h2,) + output[1:] if tup else h2

    handle = target.register_forward_hook(hook)
    all_rows, summary = [], {}
    alphas = sorted(alphas, key=abs)  # alpha=0 FIRST: sanity anchor before spending
    for alpha in alphas:
        b.acts.clear()
        if b.device == 'cuda':
            import torch as _t
            _t.cuda.empty_cache()
        state['coef'] = alpha * 0.05  # 5% of typical norm per unit alpha
        results = []
        for it in uniq:
            r = dict(it)
            r['backend'] = f'steer:{MODEL}@L{layer}a{alpha}'
            r['alpha'] = alpha
            try:
                r['model_output'] = b.translate(it)
            except Exception as e:
                r['model_output'] = f'GENERATION_ERROR: {str(e)[:60]}'
            results.append(r)
        scored = score(results)
        from collections import Counter
        c = Counter('faithful' if s['verdict'] == 'faithful' else
                    'VBU' if s['verdict'].startswith('drift') else 'L' for s in scored)
        summary[str(alpha)] = dict(c)
        print(f'alpha={alpha}: {dict(c)}', flush=True)
        if saveacts:
            sub = {k: v[[8, 16, 24]] for k, v in b.acts.items()}
            np.savez_compressed(f'acts-steer-a{alpha}.npz', **sub)
        if alpha == 0 and c.get('faithful', 0) < 30:
            print('SANITY FAIL: alpha=0 does not reproduce baseline behavior — aborting '
                  'before spending on the sweep', flush=True)
            sys.exit(1)
        all_rows += scored
    handle.remove()
    res_path = 'steer-random-results.jsonl' if randdir else 'steer-results.jsonl'
    with open(res_path, 'w', encoding='utf-8') as f:
        for r in all_rows:
            f.write(json.dumps(r) + '\n')
    out = 'steer-random-summary.json' if randdir else 'steer-summary.json'
    json.dump({'layer': layer, 'random': randdir, 'summary': summary}, open(out, 'w'), indent=1)
    print('wrote steer-results.jsonl, steer-summary.json')

if __name__ == '__main__':
    main()

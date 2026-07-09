"""Local open-weight backend with residual-stream capture (task 6 / probe substrate).

Runs a small instruct model on CPU over the translation items, greedy decoding
(the temp-0 analogue), and stores the residual stream at the FINAL PROMPT TOKEN of
every layer — the pre-registered probe site — as float16 next to the results.

    python hf_backend.py [model_id]        # default Qwen/Qwen2.5-0.5B-Instruct
Writes results-hf-<slug>.jsonl + acts-hf-<slug>.npz (item_id -> (layers+1, d)).

Unique surfaces only (Family A repeats surfaces per drift move; greedy output would
be identical -> pure duplication). Labels for probes come from score() as usual.
"""
import json, sys, time
import numpy as np
from pilot import build_prompt, extract_equation, score, _slug

DEFAULT_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'

class HFBackend:
    def __init__(self, model_id=DEFAULT_MODEL, max_new_tokens=120):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model_id = model_id
        self.name = f'hf:{model_id}'
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
        # transformers 5.x names the kwarg `dtype`; 4.x (Lambda images keep the
        # driver-matched torch, which needs 4.x) still uses `torch_dtype`
        import transformers
        kw = ({'dtype': dtype} if int(transformers.__version__.split('.')[0]) >= 5
              else {'torch_dtype': dtype})
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        # Llama-3.1 reports eos_token_id as a LIST; passing that as pad_token_id
        # device-asserts inside generate. Resolve to a single int up front.
        pad = self.tok.pad_token_id
        if pad is None:
            eos = self.tok.eos_token_id
            pad = eos[0] if isinstance(eos, (list, tuple)) else eos
        self.pad_id = pad
        self.acts = {}  # item_id -> (n_layers+1, d) float16
        # dual-site capture (geometry runs): also store mean-pooled-over-prompt acts
        # under key f'{item_id}::mean', with layers subsampled (step 2) on BOTH sites
        # to keep files manageable. Default off: legacy single-site format unchanged.
        self.capture_mean = False

    def _prompt_ids(self, prompt):
        msgs = [{'role': 'user', 'content': prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors='pt').to(self.device)

    def translate(self, item):
        enc = self._prompt_ids(build_prompt(item))
        with self.torch.no_grad():
            fwd = self.model(**enc, output_hidden_states=True)
            hs = fwd.hidden_states
            if self.capture_mean:
                hs = hs[::2]
                self.acts[item['item_id'] + '::mean'] = np.stack(
                    [h[0].float().mean(dim=0).cpu().numpy().astype(np.float16) for h in hs])
            self.acts[item['item_id']] = np.stack(
                [h[0, -1, :].float().cpu().numpy().astype(np.float16) for h in hs])
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.pad_id)
        text = self.tok.decode(gen[0, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        return extract_equation(text)

    def judge_implication(self, item):
        raise NotImplementedError('probe substrate covers translation tasks only')


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    items, seen = [], set()
    for path in ('data.jsonl', 'data-etp.jsonl'):
        try:
            for line in open(path, encoding='utf-8'):
                d = json.loads(line)
                if d['task'] in ('translation', 'transcription') and d['surface'] not in seen:
                    seen.add(d['surface'])
                    items.append(d)
        except FileNotFoundError:
            print(f'({path} missing, skipped)')
    print(f'{len(items)} unique surfaces; loading {model_id} ...')
    b = HFBackend(model_id)
    results, t0 = [], time.time()
    for i, item in enumerate(items):
        r = dict(item)
        r['backend'] = b.name
        try:
            r['model_output'] = b.translate(item)
        except Exception as e:  # scored as L-fail; keeps a paid run alive
            r['model_output'] = f'GENERATION_ERROR: {str(e)[:80]}'
        results.append(r)
        if (i + 1) % 20 == 0:
            rate = (time.time() - t0) / (i + 1)
            print(f'  {i+1}/{len(items)}  ({rate:.1f}s/item, ~{rate*(len(items)-i-1)/60:.0f} min left)', flush=True)
    scored = score(results)
    tag = _slug(b.name)
    with open(f'results-{tag}.jsonl', 'w', encoding='utf-8') as f:
        for s in scored:
            f.write(json.dumps(s) + '\n')
    np.savez_compressed(f'acts-{tag}.npz', **b.acts)
    from collections import Counter
    verdicts = dict(Counter(
        'faithful' if s['verdict'] == 'faithful' else
        'VBU' if s['verdict'].startswith('drift') else 'L-fail' for s in scored))
    print('verdicts:', verdicts)
    import torch, transformers
    from pilot import log_run
    log_run('hf_run', {'backend': b.name, 'n': len(scored), 'verdicts': verdicts,
                       'instrument': {'torch': torch.__version__,
                                      'transformers': transformers.__version__,
                                      'device': b.device,
                                      'max_new_tokens': b.max_new_tokens}})
    print(f'wrote results-{tag}.jsonl and acts-{tag}.npz')


if __name__ == '__main__':
    main()

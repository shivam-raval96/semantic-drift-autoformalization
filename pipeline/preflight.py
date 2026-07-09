"""Remote preflight: surface the REAL import error before paying for model downloads.
transformers' lazy loader masks inner ImportErrors as 'Could not find <class>' —
importing the modeling modules directly makes the true traceback visible."""
import sys, traceback

import torch, transformers
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '| transformers', transformers.__version__, flush=True)
assert torch.cuda.is_available(), 'no CUDA'
try:
    from transformers.models.qwen2 import modeling_qwen2   # noqa: F401
    from transformers.models.llama import modeling_llama   # noqa: F401
    from transformers import AutoModelForCausalLM           # noqa: F401
except Exception:
    traceback.print_exc()
    sys.exit(1)
print('preflight ok')

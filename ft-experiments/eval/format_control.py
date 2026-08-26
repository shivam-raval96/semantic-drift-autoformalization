#!/usr/bin/env python3
"""Unrelated in-context format tasks — is a fine-tuned model just worse at following formats?

60 items, 3 output formats, trivial content. If a model degrades on a new grammar
but scores the same here, the loss is grammar-specific rather than general.

    modal run eval/format_control.py --model 8b --adapter /models/checkpoints/v2-8b-s0/final
"""

import json
import re

import modal

app = modal.App("harsh-ft-format-control")

# Same image as eval/modal_eval.py: vLLM 0.25.x JIT-compiles flashinfer
# kernels at warmup, which needs the full CUDA toolkit (nvcc).
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("harsh-ft-grammar-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

# Container-side literals; they mirror ft-experiments/config.py (MODAL,
# GUARDRAILS) — kept literal because this module is imported inside
# containers where the registry file does not exist. The local entrypoint
# asserts against the registry at launch.
IMAGE_TAG = "nvidia/cuda:12.8.1-devel-ubuntu22.04 + python3.12 + uv:vllm"
MAX_TOKENS = 200      # short structured answers; also caps FT runaway fluency
MAX_MODEL_LEN = 4096  # prompts are ~120 tokens; caps KV allocation


def _generate(
    model_id: str,
    conversations: list,
    max_tokens: int,
    lora: dict | None = None,
    chat_kwargs: dict | None = None,
) -> dict:
    """Batch-generate greedily — modal_eval._generate minus the two-stage path."""
    import os
    import time

    # The shared workspace secret may expose the token under any of the
    # common names; vLLM/huggingface_hub want HF_TOKEN.
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"):
        if os.environ.get(key):
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            break

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    lora_request = None
    lora_kwargs = {}
    if lora is not None:
        from vllm.lora.request import LoRARequest

        lora_kwargs = {"enable_lora": True, "max_lora_rank": lora["rank"]}
        lora_request = LoRARequest("ft-adapter", 1, lora["path"])

    t0 = time.monotonic()
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=1,
        enforce_eager=False,
        **lora_kwargs,
    )
    load_s = time.monotonic() - t0
    weights.commit()  # persist freshly downloaded weights promptly

    t1 = time.monotonic()
    outputs = llm.chat(
        conversations,
        SamplingParams(temperature=0.0, max_tokens=max_tokens),
        lora_request=lora_request,
        **({"chat_template_kwargs": chat_kwargs} if chat_kwargs else {}),
    )
    rows = [
        {
            "text": out.outputs[0].text,
            "finish_reason": out.outputs[0].finish_reason,
            "completion_tokens": len(out.outputs[0].token_ids),
            "prompt_tokens": len(out.prompt_token_ids),
        }
        for out in outputs
    ]
    generate_s = time.monotonic() - t1

    return {
        "rows": rows,
        "backend": {
            "engine": "vllm",
            "vllm_version": str(vllm.__version__),
            "torch_version": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "max_model_len": MAX_MODEL_LEN,
            "chat_template": "model default via llm.chat, add_generation_prompt",
        },
        "timing": {"model_load_s": round(load_s, 1), "generate_s": round(generate_s, 1)},
    }


GUARDRAILS = dict(
    min_containers=0,
    max_containers=1,
    scaledown_window=60,  # user cap: <= 120s
    retries=0,
    timeout=900,  # 60 prompts x 200 tokens fits one 15-min window, FT included
)

# max_inputs=1: one run per container (a warm container building a second
# vLLM engine on a GPU still holding the first OOMs — modal_eval precedent).
_fn_common = dict(
    gpu="A10G", image=image, volumes={"/models": weights}, secrets=[hf_secret],
    max_inputs=1,
)


@app.function(**_fn_common, **GUARDRAILS)
def generate_a10g(
    model_id: str, conversations: list, max_tokens: int, lora: dict | None = None,
    chat_kwargs: dict | None = None,
) -> dict:
    return _generate(model_id, conversations, max_tokens, lora=lora,
                     chat_kwargs=chat_kwargs)


_fn_a100 = dict(
    gpu="A100-80GB", image=image, volumes={"/models": weights}, secrets=[hf_secret],
    max_inputs=1,
)


@app.function(**_fn_a100, **GUARDRAILS)
def generate_a100(
    model_id: str, conversations: list, max_tokens: int, lora: dict | None = None,
    chat_kwargs: dict | None = None,
) -> dict:
    return _generate(model_id, conversations, max_tokens, lora=lora,
                     chat_kwargs=chat_kwargs)


# ------------------------------------------------------------- the suite
#
# Fixed literal lists; item construction is pure index arithmetic (offsets
# coprime to the list lengths so each family cycles through all values in a
# different order). No RNG at build time or runtime.

NAMES = [
    "Amara", "Bruno", "Celine", "Dmitri", "Esther", "Farid", "Greta",
    "Hiroshi", "Ines", "Jonas", "Katya", "Leandro", "Mireille", "Nadia",
    "Omar", "Priya", "Quentin", "Rosa", "Stefan", "Tomoko",
]
COUNTS = [3, 7, 12, 19, 21, 26, 34, 38, 41, 47, 52, 58, 63, 67, 74, 79, 82, 88, 91, 96]
CITIES = [
    "Lisbon", "Oslo", "Kyoto", "Valparaiso", "Marrakesh", "Tallinn", "Cusco",
    "Bruges", "Adelaide", "Windhoek", "Quebec", "Sarajevo", "Galway",
    "Vientiane", "Cartagena", "Timisoara", "Bergen", "Matera", "Fremantle",
    "Mombasa",
]
OBJECTS = ["postcards", "lanterns", "teapots", "brass keys", "maps"]

TEXTS = [
    "{name} moved to {city} three winters ago. Since the move, {name} has "
    "catalogued {count} {obj}.",
    "{name} runs a small shop in {city}. The back shelf holds exactly "
    "{count} {obj}. Regulars stop by every morning to look at them.",
    "Every spring, {name} travels to {city} for the market season. This "
    "year's trip produced {count} {obj}, all carefully labelled.",
    "A collection of {count} {obj} fills the workshop that {name} keeps. "
    "The workshop stands on a quiet street in {city}.",
]

FAMILIES = ("json", "semicolon", "keyword")

INSTRUCTIONS = {
    "json": "Answer as a JSON object with exactly the keys 'name', 'count', 'city'.",
    "semicolon": "Answer with exactly one line: ANSWER: <name>; <count>; <city>",
    "keyword": "Answer with three lines: NAME=..., COUNT=..., CITY=...",
}

TASK = ("Read the text and extract three facts: the person's name, "
        "how many {obj} there are, and the city.")

N_PER_FAMILY = 20


def build_items() -> list:
    """All 60 items, deterministically. Each stores the expected triple."""
    items = []
    for fi, family in enumerate(FAMILIES):
        for j in range(N_PER_FAMILY):
            name = NAMES[(j + 7 * fi) % len(NAMES)]
            count = COUNTS[(3 * j + fi) % len(COUNTS)]
            city = CITIES[(j + 13 * fi) % len(CITIES)]
            obj = OBJECTS[(j + fi) % len(OBJECTS)]
            text = TEXTS[(j + fi) % len(TEXTS)].format(
                name=name, city=city, count=count, obj=obj)
            prompt = (f"{TASK.format(obj=obj)}\n\nText: {text}\n\n"
                      f"{INSTRUCTIONS[family]}")
            items.append({
                "id": f"{family}-{j:02d}",
                "family": family,
                "text": text,
                "prompt": prompt,
                "expected": {"name": name, "count": count, "city": city},
            })
    return items


# ------------------------------------------------------------- grading
#
# compliance = the format parses; accuracy = compliance + values match.
# _norm repairs the cosmetic damage checkform._clean repairs (backtick/bold
# wrappers, trailing sentence periods) and casefolds — the values themselves
# never begin or end with a stripped character.


def _norm(value) -> str:
    s = str(value).strip().strip("`*\"'")
    return s.rstrip(".").strip().casefold()


def _values_match(got: dict, expected: dict) -> bool:
    return all(_norm(got[k]) == _norm(expected[k]) for k in ("name", "count", "city"))


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _parse_json_answer(response: str):
    """The strict parse first; then the fenced block / brace-span repairs
    (the JSON-family analogue of the labeled-line scan below)."""
    candidates = [response.strip()]
    candidates += [m.strip() for m in _FENCE_RE.findall(response)]
    start, end = response.find("{"), response.rfind("}")
    if start != -1 and end > start:
        candidates.append(response[start:end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _grade_json(response: str, expected: dict) -> tuple:
    obj = _parse_json_answer(response)
    if obj is None:
        return False, False, "no parseable JSON object"
    keys = {str(k).casefold(): v for k, v in obj.items()}
    if len(obj) != 3 or set(keys) != {"name", "count", "city"}:
        return False, False, f"keys {sorted(map(str, obj))} != exactly name/count/city"
    return True, _values_match(keys, expected), None


# Labeled-line extraction mirrors checkform._LINE_RE's tolerance: optional
# quote/list/bold decoration, the label, then the value; last occurrence wins.


def _label_re(label: str, sep: str) -> re.Pattern:
    return re.compile(
        rf"^[ \t>*+-]*(?:\*\*)?\s*{label}\b\s*(?:\*\*)?\s*{sep}\s*(?P<value>.+)$",
        re.IGNORECASE | re.MULTILINE,
    )


_ANSWER_RE = _label_re("ANSWER", ":")
_KW_RES = {key: _label_re(key, "=") for key in ("NAME", "COUNT", "CITY")}


def _grade_semicolon(response: str, expected: dict) -> tuple:
    matches = _ANSWER_RE.findall(response)
    if not matches:
        return False, False, "no 'ANSWER:' line"
    fields = matches[-1].split(";")
    if len(fields) != 3:
        return False, False, f"{len(fields)} ';'-separated fields, need 3"
    got = dict(zip(("name", "count", "city"), fields))
    return True, _values_match(got, expected), None


def _grade_keyword(response: str, expected: dict) -> tuple:
    got = {}
    for key, pattern in _KW_RES.items():
        matches = pattern.findall(response)
        if not matches:
            return False, False, f"no '{key}=' line"
        got[key.casefold()] = matches[-1]
    return True, _values_match(got, expected), None


_GRADERS = {"json": _grade_json, "semicolon": _grade_semicolon, "keyword": _grade_keyword}


def grade_item(item: dict, response: str) -> dict:
    compliant, accurate, detail = _GRADERS[item["family"]](response, item["expected"])
    return {"compliant": compliant, "accurate": accurate, "detail": detail}


# ---------------------------------------------------- dry-run controls


def render_expected(item: dict) -> str:
    """The expected triple in the item's own format — must grade
    compliant + accurate (positive control)."""
    e = item["expected"]
    if item["family"] == "json":
        return json.dumps({"name": e["name"], "count": e["count"], "city": e["city"]})
    if item["family"] == "semicolon":
        return f"ANSWER: {e['name']}; {e['count']}; {e['city']}"
    return f"NAME={e['name']}\nCOUNT={e['count']}\nCITY={e['city']}"


def render_broken(item: dict) -> str:
    """Deliberately format-broken (a field/line missing) — must grade
    non-compliant (negative control)."""
    e = item["expected"]
    if item["family"] == "json":
        return json.dumps({"name": e["name"], "count": e["count"]})
    if item["family"] == "semicolon":
        return f"ANSWER: {e['name']}; {e['count']}"
    return f"NAME={e['name']}\nCOUNT={e['count']}"


def render_wrong_values(item: dict) -> str:
    """Right format, wrong content — must grade compliant but inaccurate."""
    wrong = dict(item, expected=dict(item["expected"], name="Zebulon"))
    return render_expected(wrong)


def run_dry():
    """Local, GPU-free verification: determinism + grader controls."""
    items = build_items()
    assert items == build_items(), "build_items is not deterministic"
    per_family = {f: sum(1 for i in items if i["family"] == f) for f in FAMILIES}
    assert len(items) == 60 and all(n == N_PER_FAMILY for n in per_family.values()), per_family
    assert len({i["id"] for i in items}) == 60, "duplicate item ids"

    for item in items:
        # extraction really is trivial: every expected value sits verbatim in the text
        e = item["expected"]
        assert e["name"] in item["text"], item["id"]
        assert str(e["count"]) in item["text"], item["id"]
        assert e["city"] in item["text"], item["id"]

        # positive control, plain and under the cosmetic damage we repair
        g = grade_item(item, render_expected(item))
        assert g["compliant"] and g["accurate"], (item["id"], g)
        g = grade_item(item, f"```\n{render_expected(item)}\n```")
        assert g["compliant"] and g["accurate"], (item["id"], "fenced", g)

        # negative controls: broken format, prose, wrong values
        g = grade_item(item, render_broken(item))
        assert not g["compliant"], (item["id"], "broken accepted", g)
        g = grade_item(item, "I cannot determine those facts.")
        assert not g["compliant"], (item["id"], "prose accepted", g)
        g = grade_item(item, render_wrong_values(item))
        assert g["compliant"] and not g["accurate"], (item["id"], "wrong values", g)
        if item["family"] == "json":  # exactly-the-keys clause: an extra key fails
            g = grade_item(item, json.dumps(dict(item["expected"], extra=1)))
            assert not g["compliant"], (item["id"], "extra key accepted", g)

    print(f"= built 60 items ({', '.join(f'{f} {n}' for f, n in per_family.items())}); "
          "second build byte-identical")
    print("= positive controls: 60/60 expected renderings compliant + accurate, "
          "plain and code-fenced")
    print("= negative controls: 60/60 broken renderings non-compliant; prose "
          "non-compliant; wrong-value renderings compliant but inaccurate; "
          "JSON extra-key rejected")
    sample = items[0]
    print(f"\n========== sample item ({sample['id']}) ==========")
    print(sample["prompt"])
    print(f"---------- expected {sample['expected']}")
    print(f"---------- expected rendering\n{render_expected(sample)}")
    print("========== end sample ==========\n")
    print("DRY RUN OK: deterministic build; every grader accepts its own "
          "expected rendering and rejects broken/wrong ones.")


# ------------------------------------------------------------ entrypoint


@app.local_entrypoint()
def main(
    model: str = "8b",
    adapter: str = "",
    adapter_rank: int = 16,
    dry_run: bool = False,
):
    import hashlib
    import importlib.util
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    if dry_run:
        run_dry()
        return

    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("ft_root_config", here.parent / "config.py")
    ftc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ftc)

    # Container-side literals must mirror the registry.
    assert IMAGE_TAG == ftc.MODAL["image_tag"], "image drifted from config.MODAL"
    for key, want in ftc.GUARDRAILS.items():
        assert GUARDRAILS[key] == want, f"guardrail {key} drifted from config.GUARDRAILS"

    entry = ftc.MODELS[model]
    if entry.get("parked"):
        raise SystemExit(f"{entry['hf_id']} is parked pending an explicit go — not runnable")
    model_id = entry["hf_id"]

    run_dry()  # free local gate: never spend GPU on a suite failing its own controls

    items = build_items()
    conversations = [[{"role": "user", "content": item["prompt"]}] for item in items]
    suite_sha = hashlib.sha256(
        "\n\n".join(item["prompt"] for item in items).encode()).hexdigest()
    lora = {"path": adapter, "rank": adapter_rank} if adapter else None
    # Qwen3 templates think by default; the frozen protocol is no-think.
    chat_kwargs = {"enable_thinking": False} if "qwen3" in model_id.lower() else None
    fn = generate_a100 if entry["gpu"] == "A100-80GB" else generate_a10g
    print(f"= resolved: model={model_id} gpu={entry['gpu']} "
          f"adapter={adapter or 'NONE (base)'} n={len(items)} suite={suite_sha[:12]}")

    t0 = time.monotonic()
    result = fn.remote(model_id, conversations, MAX_TOKENS,
                       lora=lora, chat_kwargs=chat_kwargs)
    wall_s = time.monotonic() - t0

    rows = []
    for item, gen in zip(items, result["rows"]):
        rows.append({
            "id": item["id"],
            "family": item["family"],
            "prompt": item["prompt"],
            "expected": item["expected"],
            "response": gen["text"],
            "finish_reason": gen["finish_reason"],
            "completion_tokens": gen["completion_tokens"],
            "prompt_tokens": gen["prompt_tokens"],
            **grade_item(item, gen["text"]),
        })

    summary = {}
    for family in FAMILIES:
        mine = [r for r in rows if r["family"] == family]
        compliant = sum(1 for r in mine if r["compliant"])
        accurate = sum(1 for r in mine if r["accurate"])
        summary[family] = {
            "n": len(mine),
            "compliant": compliant,
            "compliance_pct": round(100 * compliant / len(mine), 1),
            "accurate": accurate,
            "accuracy_pct": round(100 * accurate / len(mine), 1),
        }

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": ("format-following control (v2 DESIGN.md): separates "
                    "grammar-specific lock-in from general loss of in-context "
                    "format following"),
        "model": model_id,
        "model_key": model,
        "adapter": ({"path": adapter, "rank": adapter_rank} if adapter else None),
        "n_items": len(items),
        "suite_sha256": suite_sha,
        "instructions": INSTRUCTIONS,
        "sampling": {"temperature": 0.0, "max_tokens": MAX_TOKENS, "greedy": True},
        "chat_template_kwargs": chat_kwargs,
        "gpu_requested": entry["gpu"],
        "backend": result["backend"],
        "image": IMAGE_TAG,
        "guardrails": GUARDRAILS,
        "timing": {**result["timing"], "wall_s": round(wall_s, 1)},
        "summary": summary,
        "items": rows,
    }

    out_dir = ftc.PATHS["runs"] / "ft-v2" / "format-control"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}{'-ft' if adapter else ''}.json"
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{model_id} · format control · n={len(items)} · wall {wall_s:.0f}s "
          f"-> {out_path}")
    for family, s in summary.items():
        print(f"  {family:10s} n={s['n']:2d}  compliance {s['compliance_pct']:5.1f}%  "
              f"accuracy {s['accuracy_pct']:5.1f}%")


if __name__ == "__main__":
    import sys

    if "--dry-run" in sys.argv:
        run_dry()
    else:
        raise SystemExit("use `modal run ft-experiments/v2/format_control.py --model ...` "
                         "to evaluate, or `--dry-run` for the local checks")

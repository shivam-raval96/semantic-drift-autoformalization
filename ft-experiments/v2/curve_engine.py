#!/usr/bin/env python3
"""Checkpoint-curve evals in ONE container: one vLLM engine, N LoRA adapters.

Problem this solves: run_curve.sh evaluates N checkpoints by launching N
independent modal_eval runs, each paying the full engine-load cost (~6 min
for 32B) for ~2 min of generation. This harness loads the base model ONCE
(enable_lora=True) and iterates (checkpoint x arm) with vLLM's per-request
LoRA (LoRARequest), so the GPU spends its time generating, not loading.

Everything protocol-critical is IDENTICAL to eval/modal_eval.py — prompts
are built locally by the same helpers (checkform.build_prompt +
benchmark.wrap_prompt), the pinned template-SHA gate runs first, grading is
local via modal_eval's own _grade_chunk (imported by path), and each
(checkpoint, arm) writes a standard-runner-shaped directory:

    runs/<out-prefix>-<ckpt>/<model>-<arm>-limit<N>/
        results.jsonl  summary.json  run_meta.json

so base_table / compare_table / notebooks read curve points unchanged.
run_meta.json additionally records produced_by + the adapter path.

Usage (GPU run — one A100-80GB container for the whole curve):

    modal run v2/curve_engine.py --model qwen3-32b --run-name v2-qwen3-32b-s0 \
        --checkpoints step-0,step-100,step-300,step-500,step-700,step-900 \
        --arms story,story-bfar --limit 200 \
        --out-prefix curve-v2-v2-qwen3-32b-s0

    (checkpoint, arm) pairs whose output dir already holds summary.json are
    skipped, so a killed run resumes from where it stopped.

Validation (local, no GPU, no Modal calls — also runs under plain python):

    python v2/curve_engine.py --dry-run --model qwen3-32b
        Simulates eval/modal_eval.py's ACTUAL main() (remote calls stubbed
        with a recorder) at limit=5 for the A arm (story) and B arm
        (story-bfar), then asserts: (a) this harness's prompts are
        byte-equal to the conversations modal_eval sent, (b) the pinned
        template SHAs hold, (c) output-dir naming matches the standard
        runner's for the same inputs, (d) grading-path selection matches
        (story -> checkform.grade, story-bfar -> grammars.grade_b b_far) —
        proven by full-row-identical grading of canned well-formed and
        garbage responses in both grammars.

    python v2/curve_engine.py \
        --equivalence-check runs/<std-tag>/<model>-<arm>-limitN/results.jsonl \
        --equivalence-new runs/<curve-tag>/<model>-<arm>-limitN/results.jsonl
        Post-first-GPU-use check: asserts the multi-adapter engine produced
        row-by-row identical verdict statuses (and buckets) to a
        standard-runner (modal_eval --adapter ...) run of the same
        (checkpoint, arm). Run once after the first real curve to certify
        the shared-engine LoRA path, then trust the harness.
"""

import modal

app = modal.App("harsh-ft-curve-eval")

# Byte-identical image spec to eval/modal_eval.py so Modal reuses the same
# cached build (vLLM JIT-compiles flashinfer kernels -> devel base image).
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/models/hf"})
)
weights = modal.Volume.from_name("harsh-ft-grammar-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

# Container-side literals mirroring ft-experiments/config.py (EVAL, MODAL) —
# literal for the same reason as modal_eval.py: this module is imported in
# containers where the registry file does not exist. The local entrypoint
# asserts them against the registry before launching.
IMAGE_TAG = "nvidia/cuda:12.8.1-devel-ubuntu22.04 + python3.12 + uv:vllm"
MAX_TOKENS = 4096
MAX_MODEL_LEN = 8192

# One long-lived container per curve: 2h cap covers ~12 (ckpt, arm) pairs of
# limit-200 32B generation plus one engine load with slack; retries=0 and
# resume-by-summary.json mean a timeout costs only the unfinished pairs.
CURVE_GUARDRAILS = dict(
    min_containers=0,
    max_containers=1,
    scaledown_window=60,  # user cap: <= 120s
    retries=0,
    timeout=7200,
)

# Modal rejects the retries= param on generator functions outright — they
# never retry, i.e. retries=0 by construction; the stamped guardrails above
# still record the effective policy.
_GEN_FN_GUARDRAILS = {k: v for k, v in CURVE_GUARDRAILS.items() if k != "retries"}


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/models": weights},
    secrets=[hf_secret],
    max_inputs=1,  # same OOM rationale as modal_eval: never reuse a warm engine
    **_GEN_FN_GUARDRAILS,
)
def curve_generate(
    model_id: str,
    conversations_by_arm: dict,
    pairs: list,  # [[checkpoint, arm], ...] in execution order
    adapters: dict,  # checkpoint -> absolute path on the volume
    max_tokens: int,
    max_lora_rank: int,
    chat_kwargs: dict | None = None,
):
    """Generator: ONE engine load, then yield one result per (ckpt, arm).

    Yielding streams each pair's raw generations back as soon as it lands,
    so the local side grades and persists incrementally — a mid-run failure
    keeps every completed pair on disk.
    """
    import os
    import time

    # Same token normalization as modal_eval._generate: the shared workspace
    # secret may expose the token under any common name; vLLM wants HF_TOKEN.
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"):
        if os.environ.get(key):
            os.environ.setdefault("HF_TOKEN", os.environ[key])
            break

    missing = [ck for ck, path in adapters.items() if not os.path.isdir(path)]
    if missing:
        raise FileNotFoundError(
            f"adapter dirs missing on volume: {missing} (looked under {sorted(adapters.values())})"
        )

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    t0 = time.monotonic()
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=1,
        enforce_eager=False,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
    )
    load_s = round(time.monotonic() - t0, 1)
    weights.commit()  # persist freshly downloaded base weights promptly

    backend = {
        "engine": "vllm",
        "vllm_version": str(vllm.__version__),
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "max_model_len": MAX_MODEL_LEN,
        "chat_template": "model default via llm.chat, add_generation_prompt",
        "lora": {"enable_lora": True, "max_lora_rank": max_lora_rank},
    }
    # Stable int id per adapter — vLLM caches adapters by id.
    lora_ids = {ck: i for i, ck in enumerate(adapters, start=1)}

    for ck, arm in pairs:
        t1 = time.monotonic()
        outputs = llm.chat(
            conversations_by_arm[arm],
            SamplingParams(temperature=0.0, max_tokens=max_tokens),
            lora_request=LoRARequest(ck, lora_ids[ck], adapters[ck]),
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
        yield {
            "checkpoint": ck,
            "arm": arm,
            "rows": rows,
            "stage1_rows": None,  # _grade_chunk contract; curve has no two-stage
            "backend": backend,
            "timing": {"model_load_s": load_s, "generate_s": round(time.monotonic() - t1, 1)},
        }


# ---------------------------------------------------------------- local side
# Everything below runs on the laptop: registry, prompt building, grading,
# writing run dirs. No repo code ever ships into the container.


def _load_stack():
    """Load the registry, the repo helpers, and modal_eval (for _grade_chunk).

    Same loading pattern as eval/modal_eval.py main(): registry by path,
    repo modules via sys.path. modal_eval itself is imported by path so we
    reuse its _grade_chunk (row schema) verbatim instead of copying it.
    """
    import importlib.util
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent  # ft-experiments/v2
    spec = importlib.util.spec_from_file_location("ft_root_config", here.parent / "config.py")
    ftc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ftc)
    sys.path.insert(0, str(ftc.REPO))
    import benchmark  # noqa: F401  (wrap_prompt, bucket_of)
    import checkform  # noqa: F401  (build_prompt, grade, PROMPT_PATH)

    spec_me = importlib.util.spec_from_file_location(
        "curve_modal_eval", here.parent / "eval" / "modal_eval.py")
    me = importlib.util.module_from_spec(spec_me)
    spec_me.loader.exec_module(me)
    return ftc, benchmark, checkform, me, here


def _assert_frozen_protocol(ftc):
    """The pinned-SHA template gate + registry/literal agreement.

    Mirrors the gate at the top of eval/modal_eval.py main().
    """
    import hashlib

    for name, want in ftc.EVAL["template_shas"].items():
        got = hashlib.sha256((ftc.PATHS["prompts"] / name).read_bytes()).hexdigest()
        assert got == want, f"frozen template changed on disk: {name}"
    assert MAX_TOKENS == ftc.EVAL["max_tokens"], "MAX_TOKENS drifted from registry"
    assert MAX_MODEL_LEN == ftc.EVAL["max_model_len"]["single"], \
        "MAX_MODEL_LEN drifted from registry"


def _arm_plumbing(ftc, checkform, arm, here):
    """template / wrap_form / source_field / grade_fn for one arm.

    Minimal duplication of the template + B_GRAMMAR block inside
    eval/modal_eval.py main() (the source of truth — it is inline in a
    local_entrypoint there, so it cannot be imported piecemeal). The
    --dry-run mode asserts this stays behavior-identical to the original
    by running modal_eval's actual main() against it.
    """
    import importlib.util
    import sys

    assert arm != "two-stage", "the curve harness does not support the two-stage arm"
    assert arm in ftc.EVAL["arms"], f"unknown arm {arm!r}"
    template = {
        "story": checkform.PROMPT_PATH,
        "literal": ftc.PATHS["prompts"] / "literal_prompt.md",
        "story-bnear": ftc.PATHS["prompts"] / "formalize_prompt_bnear.md",
        "story-bfar": ftc.PATHS["prompts"] / "formalize_prompt_bfar.md",
        "literal-bnear": ftc.PATHS["prompts"] / "literal_prompt_bnear.md",
        "literal-bfar": ftc.PATHS["prompts"] / "literal_prompt_bfar.md",
    }[arm]
    B_GRAMMAR = {"story-bnear": "b_near", "story-bfar": "b_far",
                 "literal-bnear": "b_near", "literal-bfar": "b_far"}
    wrap_form = "literal" if arm.startswith("literal") else "story"
    source_field = "literal" if arm.startswith("literal") else "story"
    b_key = B_GRAMMAR.get(arm)
    if b_key is not None:
        v2g = sys.modules.get("v2_grammars")
        if v2g is None:
            spec_g = importlib.util.spec_from_file_location(
                "v2_grammars", here / "grammars.py")
            v2g = importlib.util.module_from_spec(spec_g)
            sys.modules["v2_grammars"] = v2g  # dataclass creation needs this
            spec_g.loader.exec_module(v2g)

        def grade_fn(text, meta, _g=v2g, _k=b_key):
            return _g.grade_b(text, _k, meta["canonical_e"], meta["canonical_f"])
    else:
        grade_fn = checkform.grade
    return {"template": template, "wrap_form": wrap_form,
            "source_field": source_field, "grade_fn": grade_fn, "b_key": b_key}


def _load_rows(ftc, limit):
    """eval_v1 rows in tier order — same loop as modal_eval main()."""
    import json

    rows = []
    for tier in ftc.EVAL["tiers"]:
        for line in (ftc.PATHS["eval_v1"] / f"eval_{tier}.jsonl").read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def _build_prompts(benchmark, checkform, rows, plumb, model_id):
    """Frozen-protocol prompts — same two calls as modal_eval main()."""
    prompts = []
    for r in rows:
        base = checkform.build_prompt({"story": r[plumb["source_field"]]},
                                      template_path=plumb["template"])
        prompts.append(benchmark.wrap_prompt(base, "off", model_id, plumb["wrap_form"]))
    return prompts


def _pair_out_dir(ftc, out_tag, model_key, arm, limit):
    """Standard runner's exact naming (incl. the -limitN clobber guard)."""
    dir_name = f"{model_key}-{arm}" + (f"-limit{limit}" if limit else "")
    return ftc.PATHS["runs"] / out_tag / dir_name


def _summarize(results_rows):
    """Per-tier three-way summary — same computation as modal_eval main()."""
    tiers = sorted({row["tier"] for row in results_rows})
    buckets = ("exact", "correct-swapped", "correct-dualized", "wrong", "unparseable")
    summary = {}
    for tier in tiers:
        mine = [row for row in results_rows if row["tier"] == tier]
        counts = {b: sum(1 for row in mine if row["bucket"] == b) for b in buckets}
        correct = counts["exact"] + counts["correct-swapped"] + counts["correct-dualized"]
        summary[tier] = {
            "n": len(mine),
            **counts,
            "correct": correct,
            "correct_pct": round(100 * correct / len(mine), 1),
            "unparseable_pct": round(100 * counts["unparseable"] / len(mine), 1),
            "length_finishes": sum(1 for row in mine if row["finish_reason"] == "length"),
        }
    return summary


def _write_pair(ftc, out_tag, model_key, arm, limit, graded, meta):
    """Persist one (checkpoint, arm): results.jsonl + summary.json + run_meta.json.

    Written whole (not appended): the remote side generates a pair
    atomically, so a landed pair is always complete.
    """
    import json

    out_dir = _pair_out_dir(ftc, out_tag, model_key, arm, limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for row in graded:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(_summarize(graded), indent=2) + "\n")
    return out_dir


def _pair_meta(ftc, model_id, arm, plumb, benchmark, n, limit, adapter_path,
               adapter_rank, backend, timing, wall_s, run_name, ck, out_prefix,
               checkpoints, arms):
    """run_meta.json — modal_eval's schema plus multi-adapter provenance."""
    import hashlib
    from datetime import datetime, timezone

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "checkpoint-curve",
        "model": model_id,
        "arm": arm,
        "regime": "off",
        "eval_version": "v1",
        "n": n,
        "limit": limit or None,
        "sampling": {"temperature": 0.0, "max_tokens": MAX_TOKENS, "greedy": True},
        "backend": backend,
        "gpu_requested": "A100-80GB",
        "adapter": {"path": adapter_path, "rank": adapter_rank},
        "image": IMAGE_TAG,
        "guardrails": CURVE_GUARDRAILS,
        "chunk_size": n,  # one batch per (checkpoint, arm)
        "max_model_len": MAX_MODEL_LEN,
        "prompt_template": plumb["template"].name,
        "prompt_template_sha256": hashlib.sha256(plumb["template"].read_bytes()).hexdigest(),
        "regime_suffix": benchmark.wrap_prompt("", "off", model_id, plumb["wrap_form"]),
        "timing": {**timing, "wall_s": round(wall_s, 1)},
        # -- additions over eval/modal_eval.py's run_meta ------------------
        "produced_by": "v2/curve_engine.py — one-container multi-adapter "
                       "checkpoint-curve harness (shared vLLM engine, "
                       "per-request LoRARequest)",
        "curve": {
            "run_name": run_name,
            "checkpoint": ck,
            "out_prefix": out_prefix,
            "checkpoints_requested": checkpoints,
            "arms_requested": arms,
            "engine_load_shared": True,  # model_load_s is the shared one-time load
        },
    }


# ------------------------------------------------------------------ dry run


def _dry_run(model_key: str, arms: list):
    """Local cross-check against eval/modal_eval.py — no GPU, no Modal calls.

    Runs modal_eval's ACTUAL main() (via .info.raw_f) with every remote
    generate function replaced by a recorder stub that captures the exact
    conversations/lora/chat_kwargs sent and returns canned responses, then
    asserts this harness reproduces every protocol-visible byte.
    """
    import hashlib
    import json
    import shutil

    ftc, benchmark, checkform, me, here = _load_stack()
    _assert_frozen_protocol(ftc)
    print(f"[dry-run] (b) template SHA gate: {len(ftc.EVAL['template_shas'])} "
          f"pinned templates verified")

    entry = ftc.MODELS[model_key]
    model_id = entry["hf_id"]
    chat_kwargs = {"enable_thinking": False} if "qwen3" in model_id.lower() else None
    fake_adapter = "/models/checkpoints/DRYRUN/step-0"
    scratch_tag = "curve-dryrun-scratch"  # under runs/, removed in finally
    limit = 5

    # Canned responses: index-cycled [well-formed-in-THIS-arm's-grammar,
    # garbage]. Well-formed A text is unparseable under grade_b(b_far) and
    # vice versa, so a swapped grading path cannot produce identical rows.
    canned_by_arm = {}
    for arm in arms:
        plumb = _arm_plumbing(ftc, checkform, arm, here)
        if plumb["b_key"] == "b_far":
            good = "LAW: (x ∘ y) = ((y ∘ y) ∘ x)\nDERIVE: (x ∘ y) = (y ∘ x)"
        elif plumb["b_key"] == "b_near":
            good = "GIVEN: f(x, y) = f(f(y, y), x)\nSHOW: f(x, y) = f(y, x)"
        else:
            good = "ASSUME: op(x, y) = op(op(y, y), x)\nASK: op(x, y) = op(y, x)"
        canned_by_arm[arm] = [good, "I cannot determine the answer."]

    class _Recorder:
        """Stands in for every modal generate function; returns canned rows."""

        def __init__(self):
            self.canned = None
            self.calls = []

        def remote(self, model_id, conversations, max_tokens, *args, **kwargs):
            self.calls.append({"model_id": model_id, "conversations": conversations,
                               "max_tokens": max_tokens, "kwargs": kwargs})
            rows = [{"text": self.canned[i % len(self.canned)], "finish_reason": "stop",
                     "completion_tokens": 0, "prompt_tokens": 0}
                    for i in range(len(conversations))]
            return {"rows": rows, "stage1_rows": None,
                    "backend": {"engine": "dry-run-stub"},
                    "timing": {"model_load_s": 0.0, "generate_s": 0.0}}

    recorder = _Recorder()
    for name in ("generate_a10g", "generate_a10g_ft", "generate_a100", "generate_a100_ft",
                 "generate_a10g_two_stage", "generate_a10g_two_stage_ft",
                 "generate_a100_two_stage", "generate_a100_two_stage_ft"):
        setattr(me, name, recorder)

    scratch_root = ftc.PATHS["runs"] / scratch_tag
    failures = []
    try:
        if scratch_root.exists():
            shutil.rmtree(scratch_root)
        for arm in arms:
            recorder.canned = canned_by_arm[arm]
            recorder.calls = []
            # modal_eval's REAL main(), remote calls stubbed.
            me.main.info.raw_f(model=model_key, arm=arm, limit=limit,
                               adapter=fake_adapter, adapter_rank=16,
                               out_tag=scratch_tag)
            assert len(recorder.calls) == 1, "expected exactly one chunk at limit=5"
            call = recorder.calls[0]

            # Harness side, built independently through this module's path.
            plumb = _arm_plumbing(ftc, checkform, arm, here)
            rows = _load_rows(ftc, limit)
            prompts = _build_prompts(benchmark, checkform, rows, plumb, model_id)
            conversations = [[{"role": "user", "content": p}] for p in prompts]

            # (a) byte-equal prompts, plus identical request plumbing.
            if conversations != call["conversations"]:
                failures.append(f"{arm}: prompt conversations differ from modal_eval's")
            if call["model_id"] != model_id:
                failures.append(f"{arm}: model_id differs")
            if call["kwargs"].get("chat_kwargs") != chat_kwargs:
                failures.append(f"{arm}: chat_kwargs differ "
                                f"({call['kwargs'].get('chat_kwargs')} != {chat_kwargs})")
            if call["kwargs"].get("lora") != {"path": fake_adapter, "rank": 16}:
                failures.append(f"{arm}: lora request dict differs")
            print(f"[dry-run] (a) {arm}: {len(conversations)} prompts byte-equal to "
                  f"modal_eval's sent conversations (chat_kwargs={chat_kwargs})")

            # (c) output-dir naming: the dir modal_eval just wrote must be
            # exactly the dir this harness would write for the same inputs.
            sim_dir = _pair_out_dir(ftc, scratch_tag, model_key, arm, limit)
            if not (sim_dir / "results.jsonl").exists():
                failures.append(f"{arm}: naming mismatch — modal_eval did not write "
                                f"{sim_dir} (found: {sorted(p.name for p in scratch_root.glob('*'))})")
            else:
                print(f"[dry-run] (c) {arm}: out-dir naming matches standard runner "
                      f"({sim_dir.relative_to(ftc.PATHS['runs'])})")
            full_name = _pair_out_dir(ftc, "t", model_key, arm, 0).name
            assert full_name == f"{model_key}-{arm}", "full-run naming drifted"

            # (d) grading-path selection: identity asserts + full-row equality.
            if plumb["b_key"] is None and plumb["grade_fn"] is not checkform.grade:
                failures.append(f"{arm}: A-arm grade fn is not checkform.grade")
            if plumb["b_key"] is not None:
                _g, _k = plumb["grade_fn"].__defaults__
                if _k != plumb["b_key"] or not hasattr(_g, "grade_b"):
                    failures.append(f"{arm}: B-arm grade fn is not grammars.grade_b({plumb['b_key']})")
            sim_rows = [json.loads(l) for l in
                        (sim_dir / "results.jsonl").open(encoding="utf-8") if l.strip()]
            result = {"rows": [{"text": canned_by_arm[arm][i % 2], "finish_reason": "stop",
                                "completion_tokens": 0, "prompt_tokens": 0}
                               for i in range(limit)],
                      "stage1_rows": None}
            my_rows = me._grade_chunk(rows, prompts, result, arm, model_id, "", "",
                                      plumb["grade_fn"], benchmark.bucket_of, hashlib)
            my_rows = json.loads(json.dumps(my_rows, ensure_ascii=False))
            if my_rows != sim_rows:
                failures.append(f"{arm}: graded rows differ from modal_eval's "
                                f"(schema/verdict/hash drift)")
            else:
                statuses = [r["verdict"]["status"] for r in my_rows]
                grader = ("checkform.grade" if plumb["b_key"] is None
                          else f"grammars.grade_b[{plumb['b_key']}]")
                print(f"[dry-run] (d) {arm}: grader={grader}; all {len(my_rows)} rows "
                      f"byte-identical to modal_eval's (statuses {statuses})")
    finally:
        if scratch_root.exists():
            shutil.rmtree(scratch_root)

    if failures:
        for f in failures:
            print(f"[dry-run] FAIL {f}")
        raise SystemExit("[dry-run] FAILED")
    print(f"[dry-run] PASS — harness is protocol-identical to eval/modal_eval.py "
          f"for arms {arms} (model {model_key}, limit {limit})")


# --------------------------------------------------------- equivalence check


def _equivalence_check(ref_path: str, new_path: str):
    """Row-by-row verdict equivalence: standard runner vs this harness.

    To be run ONCE after the first real GPU use of the curve harness, e.g.
    against an existing single-adapter run of the same (checkpoint, arm):
    asserts identical pair_id order and identical verdict statuses (and
    buckets) row-by-row. Raw response text differences are reported as info
    only — vLLM batch composition may legally perturb sampling-irrelevant
    bytes, but greedy verdicts must not move.
    """
    import json
    from pathlib import Path

    def load(p):
        return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]

    ref, new = load(ref_path), load(new_path)
    assert len(ref) == len(new), f"row counts differ: {len(ref)} vs {len(new)}"
    ref_ids = [r["pair_id"] for r in ref]
    new_ids = [r["pair_id"] for r in new]
    assert ref_ids == new_ids, "pair_id sequences differ — not the same eval slice"

    status_diffs, bucket_diffs, response_diffs = [], [], 0
    for i, (a, b) in enumerate(zip(ref, new)):
        if a["verdict"]["status"] != b["verdict"]["status"]:
            status_diffs.append((i, a["pair_id"], a["verdict"]["status"], b["verdict"]["status"]))
        if a["bucket"] != b["bucket"]:
            bucket_diffs.append((i, a["pair_id"], a["bucket"], b["bucket"]))
        if a["response"] != b["response"]:
            response_diffs += 1

    print(f"[equivalence] {len(ref)} rows · response text differs on {response_diffs} "
          f"(info only)")
    for i, pid, sa, sb in status_diffs[:20]:
        print(f"[equivalence] STATUS DIFF row {i} {pid}: ref={sa} new={sb}")
    for i, pid, ba, bb in bucket_diffs[:20]:
        print(f"[equivalence] BUCKET DIFF row {i} {pid}: ref={ba} new={bb}")
    assert not status_diffs and not bucket_diffs, (
        f"NOT equivalent: {len(status_diffs)} status diffs, {len(bucket_diffs)} bucket diffs")
    print("[equivalence] PASS — verdict statuses and buckets identical row-by-row")


# ---------------------------------------------------------------- real run


def _run_curve(model_key, run_name, checkpoints, arms, limit, out_prefix, adapter_rank):
    import hashlib
    import time

    ftc, benchmark, checkform, me, here = _load_stack()
    _assert_frozen_protocol(ftc)
    entry = ftc.MODELS[model_key]
    if entry.get("parked"):
        raise SystemExit(f"{entry['hf_id']} is parked pending an explicit go — not runnable")
    model_id = entry["hf_id"]
    # Qwen3 templates think by default; the frozen protocol is no-think.
    chat_kwargs = {"enable_thinking": False} if "qwen3" in model_id.lower() else None

    plumb_by_arm = {arm: _arm_plumbing(ftc, checkform, arm, here) for arm in arms}
    rows = _load_rows(ftc, limit)
    prompts_by_arm = {
        arm: _build_prompts(benchmark, checkform, rows, plumb, model_id)
        for arm, plumb in plumb_by_arm.items()
    }
    conversations_by_arm = {
        arm: [[{"role": "user", "content": p}] for p in prompts]
        for arm, prompts in prompts_by_arm.items()
    }
    adapters = {ck: f"{ftc.PATHS['checkpoints_volume']}/{run_name}/{ck}"
                for ck in checkpoints}

    # Resume: a (checkpoint, arm) with summary.json on disk is complete.
    pairs, skipped = [], []
    for ck in checkpoints:
        for arm in arms:
            out_dir = _pair_out_dir(ftc, f"{out_prefix}-{ck}", model_key, arm, limit)
            (skipped if (out_dir / "summary.json").exists() else pairs).append((ck, arm))
    for ck, arm in skipped:
        print(f"= skip {ck} {arm}: summary.json exists")
    if not pairs:
        print("= nothing to do — every (checkpoint, arm) already has summary.json")
        return
    print(f"= resolved: model={model_id} run={run_name} n={len(rows)} "
          f"limit={limit or 'full'} pending={len(pairs)}/{len(checkpoints) * len(arms)} "
          f"pairs · ONE container, ONE engine load")

    arms_needed = sorted({arm for _, arm in pairs}, key=arms.index)
    adapters_needed = {ck: adapters[ck] for ck in checkpoints
                       if any(p == ck for p, _ in pairs)}
    t0 = time.monotonic()
    landed = 0
    for result in curve_generate.remote_gen(
        model_id,
        {arm: conversations_by_arm[arm] for arm in arms_needed},
        [list(p) for p in pairs],
        adapters_needed,
        MAX_TOKENS,
        adapter_rank,
        chat_kwargs,
    ):
        ck, arm = result["checkpoint"], result["arm"]
        assert len(result["rows"]) == len(rows), \
            f"{ck} {arm}: got {len(result['rows'])} generations for {len(rows)} rows"
        plumb = plumb_by_arm[arm]
        graded = me._grade_chunk(rows, prompts_by_arm[arm], result, arm, model_id,
                                 "", "", plumb["grade_fn"], benchmark.bucket_of, hashlib)
        meta = _pair_meta(ftc, model_id, arm, plumb, benchmark, len(rows), limit,
                          adapters[ck], adapter_rank, result["backend"],
                          result["timing"], time.monotonic() - t0, run_name, ck,
                          out_prefix, checkpoints, arms)
        out_dir = _write_pair(ftc, f"{out_prefix}-{ck}", model_key, arm, limit,
                              graded, meta)
        landed += 1
        n_ok = sum(1 for r in graded if r["verdict"]["status"] == "correct")
        n_unp = sum(1 for r in graded if r["verdict"]["status"] == "unparseable")
        print(f"= landed {landed}/{len(pairs)} {ck} {arm}: correct {n_ok}/{len(graded)} "
              f"unparseable {n_unp} (gen {result['timing']['generate_s']}s) "
              f"-> {out_dir.relative_to(ftc.PATHS['runs'].parent)}", flush=True)
    print(f"= curve complete: {landed} pairs in {time.monotonic() - t0:.0f}s wall "
          f"(engine loaded once: {result['timing']['model_load_s']}s)")


# --------------------------------------------------------------- entrypoints


def _dispatch(model, run_name, checkpoints, arms, limit, out_prefix, adapter_rank,
              dry_run, equivalence_check, equivalence_new):
    arms_list = [a for a in arms.split(",") if a]
    if equivalence_check or equivalence_new:
        assert equivalence_check and equivalence_new, \
            "--equivalence-check needs both paths (--equivalence-check REF --equivalence-new NEW)"
        _equivalence_check(equivalence_check, equivalence_new)
        return
    if dry_run:
        _dry_run(model, arms_list)
        return
    assert run_name and checkpoints and out_prefix, \
        "real runs need --run-name, --checkpoints and --out-prefix"
    ckpts = [c for c in checkpoints.split(",") if c]
    _run_curve(model, run_name, ckpts, arms_list, limit, out_prefix, adapter_rank)


@app.local_entrypoint()
def main(
    model: str = "qwen3-32b",
    run_name: str = "",
    checkpoints: str = "",
    arms: str = "story,story-bfar",
    limit: int = 0,
    out_prefix: str = "",
    adapter_rank: int = 16,
    dry_run: bool = False,
    equivalence_check: str = "",
    equivalence_new: str = "",
):
    _dispatch(model, run_name, checkpoints, arms, limit, out_prefix, adapter_rank,
              dry_run, equivalence_check, equivalence_new)


if __name__ == "__main__":
    # --dry-run and --equivalence-check are pure-local and also runnable
    # without Modal credentials: python v2/curve_engine.py --dry-run
    import argparse

    cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    cli.add_argument("--model", default="qwen3-32b")
    cli.add_argument("--run-name", default="")
    cli.add_argument("--checkpoints", default="")
    cli.add_argument("--arms", default="story,story-bfar")
    cli.add_argument("--limit", type=int, default=0)
    cli.add_argument("--out-prefix", default="")
    cli.add_argument("--adapter-rank", type=int, default=16)
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--equivalence-check", default="")
    cli.add_argument("--equivalence-new", default="")
    a = cli.parse_args()
    if not (a.dry_run or a.equivalence_check):
        raise SystemExit("GPU runs go through `modal run v2/curve_engine.py ...`; "
                         "plain python supports --dry-run / --equivalence-check only")
    _dispatch(a.model, a.run_name, a.checkpoints, a.arms, a.limit, a.out_prefix,
              a.adapter_rank, a.dry_run, a.equivalence_check, a.equivalence_new)

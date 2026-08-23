#!/usr/bin/env python3
"""Where do the problems sit in one fixed frame, and how does that move as
fine-tuning proceeds?

QUESTION
--------
Principal component analysis finds the directions along which a set of vectors
varies most, so a very high-dimensional cloud can be drawn in two dimensions.
Applied to this project's activations on a single untrained model, it showed the
dataset's known structure: problems separate by which surface form they are
written in, and cluster by which underlying law they encode.

We now have fine-tuning checkpoints for the task of turning a themed story into
Rigid Grammar -- the terse notation an answer is written in -- published at the
Hugging Face repository `SoHarshh/mars-v-ft-checkpoints`. The question is what
happens to that picture as the model learns:

    Fixing one set of axes and never refitting them, where do the problems sit
    in that frame at each checkpoint, and how does the arrangement move?

Fixing the axes is the whole design. Principal components carry an arbitrary
sign and, when two explain similar amounts of variance, rotate freely into one
another, so a basis refitted per checkpoint would produce pictures that differ
for reasons having nothing to do with the model. One basis is fitted once and
every checkpoint is projected into it, so movement in the picture is movement of
the model.

WHY CHECKPOINTS CAN BE COMPARED AT ALL
--------------------------------------
Comparing activations between two models is normally meaningless: each model's
internal axes are arbitrary, so a shared frame could not be built. That does not
apply here. These are LoRA adapters -- the base model's weights are frozen and
training adds only a small low-rank correction -- so the coordinate system of
the residual stream, the running internal representation each layer reads and
writes, is identical at every checkpoint. Step 0 of each run is the freshly
initialized, untrained adapter, giving each trajectory an exact origin.

THE FRAME
---------
Fitted on the **step-0 activations**, the model before the manipulation, so the
frame is not contaminated by the training being observed; and on **half the
laws**, with every figure drawn from the other half, so the points being
interpreted are never the points that defined the axes.

Two frames come off the same activations:

- **Form frame.** Fitted on the story and plain-description rows only. Axes
  encode formality. Watch whether the story cloud migrates toward the
  description cloud, and where the notation sits relative to both.
- **Law frame.** Each form's mean is subtracted first, removing the difference
  between forms, and the basis is fitted on what remains. Axes encode which law.
  Watch whether the renderings of one law pull together. Without the
  subtraction this frame would not exist: the gap between forms is far larger
  than the gap between laws, so the components would only rediscover the forms.

**Rigid Grammar is excluded from both fits and projected in afterwards.** It is
14 words against 169 and 213, so it would otherwise capture the leading
component and spend it on a distinction already known. Projecting means it gets
coordinates in the frame -- same centre, same axes -- and appears in the
picture, without having had a vote in where the axes point.

WHAT EACH OUTCOME WOULD MEAN
----------------------------
- The clouds move, and move differently under the two training recipes: the
  geometry tracks what the model learned, and the direction of movement in a
  frame whose axes have been characterised says what changed in nameable terms.
- The clouds move the same way under both recipes: the movement tracks the
  weights having been perturbed, not the task having been learned. Reporting the
  task run alone would then be misleading.
- The clouds barely move while accuracy changes enormously: fine-tuning changes
  what the model does with its representation rather than the representation
  itself, which redirects attention to the layers after this one.
- The clouds move but the share of variance outside the fixed frame grows
  sharply: the real change is in directions the untrained model was not using,
  and a step-0 frame is the wrong instrument. The honest follow-up is a frame
  fitted on the final checkpoint.
- An undifferentiated blob throughout: a null result, and an acceptable one.
  This is cheap and exploratory.

CONTROLS AND WHAT LIMITS THIS
-----------------------------
- **Two recipes, and this is the main control.** Both runs share the base model,
  the adapter settings, the seed and the schedule, and differ only in training
  data. `task-pairs` learns the task; on the 32-billion-parameter base its
  accuracy goes from 34% to 99.7%. `grammar-only`, trained on bare notation with
  no story attached, learns to emit flawless notation that says the wrong thing;
  on the same base its accuracy goes from 34% to 3%. A rank-16 correction on
  every projection of every layer moves everything a little, so without a run
  that changed greatly *without* learning the task, "the clouds moved" says only
  that the weights moved.
- **Laws the model was fine-tuned on are excluded.** Reading activations for a
  memorized law measures memorization, not learning. Exclusion is by law class,
  folding together an equation, its swapped sides, its dual and both, so a law
  cannot slip through by arriving swapped. It is not a small effect: 44 of the
  100 laws in this cell appear in the fine-tuning corpus, leaving 56.
- **Same laws at every checkpoint**, seeded and identical, so motion in the
  picture is per-problem motion and accuracy differences are paired.
- **Step 0 of both runs must agree.** Both are untrained adapters, so their
  activations should be identical to floating-point noise. The run checks this
  and refuses to continue if they differ, because a mismatch means adapter
  switching is not doing what it claims and every later number is worthless.
- **Complexity is pinned exactly.** Every problem is drawn from one cell -- three
  operations per equation, three variables, depth three, on both sides -- so
  within a form every text is the same length to the word.
- **Text length and surface form cannot be separated here.** That pinning has a
  cost. Every story is 169 words and every description 213, with no overlap, so
  "story versus description" and "169 versus 213 tokens" are the same variable
  in this dataset. Excluding the 14-word notation from the fit stops it taking
  the leading axis, but nothing here resolves the story-description case, and
  with a single theme there is no length-matched second form to check against.
  The word count of every form is recorded and printed beside the axes so the
  ambiguity is visible rather than implied.
- **Activations are read on the bare text, with no task instructions**, so all
  three forms are embedded identically and no form is advantaged by an
  instruction that suits it. Accuracy is measured separately, with the real
  formalization prompt, since asking the model to formalize text that is already
  the notation is not a task.
- **This is descriptive.** A cloud moving alongside accuracy does not show the
  movement causes the accuracy.
- **One base model.** The 8-billion-parameter Llama base has both recipes and
  fits on one card. The 32-billion-parameter base also has both and is the
  replication. No earlier run in this repository is a baseline, since those used
  a 4-billion-parameter model; step 0 is the baseline.

RUNNING IT
----------
    python3 law-representation-across-finetuning.py --quick        # smoke test
    python3 law-representation-across-finetuning.py                # the real run
    python3 law-representation-across-finetuning.py --analyze-only # redo figures

The generating half needs one GPU. The 8-billion-parameter base occupies about
16 GB at 16-bit precision, and every checkpoint's adapter is held alongside it
so that switching checkpoints is not a reload: 21 adapters at 168 MB each is a
further 2 to 4 GB depending on the precision they load at. That fits a 24 GB
card with little room and a 40 GB card comfortably. On a card too small for
both trajectories at once, `--recipe` reads one at a time into the same output
directory, which gives the same result.

`meta-llama/Llama-3.1-8B-Instruct` is a gated repository: the licence has to be
accepted on the Hugging Face website and `huggingface-cli login` run, or the
model download fails with a permission error before anything else happens.

The analysis half needs no GPU, no PyTorch and no model. `--analyze-only` reads
the stored activations, so every table and figure can be redone on a laptop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import dataset, finetune, grading, pca, runs, stats
from shared.vendor import ensure_on_path

ensure_on_path()

from benchmark import FORMS as PROMPT_FORMS, load_equations, wrap_prompt  # noqa: E402
from checkform import build_prompt  # noqa: E402

EXPERIMENT = "law-representation-across-finetuning"

TRAINED_LAWS = Path(__file__).resolve().parent / "training-laws.json"

# One cell of the law list: three operations per equation, all on one side,
# three variables, depth three. Pinning all four means every text in a given
# form renders to exactly the same length, so nothing in the picture is a
# complexity gradient in disguise.
CELL = (0, 3, 3, 3)

# graft is one of the three themes the fine-tune was trained on; tea was held
# out of training entirely. Rigid Grammar is the notation an answer is written
# in. Equal counts of all three, but only the first two choose the axes.
FIT_FORMS = ("graft", "literal")
PROJECTED_FORMS = ("rg",)
ALL_FORMS = FIT_FORMS + PROJECTED_FORMS

FORM_LABELS = {
    "graft": "story (graft theme, trained on)",
    "literal": "plain description (never trained as input)",
    "rg": "Rigid Grammar (the answer notation)",
}

DEFAULT_BASE = "llama-3.1-8b"

# Llama-3.1-8B has 32 decoder blocks, so hidden-state indices run 0 to 32, where
# 0 is the embedding output.
#
# Spread across the stack rather than concentrated, because nothing available
# beforehand justifies concentrating. The per-module weight-change records
# published alongside the checkpoints (`trajectory.npz`) show the correction
# spread nearly evenly: every layer takes between 1.9% and 3.8% of the total
# change, against 3.1% for a perfectly even split. The only thing both recipes
# agree on is that the last three layers move least; above that they disagree,
# with the task run also moving layers 3 and 5 heavily and the notation-only run
# not. So the records rule out reading only the top of the stack and otherwise
# say little.
#
# 20 is the reference merely because the figures need one. Every quantity is
# recomputed at every layer here and the whole profile is reported, so a reader
# can see whether the reference layer was a lucky pick; `--layer` changes it
# without re-running the model.
DEFAULT_LAYERS = (0, 8, 16, 20, 24, 32)
REFERENCE_LAYER = 20

# Read positions. The text is presented as a bare chat message with no
# instructions, so the span covers the whole problem text.
POSITIONS = {
    "problem_end": "the last token of the problem text",
    "problem_mean": "averaged over every token of the problem text",
    "prompt_end": "the last token before the model would start writing",
}
REFERENCE_POSITION = "problem_mean"

DEFAULT_PAIRS = 200
DEFAULT_GRADE_N = 100
DEFAULT_ANSWER_TOKENS = 256
DEFAULT_COMPONENTS = 2


# --------------------------------------------------------------- the sweep


def build_conditions(base: str, only: Optional[str] = None) -> List[runs.Condition]:
    """One condition per checkpoint, across both recipes.

    Declared as data rather than as nested loops so each cell records the
    checkpoint it came from and an interrupted run resumes at the checkpoint it
    died in rather than starting the trajectory over.

    `only` restricts to one recipe. That is a concession to card size, not to
    the design: every checkpoint's adapter is held in memory at once, so one
    recipe at a time roughly halves what the adapters cost. Both recipes
    written into the same run directory give the same result as one invocation,
    since each checkpoint is read independently and the frame is fitted at
    analysis time. Running only one and stopping there does not, because the
    contrast between them is the control.
    """
    task, grammar = finetune.paired_runs(base)
    conditions = []
    for run in (task, grammar):
        recipe = run.name.split("_", 1)[1]
        if only and recipe != only:
            continue
        for step in run.steps:
            conditions.append(
                runs.Condition(
                    "{}_step-{}".format(recipe, step),
                    recipe=recipe,
                    step=step,
                    run_name=run.name,
                    base_model=run.base_model,
                    subfolder=run.subfolder(step),
                )
            )
    return conditions


def select_samples(pairs: int, seed: int) -> List[dict]:
    """The problems, with every law the fine-tune saw screened out first."""
    equations, _ = load_equations()
    trained = finetune.load_trained_law_hashes(TRAINED_LAWS)
    pool = dataset.index_by_shape(equations)[CELL]
    clean = finetune.untouched_equation_numbers(equations, trained, pool)
    if not clean:
        raise SystemExit("every law in cell {} appears in training".format(CELL))
    print(
        "cell {}: {} laws, {} clean after excluding those the fine-tune saw "
        "({} ordered pairs available)".format(
            CELL, len(pool), len(clean), dataset.orderable(clean)
        )
    )
    return dataset.sample_depth_balanced(
        equations,
        per_cell=pairs,
        seed=seed,
        cells=(CELL,),
        form="story",
        label_prefix="E",
        allowed=set(clean),
    )


def rendered(samples: Sequence[dict]) -> Dict[str, List[str]]:
    """Every problem in every surface form, aligned by position."""
    per_form: Dict[str, List[str]] = {form: [] for form in ALL_FORMS}
    for sample in samples:
        texts = dataset.render_all_forms(sample["metadata"])
        for form in ALL_FORMS:
            per_form[form].append(texts[form])
    return per_form


def task_prompts(samples: Sequence[dict], texts: Sequence[str]) -> List[str]:
    """The real formalization prompt, for the accuracy measurement only.

    Kept apart from the activation reading on purpose. Activations are taken on
    the bare text so that no form is advantaged by instructions that suit it;
    accuracy has to be measured on the task as the model was fine-tuned on it.
    """
    template = PROMPT_FORMS["story"][1]
    return [
        wrap_prompt(
            build_prompt(
                {"story": text, "metadata": sample["metadata"]},
                template_path=template,
            ),
            "off",  # this base model has no reasoning mode
            "",
            "story",
        )
        for sample, text in zip(samples, texts)
    ]


# ------------------------------------------------------------ reading the model


def activations_path(run: runs.RunDirectory, name: str) -> Path:
    return run.path / "activations-{}.npz".format(name)


def pending(
    run: runs.RunDirectory, conditions: Sequence[runs.Condition], force: bool
) -> List[runs.Condition]:
    """Checkpoints still to read, counting a missing activation file as unfinished.

    The shared helper knows only about records; here a checkpoint is not done
    until both halves exist, so a half-written one is redone rather than
    silently analysed with activations from a different run.
    """
    todo = []
    for condition in conditions:
        if force:
            run.drop(condition.name)
            activations_path(run, condition.name).unlink(missing_ok=True)
        path = activations_path(run, condition.name)
        if not (run.has(condition.name) and path.exists()):
            run.drop(condition.name)
            path.with_name(path.stem + ".partial.npz").unlink(missing_ok=True)
            todo.append(condition)
    return todo


def run_experiment(
    args, run: runs.RunDirectory, conditions: List[runs.Condition]
) -> None:
    from shared import generation, hooks, model as model_module

    todo = pending(run, conditions, force=args.force)
    if not todo:
        print("every checkpoint already has records and activations")
        return

    print(runs.describe(conditions, run))
    print("\nreading {} of {} checkpoints".format(len(todo), len(conditions)))

    samples = select_samples(args.pairs, args.seed)
    texts = rendered(samples)
    for form in ALL_FORMS:
        lengths = {len(t.split()) for t in texts[form]}
        print(
            "  {:<8} {:>3} words{}".format(
                form,
                min(lengths),
                "" if len(lengths) == 1 else " to {}".format(max(lengths)),
            )
        )

    base_model_name = conditions[0]["base_model"]
    subfolders = {c.name: c["subfolder"] for c in conditions}
    model, tokenizer = model_module.load_adapters(
        base_model_name, finetune.HF_REPO, subfolders
    )
    model_module.check_layers(model, args.layers)

    # The text as the model sees it, with no task instructions: identical
    # framing for all three forms, so nothing in the geometry is a difference
    # between instructions.
    generator = generation.Generator(model, tokenizer, batch_size=args.batch_size)
    flat_texts, flat_forms, flat_ids = [], [], []
    for form in ALL_FORMS:
        for sample, text in zip(samples, texts[form]):
            flat_texts.append(text)
            flat_forms.append(form)
            flat_ids.append(sample["pair_id"])
    chats = [generator.build_chat(text) for text in flat_texts]

    grade_samples = samples[: args.grade_n] if args.grade_n else []
    grade_prompts = task_prompts(grade_samples, texts["graft"][: args.grade_n])

    for condition in todo:
        model_module.select_adapter(model, condition.name)

        captured = hooks.capture_residuals(
            model,
            tokenizer,
            chats,
            args.layers,
            batch_size=args.act_batch_size,
            spans=flat_texts,
            progress="{}: activations".format(condition.name),
        )
        arrays = {
            "{}_L{}".format(name, layer): captured[source][layer].numpy().astype(
                np.float32
            )
            for name, source in (
                ("problem_end", "span_last"),
                ("problem_mean", "span_mean"),
                ("prompt_end", "last"),
            )
            for layer in args.layers
        }
        # Written under a temporary name and moved into place only once it is
        # complete, so a crash or a dropped connection leaves no truncated file
        # that a later run would mistake for a finished checkpoint.
        # The temporary name still ends in .npz because numpy appends that
        # suffix to any path that does not, which would leave the rename below
        # pointing at a file that was never written.
        final = activations_path(run, condition.name)
        partial = final.with_name(final.stem + ".partial.npz")
        np.savez_compressed(
            partial,
            pair_ids=np.array(flat_ids),
            forms=np.array(flat_forms),
            layers=np.array(list(args.layers)),
            **arrays
        )
        partial.replace(final)

        answers = []
        if grade_prompts:
            model_module.set_seed(args.seed)
            answers = generator.generate(
                grade_prompts,
                thinking=None,  # this base model has no reasoning mode
                max_new_tokens=args.answer_tokens,
                progress="{}: answers".format(condition.name),
            )

        with run.writing(condition.name) as writer:
            for sample, answer in zip(grade_samples, answers):
                writer.write(
                    grading.grade_record(
                        answer,
                        sample,
                        extra=dict(condition.settings, condition=condition.name),
                    )
                )
        rows = run.read(condition.name)
        print(
            "  {:<24} accuracy {}".format(
                condition.name,
                grading.correct_rate(rows) if rows else "not graded",
            )
        )


# ------------------------------------------------------------------- analysis


class Reading:
    """One checkpoint's activations, with the labels needed to slice them."""

    def __init__(self, path: Path):
        blob = np.load(path, allow_pickle=False)
        self.pair_ids = [str(x) for x in blob["pair_ids"]]
        self.forms = [str(x) for x in blob["forms"]]
        self.layers = [int(x) for x in blob["layers"]]
        self._blob = blob

    def vectors(self, position: str, layer: int) -> np.ndarray:
        return self._blob["{}_L{}".format(position, layer)]

    def rows(self, forms: Sequence[str], pair_ids: Sequence[str]) -> np.ndarray:
        """Indices of the rows in those forms and those laws."""
        wanted = set(pair_ids)
        keep = set(forms)
        return np.array(
            [
                i
                for i, (form, pid) in enumerate(zip(self.forms, self.pair_ids))
                if form in keep and pid in wanted
            ]
        )


def load_readings(
    run: runs.RunDirectory, conditions: Sequence[runs.Condition]
) -> Dict[str, Reading]:
    readings = {}
    for condition in conditions:
        path = activations_path(run, condition.name)
        if path.exists():
            readings[condition.name] = Reading(path)
    if not readings:
        raise SystemExit("no activations in {}; run without --analyze-only".format(run.path))
    return readings


def check_origins_agree(readings: Dict[str, Reading], position: str, layer: int) -> float:
    """Both recipes' step 0 is an untrained adapter, so both must read the same.

    A difference here means adapter switching is not doing what it claims, in
    which case every trajectory in this run is measuring the wrong thing. The
    number is reported rather than merely asserted, because "identical" for
    floating point means small, not zero.
    """
    origins = [name for name in readings if name.endswith("_step-0")]
    if len(origins) < 2:
        return float("nan")
    first = readings[origins[0]].vectors(position, layer)
    worst = 0.0
    for other in origins[1:]:
        diff = np.abs(first - readings[other].vectors(position, layer)).max()
        worst = max(worst, float(diff))
    scale = float(np.abs(first).mean())
    if scale > 0 and worst / scale > 1e-3:
        raise SystemExit(
            "the two untrained checkpoints disagree by {:.3g} against a mean "
            "activation of {:.3g}; adapter switching is not working, so no "
            "number in this run can be trusted".format(worst, scale)
        )
    return worst


def build_frames(
    origin: Reading, fit_ids: Sequence[str], position: str, layer: int, k: int
) -> Tuple[pca.Basis, pca.Basis]:
    """The two frozen frames, both fitted on step 0 and on the fit-half laws."""
    rows = origin.rows(FIT_FORMS, fit_ids)
    vectors = origin.vectors(position, layer)[rows]
    form_basis = pca.fit(vectors, k=k, labels=list(FIT_FORMS))

    forms = [origin.forms[i] for i in rows]
    law_basis = pca.fit(
        pca.center_within_groups(vectors, forms), k=k, labels=list(FIT_FORMS)
    )
    return form_basis, law_basis


def measure(
    reading: Reading,
    form_basis: pca.Basis,
    law_basis: pca.Basis,
    held_ids: Sequence[str],
    position: str,
    layer: int,
) -> dict:
    """Everything one checkpoint contributes to the curves."""
    vectors_all = reading.vectors(position, layer)

    rows = reading.rows(ALL_FORMS, held_ids)
    forms = [reading.forms[i] for i in rows]
    coords = form_basis.project(vectors_all[rows])
    middles = pca.centroids(coords, forms)

    fit_rows = reading.rows(FIT_FORMS, held_ids)
    fitted_forms = [reading.forms[i] for i in fit_rows]
    fitted = vectors_all[fit_rows]

    # The law frame only means anything after the between-form difference is
    # removed, exactly as when it was fitted.
    centred = pca.center_within_groups(fitted, fitted_forms)
    law_coords = law_basis.project(centred)
    law_ids = [reading.pair_ids[i] for i in fit_rows]

    out = {
        "centroids": {form: middles[form].tolist() for form in middles},
        "story_to_description": float(
            np.linalg.norm(middles[FIT_FORMS[0]] - middles[FIT_FORMS[1]])
        ),
        "outside_frame": form_basis.residual_share(fitted),
        "within_over_overall_full": pca.spread_ratio(centred, law_ids),
        "within_over_overall_frame": pca.spread_ratio(law_coords, law_ids),
    }
    for form in PROJECTED_FORMS:
        projected = vectors_all[reading.rows([form], held_ids)]
        out["outside_frame_{}".format(form)] = form_basis.residual_share(projected)
        out["{}_to_description".format(form)] = float(
            np.linalg.norm(middles[form] - middles[FIT_FORMS[1]])
        )
    return out


def layer_profile(
    readings: Dict[str, Reading],
    present: Sequence[runs.Condition],
    fit_ids: Sequence[str],
    held_ids: Sequence[str],
    origin_name: str,
    position: str,
    layers: Sequence[int],
    k: int,
) -> Dict[str, dict]:
    """The same measurements at every layer read, start and end of each recipe.

    Reported so the reference layer used for the figures can be seen for what it
    is. One layer chosen after the fact, out of six, is a choice with six
    chances to look impressive; showing all six costs nothing here, since the
    activations are already on disk, and makes that choice checkable.
    """
    profile: Dict[str, dict] = {}
    for layer in layers:
        form_basis, law_basis = build_frames(
            readings[origin_name], fit_ids, position, layer, k
        )
        entry = {"form_frame_explained": form_basis.explained.tolist()}
        for recipe in ("task-pairs", "grammar-only"):
            cells = sorted(
                (c for c in present if c["recipe"] == recipe), key=lambda c: c["step"]
            )
            if not cells:
                continue
            entry[recipe] = {
                where: measure(
                    readings[cell.name], form_basis, law_basis, held_ids, position, layer
                )
                for where, cell in (("start", cells[0]), ("end", cells[-1]))
            }
        profile[str(layer)] = entry
    return profile


def analyze(args, run: runs.RunDirectory, conditions: List[runs.Condition]) -> dict:
    readings = load_readings(run, conditions)
    present = [c for c in conditions if c.name in readings]

    samples = select_samples(args.pairs, args.seed)
    fit_half, held_half = dataset.split_alternating(samples)
    fit_ids = [s["pair_id"] for s in fit_half]
    held_ids = [s["pair_id"] for s in held_half]

    position, layer = args.position, args.layer
    origin_gap = check_origins_agree(readings, position, layer)

    origin_name = next(
        c.name for c in present if c["step"] == 0 and c["recipe"] == "task-pairs"
    )
    form_basis, law_basis = build_frames(
        readings[origin_name], fit_ids, position, layer, args.components
    )

    per_checkpoint = {}
    for condition in present:
        per_checkpoint[condition.name] = dict(
            measure(
                readings[condition.name],
                form_basis,
                law_basis,
                held_ids,
                position,
                layer,
            ),
            recipe=condition["recipe"],
            step=condition["step"],
            accuracy=_accuracy(run, condition.name),
        )

    texts = rendered(samples)
    summary = {
        "cell": list(CELL),
        "pairs": len(samples),
        "laws": len({l for s in samples for l in
                     (s["metadata"]["label_e"], s["metadata"]["label_f"])}),
        "fit_laws": len(fit_ids),
        "held_out_laws": len(held_ids),
        "forms": {form: FORM_LABELS[form] for form in ALL_FORMS},
        "words_per_form": {
            form: sorted({len(t.split()) for t in texts[form]}) for form in ALL_FORMS
        },
        "position": position,
        "layer": layer,
        "layers_read": readings[origin_name].layers,
        "components": args.components,
        "frame_fitted_on": {
            "checkpoint": origin_name,
            "forms": list(FIT_FORMS),
            "laws": len(fit_ids),
        },
        "form_frame_explained": form_basis.explained.tolist(),
        "law_frame_explained": law_basis.explained.tolist(),
        "untrained_checkpoints_max_difference": origin_gap,
        "accuracy_start_to_finish": {
            recipe: start_to_finish(run, present, recipe)
            for recipe in ("task-pairs", "grammar-only")
        },
        "layer_profile": layer_profile(
            readings, present, fit_ids, held_ids, origin_name, position,
            readings[origin_name].layers, args.components,
        ),
        "checkpoints": per_checkpoint,
    }
    run.write_summary(summary)
    print(report(summary))
    figures = draw(summary, readings, form_basis, law_basis, held_ids, args, run)
    for path in figures:
        print("wrote {}".format(path))
    return summary


def _records(run: runs.RunDirectory, condition: str) -> List[dict]:
    try:
        return run.read(condition)
    except FileNotFoundError:
        return []


def _accuracy(run: runs.RunDirectory, condition: str) -> Optional[dict]:
    rows = _records(run, condition)
    return grading.correct_rate(rows).as_dict() if rows else None


def start_to_finish(
    run: runs.RunDirectory, conditions: Sequence[runs.Condition], recipe: str
) -> Optional[dict]:
    """How much accuracy moved from the untrained adapter to the last checkpoint.

    Paired on the problems the two checkpoints share and tested with McNemar's
    test, because the comparison is the same problems answered by two versions
    of the same model. At these sample sizes the paired test is the difference
    between resolving a ten-point shift and not.
    """
    cells = sorted(
        (c for c in conditions if c["recipe"] == recipe), key=lambda c: c["step"]
    )
    if len(cells) < 2:
        return None
    first, last = _records(run, cells[0].name), _records(run, cells[-1].name)
    if not first or not last:
        return None
    paired = stats.pair_records(first, last)
    return dict(
        paired.as_dict(),
        from_step=cells[0]["step"],
        to_step=cells[-1]["step"],
        summary=str(paired),
    )


def _ordered(summary: dict, recipe: str) -> List[dict]:
    cells = [c for c in summary["checkpoints"].values() if c["recipe"] == recipe]
    return sorted(cells, key=lambda c: c["step"])


def report(summary: dict) -> str:
    """The numbers in plain terms, with what each is measured against."""
    lines = [
        "",
        "{} problems, {} laws, all three-operation and screened against the "
        "fine-tuning corpus".format(summary["pairs"], summary["laws"]),
        "axes fitted once on {} ({}), {} laws; every number below is on the "
        "{} laws held out of that fit".format(
            summary["frame_fitted_on"]["checkpoint"],
            " and ".join(summary["frame_fitted_on"]["forms"]),
            summary["frame_fitted_on"]["laws"],
            summary["held_out_laws"],
        ),
        "read at layer {} (of the layers read: {}), {}".format(
            summary["layer"],
            ", ".join(str(L) for L in summary["layers_read"]),
            POSITIONS[summary["position"]],
        ),
        "the two drawn axes hold {} of the variance of the points they were "
        "fitted on".format(
            ", ".join("{:.1%}".format(v) for v in summary["form_frame_explained"])
        ),
        "word count is identical within a form and never overlaps between "
        "them: " + ", ".join(
            "{} {}".format(form, "/".join(str(w) for w in words))
            for form, words in summary["words_per_form"].items()
        ),
        "  so an axis separating the forms cannot be told apart from an axis "
        "measuring length",
    ]
    gap = summary["untrained_checkpoints_max_difference"]
    if gap == gap:  # not nan
        lines.append(
            "the two untrained checkpoints agree to {:.2g}, so adapter "
            "switching is doing what it claims".format(gap)
        )

    for recipe in ("task-pairs", "grammar-only"):
        cells = _ordered(summary, recipe)
        if not cells:
            continue
        lines += ["", "{}:".format(recipe)]
        lines.append(
            "  {:>6}  {:>9}  {:>9}  {:>9}  {:>8}".format(
                "step", "story-desc", "within/all", "outside", "accuracy"
            )
        )
        for cell in cells:
            accuracy = cell.get("accuracy")
            lines.append(
                "  {:>6}  {:>9.3f}  {:>9.3f}  {:>8.1%}  {:>8}".format(
                    cell["step"],
                    cell["story_to_description"],
                    cell["within_over_overall_full"],
                    cell["outside_frame"],
                    "{:.1%}".format(accuracy["rate"]) if accuracy else "-",
                )
            )
        first, last = cells[0], cells[-1]
        start, end = first["story_to_description"], last["story_to_description"]
        lines.append(
            "  by the last checkpoint the story and description clouds sit "
            "{:.0%} {}".format(
                abs(end / start - 1) if start else float("nan"),
                "further apart than at step 0" if end > start
                else "closer together than at step 0",
            )
        )
        moved = summary["accuracy_start_to_finish"].get(recipe)
        if moved:
            lines.append(
                "  accuracy from step {} to step {}: {}".format(
                    moved["from_step"], moved["to_step"], moved["summary"]
                )
            )
    profile = summary.get("layer_profile") or {}
    if profile:
        lines += [
            "",
            "the same two numbers at every layer read, at step 0 and at the last "
            "checkpoint, so the layer the figures use can be seen in context:",
            "  {:>5}  {:<13}  {:>18}  {:>18}".format(
                "layer", "recipe", "story-desc", "outside frame"
            ),
        ]
        for layer in sorted(profile, key=int):
            entry = profile[layer]
            for recipe in ("task-pairs", "grammar-only"):
                cell = entry.get(recipe)
                if not cell:
                    continue
                lines.append(
                    "  {:>5}  {:<13}  {:>7.1f} to {:>7.1f}  {:>7.1%} to {:>7.1%}{}".format(
                        layer if recipe == "task-pairs" else "",
                        recipe,
                        cell["start"]["story_to_description"],
                        cell["end"]["story_to_description"],
                        cell["start"]["outside_frame"],
                        cell["end"]["outside_frame"],
                        "  <- the figures use this layer"
                        if int(layer) == summary["layer"] and recipe == "task-pairs"
                        else "",
                    )
                )

    lines += [
        "",
        "story-desc: how far apart the story and plain-description clouds sit "
        "in the frozen frame",
        "within/all: spread among the renderings of one law over the spread of "
        "everything; lower means the renderings of a law sit closer together",
        "outside:    the share of each checkpoint's spread the two drawn axes "
        "do not capture; if this climbs, the picture is missing the change",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------------- figures


def draw(
    summary: dict,
    readings: Dict[str, Reading],
    form_basis: pca.Basis,
    law_basis: pca.Basis,
    held_ids: Sequence[str],
    args,
    run: runs.RunDirectory,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"graft": "#1f77b4", "literal": "#d62728", "rg": "#7f7f7f"}
    paths = []
    held = ["{:.1%} of fitted variance".format(v)
            for v in summary["form_frame_explained"]]

    # Figure 1 -- the clouds, in the frame, at a handful of checkpoints.
    recipes = [r for r in ("task-pairs", "grammar-only") if _ordered(summary, r)]
    shown = {r: _pick(_ordered(summary, r), args.panels) for r in recipes}
    columns = max(len(v) for v in shown.values())
    fig, axes = plt.subplots(
        len(recipes), columns,
        figsize=(2.5 * columns, 2.8 * len(recipes)),
        squeeze=False, sharex=True, sharey=True,
    )
    for row, recipe in enumerate(recipes):
        for col in range(columns):
            ax = axes[row][col]
            if col >= len(shown[recipe]):
                ax.axis("off")
                continue
            cell = shown[recipe][col]
            name = "{}_step-{}".format(recipe, cell["step"])
            reading = readings[name]
            for form in ALL_FORMS:
                rows = reading.rows([form], held_ids)
                coords = form_basis.project(
                    reading.vectors(summary["position"], summary["layer"])[rows]
                )
                ax.scatter(
                    coords[:, 0], coords[:, 1], s=6, alpha=0.55,
                    color=colours[form], linewidths=0,
                    label=FORM_LABELS[form] if row == 0 and col == 0 else None,
                )
            ax.set_title("step {}".format(cell["step"]), fontsize=8)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(
                    "{}\ncomponent 2 ({})".format(recipe, held[1]), fontsize=7
                )
            if row == len(recipes) - 1:
                ax.set_xlabel("component 1 ({})".format(held[0]), fontsize=7)
    fig.suptitle(
        "The same problems, in axes fitted once on the untrained model and never "
        "refitted\n"
        "so movement between panels is movement of the model, not of the fit",
        fontsize=10,
    )
    fig.legend(loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    path = run.figure_path("{}-frame.png".format(EXPERIMENT))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # Figure 2 -- the quantities that do not depend on reading a scatter plot.
    panels = [
        ("story_to_description",
         "Distance between the story and\nplain-description clouds"),
        ("within_over_overall_full",
         "Spread within one law over\nspread overall (lower = tighter)"),
        ("outside_frame",
         "Share of spread the two drawn\naxes do not capture"),
        ("accuracy", "Accuracy on the same problems"),
    ]
    # The two recipes run for very different numbers of steps -- 1041 against 75
    # -- so plotting against the raw step crushes the shorter one into the left
    # edge and makes it unreadable. The x axis is each run's own progress
    # through its training instead, with the step counts named in the legend so
    # nobody reads "1.0" as the same amount of training in both.
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.2))
    for ax, (key, title) in zip(axes, panels):
        for recipe, style in (("task-pairs", "-o"), ("grammar-only", "--s")):
            cells = _ordered(summary, recipe)
            if not cells:
                continue
            final = max(c["step"] for c in cells) or 1
            x = [c["step"] / final for c in cells]
            if key == "accuracy":
                values = [c["accuracy"]["rate"] if c.get("accuracy") else None
                          for c in cells]
                kept = [(a, v) for a, v in zip(x, values) if v is not None]
                if not kept:
                    continue
                x, values = zip(*kept)
            else:
                values = [c[key] for c in cells]
            ax.plot(
                x, values, style, markersize=3.5, linewidth=1.2,
                label="{} ({} steps)".format(recipe, final),
            )
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("share of this recipe's training done", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25, linewidth=0.5)
        if key in ("accuracy", "outside_frame"):
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: "{:.0%}".format(v))
            )
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Two training recipes over the same base model, same settings, same seed: "
        "one learns the task, one only learns the notation",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = run.figure_path("{}-curves.png".format(EXPERIMENT))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def _pick(cells: List[dict], count: int) -> List[dict]:
    """Evenly spaced checkpoints, always keeping the first and the last."""
    if len(cells) <= count:
        return cells
    idx = sorted({round(i * (len(cells) - 1) / (count - 1)) for i in range(count)})
    return [cells[i] for i in idx]


# ------------------------------------------------------------------------ main


def main(argv: Optional[List[str]] = None) -> int:
    cli = runs.base_parser(__doc__.split("\n")[0])
    cli.add_argument("--base", default=DEFAULT_BASE,
                     help="which base model's pair of recipes (default: %(default)s)")
    cli.add_argument("--recipe", choices=("task-pairs", "grammar-only"),
                     help="read only this recipe's checkpoints, to halve what "
                          "the adapters cost on a smaller card. Run both into "
                          "the same --out-dir, then --analyze-only without "
                          "this flag to compare them")
    cli.add_argument("--pairs", type=int, default=DEFAULT_PAIRS,
                     help="problems to draw (default: %(default)s)")
    cli.add_argument("--layers", type=lambda t: tuple(int(x) for x in t.split(",")),
                     default=DEFAULT_LAYERS,
                     help="layers to read (default: %(default)s)")
    cli.add_argument("--layer", type=int, default=REFERENCE_LAYER,
                     help="the layer the figures use (default: %(default)s)")
    cli.add_argument("--position", default=REFERENCE_POSITION, choices=sorted(POSITIONS),
                     help="the read position the figures use (default: %(default)s)")
    cli.add_argument("--components", type=int, default=DEFAULT_COMPONENTS)
    cli.add_argument("--grade-n", type=int, default=DEFAULT_GRADE_N,
                     help="problems to grade per checkpoint, 0 to skip "
                          "(default: %(default)s)")
    cli.add_argument("--answer-tokens", type=int, default=DEFAULT_ANSWER_TOKENS)
    cli.add_argument("--batch-size", type=int, default=8)
    cli.add_argument("--act-batch-size", type=int, default=4)
    cli.add_argument("--panels", type=int, default=6,
                     help="checkpoints drawn per recipe in the scatter figure")
    args = cli.parse_args(argv)

    if args.quick:
        args.pairs = min(args.pairs, 6)
        args.layers = tuple(L for L in (0, 16, 32) if L in args.layers) or (16,)
        args.layer = args.layers[len(args.layers) // 2]
        args.grade_n = min(args.grade_n, 2)
        args.answer_tokens = min(args.answer_tokens, 64)
        args.panels = 3

    if args.layer not in args.layers:
        raise SystemExit(
            "--layer {} is not among the layers being read ({})".format(
                args.layer, ", ".join(str(L) for L in args.layers)
            )
        )

    out_dir = runs.resolve_out_dir(args, __file__)
    conditions = build_conditions(args.base)
    to_read = build_conditions(args.base, args.recipe)
    run = runs.RunDirectory.open(
        out_dir,
        EXPERIMENT,
        meta={
            "base": args.base,
            "cell": list(CELL),
            "pairs": args.pairs,
            "seed": args.seed,
            "layers": list(args.layers),
            "forms": list(ALL_FORMS),
            "fit_forms": list(FIT_FORMS),
            "grade_n": args.grade_n,
            "hf_repo": finetune.HF_REPO,
            "trained_laws": json.loads(TRAINED_LAWS.read_text())["n_law_hashes"],
        },
        force=args.force,
    )

    if not args.analyze_only:
        run_experiment(args, run, to_read)
    # Analysis always looks at every checkpoint present on disk, whatever this
    # invocation was asked to read, so two one-recipe runs into the same
    # directory still produce the comparison between them.
    analyze(args, run, conditions)
    return 0


if __name__ == "__main__":
    sys.exit(main())

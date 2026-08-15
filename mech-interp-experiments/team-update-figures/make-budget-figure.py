"""Redraws the thinking-budget dose-response for the team update.

Numbers are transcribed from the two findings docs, not recomputed: experiment 1 part 2
(`exp01-outputs/budget-summary.json`) and experiment 3 section F2 (`budget-sweep-*.jsonl`,
which live in Drive). Regenerate from those artifacts directly if they are ever synced local.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUDGETS = [0, 32, 64, 128, 256, 512, 1024]
STORY = [9.6, 1.9, 1.9, 1.9, 11.5, 38.5, 78.8]
LITERAL = [19.2, 17.3, 19.2, 17.3, 36.5, 96.2, 100.0]
STORY_CEILING = 90.4

# Experiment 3's independent sweep, prompted stories, one per law. Different harness and
# prompt, so it is a replication of the shape rather than a second measurement of the curve.
EXP3_BUDGETS = [0, 64, 128, 256, 512]
EXP3_STORY = [15.4, 11.5, 17.3, 25.0, 44.2]

x = range(len(BUDGETS))
exp3_x = [BUDGETS.index(b) for b in EXP3_BUDGETS]

fig, ax = plt.subplots(figsize=(8, 5))

ax.axhline(STORY_CEILING, color="black", ls="--", lw=1, zorder=1)
ax.text(
    5.95,
    STORY_CEILING + 1.8,
    "story, unlimited thinking (90.4%)",
    fontsize=8,
    color="black",
    ha="right",
)

ax.plot(x, LITERAL, "-o", color="tab:orange", label="literal-NL", zorder=3)
ax.plot(x, STORY, "-o", color="tab:blue", label="story", zorder=3)
ax.plot(
    exp3_x,
    EXP3_STORY,
    ":s",
    color="tab:blue",
    alpha=0.55,
    markersize=5,
    label="story, exp 3 independent sweep",
    zorder=2,
)

# The sub-floor dip: cut-off traces leave a half-finished draft for the grader to score.
ax.axhline(STORY[0], color="tab:blue", ls=":", lw=1, alpha=0.5, zorder=1)
ax.annotate(
    "below the B=0 floor\n(graded on a cut-off draft)",
    xy=(2, 1.9),
    xytext=(1.35, 26),
    fontsize=8,
    color="tab:blue",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8),
)
ax.annotate(
    "~2x budget offset",
    xy=(5, 67),
    xytext=(4.05, 68),
    fontsize=8,
    color="dimgray",
    ha="center",
)
ax.annotate(
    "",
    xy=(6, 78.8),
    xytext=(5, 96.2),
    arrowprops=dict(arrowstyle="<->", color="dimgray", lw=0.9),
)

ax.set_xticks(list(x))
ax.set_xticklabels([str(b) for b in BUDGETS])
ax.set_xlabel("thinking-token budget B")
ax.set_ylabel("correct (%)")
ax.set_ylim(0, 105)
ax.set_title("Accuracy is a function of thinking-token budget\n(52 eval pairs, Qwen3-4B)")
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("03-budget-dose-response.png", dpi=150)
print("wrote 03-budget-dose-response.png")

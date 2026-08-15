"""Redraws section C's retrieval curve for the team update.

The notebook's own figure plots three of the four position x pool combinations and omits the
strongest one (last token, story x theme), and draws its floor at the TF-IDF baseline rather than
the stronger length-only floor from experiment 0. Numbers are transcribed from the section C/D
table in `03-pca-activation-structure-findings.md`.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAYERS = [0, 6, 12, 18, 24, 30]

# Read position x candidate pool. "story x theme" restricts both query and candidates to stories
# of *different* themes, which share almost no vocabulary; "all" lets any other surface form win.
LAST_STORY = [1.9, 44.2, 59.1, 85.6, 86.1, 56.2]
LAST_ALL = [1.9, 30.8, 43.6, 64.1, 62.5, 39.7]
POOLED_STORY = [12.0, 26.4, 16.8, 24.5, 42.3, 19.7]
POOLED_ALL = [8.7, 18.3, 12.2, 17.3, 29.2, 13.8]

CHANCE = 1.9  # 1 / 52 laws
TFIDF_STORY = 4.8
EXP0_FLOOR = 24.5  # strongest model-free floor: word count alone, from experiment 0

fig, ax = plt.subplots(figsize=(8.5, 5.5))

ax.axhspan(0, EXP0_FLOOR, color="lightcoral", alpha=0.13, zorder=0)
ax.axhline(EXP0_FLOOR, color="firebrick", ls="--", lw=1.2, zorder=2)
ax.text(
    30,
    EXP0_FLOOR + 1.4,
    "strongest model-free floor: length alone, exp 0 (24.5%)",
    fontsize=8,
    color="firebrick",
    ha="right",
)
ax.axhline(TFIDF_STORY, color="gray", ls=":", lw=1, zorder=2)
ax.text(30, TFIDF_STORY + 1.4, "TF-IDF (4.8%)", fontsize=8, color="gray", ha="right")
ax.axhline(CHANCE, color="gray", ls="-", lw=0.8, alpha=0.6, zorder=2)
ax.text(20.5, CHANCE + 1.5, "chance (1.9%)", fontsize=8, color="gray")

ax.plot(LAYERS, LAST_STORY, "-o", color="tab:green", lw=2.6, ms=7,
        label="last token, story x theme  (the headline)", zorder=5)
ax.plot(LAYERS, LAST_ALL, "--s", color="tab:green", alpha=0.55, lw=1.5,
        label="last token, all surfaces", zorder=4)
ax.plot(LAYERS, POOLED_STORY, "-^", color="tab:blue", alpha=0.8, lw=1.5,
        label="mean-pooled, story x theme", zorder=4)
ax.plot(LAYERS, POOLED_ALL, "--v", color="tab:blue", alpha=0.45, lw=1.5,
        label="mean-pooled, all surfaces", zorder=3)

ax.annotate(
    "peak 86.1%\n(layers 18-24)",
    xy=(24, 86.1),
    xytext=(20.2, 97),
    fontsize=9,
    color="tab:green",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="tab:green", lw=1),
)
ax.annotate(
    "at the embedding layer the last token\ncarries no law signal at all",
    xy=(0.55, 2.4),
    xytext=(13.5, 9.5),
    fontsize=8,
    color="dimgray",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8),
)

ax.set_xticks(LAYERS)
ax.set_xlabel("layer (hidden_states index, 0 = embeddings)")
ax.set_ylabel("1-NN law retrieval accuracy (%)")
ax.set_ylim(0, 104)
ax.set_xlim(-1.2, 31)
ax.set_title(
    "The same law is recognisable across surface forms by mid-stack\n"
    "nearest neighbour must be a different surface form; 52 laws x 6 forms, Qwen3-4B"
)
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("01-semantic-core-by-layer.png", dpi=150)
print("wrote 01-semantic-core-by-layer.png")

"""Forest-style summary of the fragility fraction across the two corpora."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parent
FIG = OUT.parent / "latex" / "figs"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

IND = "#3730a3"   # tier A indigo
AMB = "#92400e"   # tier B amber-brown
GRN = "#046a38"   # tier C green
IN2 = "#1c1c1c"

cap = pd.read_csv(OUT / "phi_within_subject.csv")          # MMLU, phi_model
beh = pd.read_csv(OUT / "phi_moralchoice.csv")

short = {
    "openai_gpt-4": "GPT-4", "openai_gpt-3.5-turbo": "GPT-3.5-turbo",
    "openai_text-davinci-003": "davinci-003", "openai_text-davinci-002": "davinci-002",
    "anthropic_claude-v1.3": "Claude-v1.3",
    "anthropic_claude-instant-v1.1": "Claude-instant-v1.1",
    "google_text-bison-001": "text-bison-001", "google_flan-t5-xl": "Flan-T5-XL",
    "cohere_command-xlarge": "Command-XL", "ai21_j2-jumbo-instruct": "J2-Jumbo",
    "bigscience_bloomz-7b1": "BLOOMZ-7B1", "meta_opt-iml-max-small": "OPT-IML",
}

fig, ax = plt.subplots(figsize=(5.4, 3.05))

hi = beh[beh.ambiguity == "high"].copy()
lo = beh[beh.ambiguity == "low"].set_index("model")
hi["label"] = hi.model.map(short).fillna(hi.model)
hi = hi.sort_values("phi_broad")
y = np.arange(len(hi))

# behavioural: strict -> broad range per model
for i, (_, r) in enumerate(hi.iterrows()):
    ax.plot([r.phi_strict, r.phi_broad], [i, i], color=AMB, lw=1.5,
            solid_capstyle="round", alpha=.85, zorder=2)
ax.scatter(hi.phi_strict, y, s=16, color=AMB, zorder=3,
           label=r"behavioural: $\phi_{\rm strict}\!\rightarrow\!\phi_{\rm broad}$ (dilemmas)")
ax.scatter(hi.phi_broad, y, s=16, color=AMB, zorder=3, marker="D")

lo_vals = [lo.loc[m, "phi_broad"] if m in lo.index else np.nan for m in hi.model]
ax.scatter(lo_vals, y, s=13, facecolors="none", edgecolors=GRN, lw=1.0,
           zorder=3, label=r"behavioural: $\phi_{\rm broad}$ (clear-answer items)")

cap_med = cap.phi_model.median()
ax.axvline(cap_med, color=IND, lw=1.4, ls="-", zorder=1)
ax.text(cap_med * 1.3, len(hi) - 2.2,
        f"capability benchmark\nmedian $\\phi$ = {cap_med:.3f}",
        color=IND, fontsize=7.2, va="center", ha="left")

for x, lab in ((0.03, r"$K=1$"), (0.17, r"$K=5$"), (0.67, r"$K=20$")):
    ax.axvline(x, color="#94a3b8", lw=0.8, ls=":", zorder=0)
    ax.text(x, len(hi) - 0.35, lab, fontsize=6.6, color="#475569",
            ha="center", va="bottom")

ax.set_yticks(y)
ax.set_yticklabels(hi.label, fontsize=7.2)
ax.set_xscale("log")
ax.set_xlim(0.0015, 1.6)
ax.set_ylim(-0.8, len(hi) + 0.45)
ax.set_xlabel(r"fragility fraction $\hat\phi$ (log scale); dotted lines = tolerable "
              r"$\phi$ at 5\% inversion risk for a pool of $K$ probes", fontsize=7)
ax.grid(axis="x", color="#e2e8f0", lw=0.5, zorder=0)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=6.8, loc="lower left", bbox_to_anchor=(0.0, -0.30),
          ncol=2, frameon=False, handletextpad=.4, columnspacing=1.2)

fig.tight_layout()
fig.savefig(FIG / "phi_forest.pdf", bbox_inches="tight")
print("wrote", FIG / "phi_forest.pdf")
print("capability median phi:", round(cap_med, 4))
print("behavioural high median phi_broad:", round(hi.phi_broad.median(), 3))

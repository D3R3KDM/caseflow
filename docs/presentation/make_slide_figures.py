"""The guide's figures again, in the slides' light colors (#EEF0EB ground, dark ink), for the presentation deck:
docs/make_figures.py and docs/make_ai_figures.py drawn with their colors swapped, into docs/presentation/img/.

    python docs/presentation/make_slide_figures.py        # from the caseflow folder
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))
import make_figures as mf  # noqa: E402

OUT = ROOT / "docs" / "presentation" / "img"
OUT.mkdir(parents=True, exist_ok=True)
LIGHT = {"BG": "#EEF0EB", "INK": "#131517", "MUTED": "#5E636A", "GRID": "#D2D6CE"}
for k, v in LIGHT.items():
    setattr(mf, k, v)
mf.OUT = OUT
plt.rcParams.update({
    "figure.facecolor": LIGHT["BG"], "axes.facecolor": LIGHT["BG"], "savefig.facecolor": LIGHT["BG"], "text.color": LIGHT["INK"],
    "axes.labelcolor": LIGHT["MUTED"], "xtick.color": LIGHT["MUTED"], "ytick.color": LIGHT["MUTED"], "axes.edgecolor": LIGHT["GRID"],
    "axes.titlecolor": LIGHT["INK"], "grid.color": LIGHT["GRID"], "savefig.dpi": 200, "font.size": 11.5, "axes.titlesize": 13,
})
import make_ai_figures as maf  # noqa: E402  (after the swap: it takes the colors from make_figures)

if __name__ == "__main__":
    which = sys.argv[1:] or ["grid", "d3q19", "operating_point", "bank", "curves", "fields", "ranking",
                             "ai:training", "ai:parity", "ai:learning", "ai:space", "ai:dust", "ai:fields"]
    for w in which:
        mod, name = (maf, w[3:]) if w.startswith("ai:") else (mf, w)
        getattr(mod, f"fig_{name}")()

"""Figures for the RBD paper, in the style of a published ML paper.

Built on scienceplots' `science` style (inward ticks, minor ticks, no grid, hairline
spines), with Paul Tol's high-contrast qualitative pair (#004488 / #BB5566), which is
colourblind-safe and stays legible in greyscale print.

Deliberately plain: no callout arrows, no floating explanatory text, no title-case
panel headers. Anything a reader needs in prose belongs in the caption.

Numbers are transcribed from the tables in main.tex and from
PRISM/stability_*.log; nothing is recomputed here.

  fig_rank_panels  tab:ca-headline + tab:synth-rank
  fig_concepts     tab:concepts, against the random-direction null
  fig_stability    the pairwise cosine matrices behind tab:stability
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  (registers the styles)
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.5, "lines.linewidth": 1.0,
})

BLUE, RED, GREY = "#004488", "#BB5566", "#8f8f8f"   # Tol high-contrast + neutral

def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=400)
    plt.close(fig)

def panel_label(ax, s):
    ax.text(0.0, 1.04, s, transform=ax.transAxes, fontsize=8, va="bottom", ha="left")

# ------------------------------------------------------------------ rank panels
BASE_CA = 0.6397
Kc  = [1, 5, 8, 10, 20]
lo_ca = [BASE_CA + v for v in (-0.0040, -0.0317, -0.0316, -0.0342, -0.0305)]
lo_ce = [0.0089, 0.0041, 0.0040, 0.0045, 0.0048]
rb_ca = [BASE_CA + v for v in (0.0267, 0.0296, 0.0310, 0.0296, 0.0300)]
rb_ce = [0.0058, 0.0045, 0.0041, 0.0039, 0.0036]

Ks  = [1, 2, 3, 4, 8]
lo_s, lo_se = [0.5088, 0.8881, 0.9775, 0.9942, 0.9964], [0.0135, 0.0015, 0.0005, 0.0004, 0.0011]
rb_s, rb_se = [0.8897, 0.9715, 0.9962, 0.9968, 0.9992], [0.0024, 0.0001, 0.0008, 0.0010, 0.0003]

fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.0))
for ax, (K, lo, loe, rb, rbe, xlim, ylim, lab) in zip(axes, [
        (Kc, lo_ca, lo_ce, rb_ca, rb_ce, (0, 21), (0.598, 0.681), "(a) Community Alignment"),
        (Ks, lo_s,  lo_se, rb_s,  rb_se, (0.4, 8.6), (0.47, 1.02), "(b) synthetic control")]):
    if lab.startswith("(a)"):
        ax.axhline(BASE_CA, color=GREY, lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.text(20.7, BASE_CA - 0.0018, "base RM", fontsize=6.5, color=GREY,
                va="top", ha="right")
    ax.errorbar(K, lo, yerr=loe, color=BLUE, marker="o", ms=2.8, mfc="none", mew=0.9,
                lw=1.0, ls="--", capsize=1.5, elinewidth=0.6, zorder=3)
    ax.errorbar(K, rb, yerr=rbe, color=RED, marker="s", ms=2.8, lw=1.0, ls="-",
                capsize=1.5, elinewidth=0.6, zorder=3)
    ax.set_xticks(K); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("rank $K$")
    panel_label(ax, lab)
axes[0].set_ylabel("held-out accuracy")
axes[0].legend(handles=[Line2D([], [], color=BLUE, ls="--", marker="o", ms=2.8,
                               mfc="none", mew=0.9, lw=1.0, label="LoRe"),
                        Line2D([], [], color=RED, ls="-", marker="s", ms=2.8,
                               lw=1.0, label="RBD")],
               loc="center left", bbox_to_anchor=(0.03, 0.56), frameon=False,
               handlelength=2.0, labelspacing=0.3)
fig.tight_layout(w_pad=1.4)
save(fig, "fig_rank_panels")

# ------------------------------------------------------------------ concepts
TAU = 0.0318                    # null_tau, results/community_alignment/wbar_concepts.csv
cname = ["confidence", "helpfulness", "fluency", "diversity", "formatting",
         "repetition", "factuality", "sycophancy", "safety", "values", "creativity"]
wbar  = [0.0766, 0.0648, 0.0616, 0.0595, 0.0530, -0.0411, 0.0353, -0.0315, 0.0297, 0.0295, -0.0219]
head  = [0.2523, 0.4935, -0.0416, 0.4083, 0.3621, -0.3939, 0.4574, -0.4007, 0.4235, 0.4444, -0.3831]
resid = [0.0696, 0.0511, 0.0628, 0.0482, 0.0430, -0.0302, 0.0226, -0.0204, 0.0180, 0.0172, -0.0113]
y = list(range(len(cname)))[::-1]

fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.6), sharey=True,
                         gridspec_kw={"width_ratios": [1.4, 1]})
for ax, series, xlim, lab in [
        (axes[0], [(wbar, BLUE, "o", True), (resid, RED, "^", False)],
         (-0.085, 0.095), r"(a) $\bar{w}$ and its residual"),
        (axes[1], [(head, GREY, "s", True)], (-0.60, 0.60), "(b) pretrained head")]:
    ax.axvspan(-TAU, TAU, color="0.90", lw=0, zorder=0)
    ax.axvline(0, color="0.65", lw=0.5, zorder=1)
    for vals, col, mk, filled in series:
        ax.plot(vals, y, marker=mk, ms=3.2, ls="none", color=col, zorder=3,
                mfc=col if filled else "none", mew=0.9)
    ax.set_xlim(*xlim); ax.set_ylim(-0.7, len(cname) - 0.3)
    ax.set_xlabel("cosine with concept vector")
    panel_label(ax, lab)
axes[0].set_yticks(y); axes[0].set_yticklabels(cname)
axes[0].tick_params(axis="y", length=0)
fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3.2, ls="none", label=r"$\bar{w}$"),
                    Line2D([], [], color=RED, marker="^", ms=3.2, ls="none", mfc="none",
                           mew=0.9, label="residual"),
                    Line2D([], [], color=GREY, marker="s", ms=3.2, ls="none", label="pretrained head"),
                    Patch(facecolor="0.90", label="random-direction null")],
           ncol=4, loc="lower center", bbox_to_anchor=(0.52, -0.06), frameon=False,
           handlelength=1.2, columnspacing=1.5, handletextpad=0.5)
fig.tight_layout(w_pad=0.6)
save(fig, "fig_concepts")

# ------------------------------------------------------------------ stability
def sym(vals, n):
    M = np.full((n, n), np.nan)
    iu = np.triu_indices(n, 1)
    M[iu] = vals; M.T[iu] = vals
    return M

seed_1500 = sym([0.7454,0.7743,0.7783,0.7178,0.7440,0.7385,0.6977,0.6783,0.7885,
                 0.9108,0.9059,0.8318,0.8607,0.8324,0.7939,0.7567,0.9239,
                 0.9493,0.8692,0.9103,0.8810,0.8310,0.7832,0.9723,
                 0.8673,0.8987,0.8764,0.8227,0.7798,0.9724,
                 0.8158,0.8082,0.7758,0.7412,0.8771,
                 0.8339,0.7881,0.7553,0.9202,
                 0.8026,0.7491,0.8907,
                 0.7301,0.8317,
                 0.7915], 10)
seed_10k = sym([0.9999,0.9999,1.0000,1.0000,0.9999,0.9999,0.9999,0.9999,1.0000,
                0.9999,0.9999,0.9999,1.0000,0.9997,0.9998,0.9997,0.9999,
                1.0000,0.9999,1.0000,0.9998,0.9999,0.9998,0.9999,
                1.0000,0.9999,0.9999,0.9999,0.9999,1.0000,
                0.9999,0.9999,0.9999,0.9999,1.0000,
                0.9998,0.9998,0.9998,0.9999,
                1.0000,1.0000,0.9999,
                1.0000,0.9999,
                0.9999], 10)
split_10k = sym([0.8613,0.8625,0.8669,0.8627,
                 0.8673,0.8712,0.8681,
                 0.8643,0.8601,
                 0.8646], 5)

panels = [(seed_1500, "(a) 10 seeds, 1{,}500 iters", 0.8223),
          (seed_10k,  "(b) 10 seeds, 10{,}000 iters", 0.9999),
          (split_10k, "(c) 5 splits, 10{,}000 iters", 0.8649)]
fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.0))
# Perceptually uniform single hue, light -> dark: the sequential rule, without
# the pure black that "Greys" hits at the top of the range. Truncated at 0.88 so
# the darkest cell stays a colour rather than becoming ink.
_mako = sns.color_palette("mako_r", as_cmap=True)
cmap = LinearSegmentedColormap.from_list("mako_trunc", _mako(np.linspace(0.0, 0.88, 256)))
cmap.set_bad("white")
for ax, (M, title, mean) in zip(axes, panels):
    # aspect="auto" so all three axes are the same height and the titles line up;
    # cell size then differs between the 10x10 and the 5x5, which is harmless.
    im = ax.imshow(M, cmap=cmap, vmin=0.6, vmax=1.0, interpolation="nearest",
                   aspect="auto")
    ax.set_title(f"{title.replace('{,}', ',')}\nmean {mean:.4f}", fontsize=7,
                 pad=3, linespacing=1.35)
    n = M.shape[0]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(range(n), fontsize=5.5); ax.set_yticklabels(range(n), fontsize=5.5)
    ax.minorticks_off()                      # the science style adds minor ticks
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
fig.tight_layout(w_pad=1.0, rect=(0, 0, 0.90, 1))
cax = fig.add_axes([0.925, 0.16, 0.016, 0.60])
cb = fig.colorbar(im, cax=cax, ticks=[0.6, 0.7, 0.8, 0.9, 1.0])
cb.set_label("pairwise cosine", fontsize=7)
cb.ax.tick_params(labelsize=6.5, length=1.5, width=0.4)
cb.ax.minorticks_off()
cb.outline.set_linewidth(0.4)
save(fig, "fig_stability")

print("wrote fig_rank_panels, fig_concepts, fig_stability")

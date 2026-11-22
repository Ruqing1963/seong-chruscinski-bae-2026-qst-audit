#!/usr/bin/env python3
"""
make_figures.py -- generates the three figures of
"Exact separable bounds for covariant entanglement witnesses and the mirrored witnesses in 4 (x) 4"

  fig_negative_region.pdf   expectation of (4/3)1 - W[1,1,1,0] on a two-parameter family of product states
  fig_product_families.pdf  the three families of product vectors that attain the separable maximum
  fig_phase_diagram.pdf     nature of mu*1 - W for the two classes of 4 (x) 4 witnesses

All quantities are computed from the closed forms proved in the paper (Theorem 2).
Requires numpy and matplotlib.   Run:  python make_figures.py
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#d9d8d2"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def L4(al):
    a, b, c, d = al
    cands = [a, b, c, d, (a + c + 1) / 2]
    if a + 1 > max(b, d):
        cands.append(((a + 1) ** 2 - b * d) / (2 * (a + 1) - b - d))
    return max(cands)


def classI(th):
    a = (2 - math.sin(th)) / 2; b = (1 + math.cos(th)) / 2
    return [a, b, 2 - a, 1 - b]


def classII(th):
    a = (1 + math.cos(th)) / 2; b = (2 - math.sin(th)) / 2
    return [a, b, 1 - a, 2 - b]


# ------------------------------------------------------------------------------------------
# Figure 1: negative region of the printed mirror
# ------------------------------------------------------------------------------------------
def figure_negative_region():
    A, B = np.meshgrid(np.linspace(0, np.pi, 601), np.linspace(0, np.pi, 601))
    cA, sA, cB, sB = np.cos(A), np.sin(A), np.cos(B), np.sin(B)
    # e = cos A |0> + sin A |2>,  f = cos B |0> - sin B |2>,  W = W[1,1,1,0]
    W = 2 * (cA ** 2 * cB ** 2 + sA ** 2 * sB ** 2) + (cA ** 2 * sB ** 2 + sA ** 2 * cB ** 2) - (cA * cB - sA * sB) ** 2
    M = 4 / 3 - W
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f6c9b6", "#f1f0ec", "#bcd6f5", BLUE])
    fig, ax = plt.subplots(figsize=(4.3, 3.45))
    im = ax.pcolormesh(A / np.pi, B / np.pi, M, cmap=cmap, norm=TwoSlopeNorm(vmin=-1 / 6, vcenter=0, vmax=4 / 3),
                       shading="auto", rasterized=True)
    ax.contour(A / np.pi, B / np.pi, M, levels=[0], colors=[INK], linewidths=1.0)
    for pt in ((0.25, 0.25), (0.75, 0.75)):
        ax.plot(*pt, marker="*", ms=9, color=INK, mec="white", mew=0.6)
    ax.annotate(r"$-\frac{1}{6}$", (0.25, 0.25), xytext=(0.36, 0.13), fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    ax.set_xlabel(r"$A/\pi$"); ax.set_ylabel(r"$B/\pi$")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-1 / 6, 0, 1 / 3, 2 / 3, 1, 4 / 3])
    cb.ax.set_yticklabels([r"$-\frac{1}{6}$", "0", r"$\frac{1}{3}$", r"$\frac{2}{3}$", "1", r"$\frac{4}{3}$"])
    cb.set_label(r"$\langle e f|\,\frac{4}{3}\mathbf{1}-W[1,1,1,0]\,|e f\rangle$")
    cb.outline.set_linewidth(0.5)
    fig.savefig(HERE / "fig_negative_region.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------------------------------
# Figure 2: the three product families
# ------------------------------------------------------------------------------------------
def figure_families():
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.55))
    t = 0.35
    panels = [
        ("(i) basis vector", {(0, 1): 1.0}, r"$\langle W\rangle=\alpha_k$"),
        ("(ii) antipodal pair", {(0, 0): 0.25, (0, 2): 0.25, (2, 0): 0.25, (2, 2): 0.25}, r"$\langle W\rangle=\frac{a+c+1}{2}$"),
        ("(iii) adjacent pair", {(0, 0): t * (1 - t), (0, 1): t * t, (1, 0): (1 - t) ** 2, (1, 1): t * (1 - t)},
         r"$\langle W\rangle=2(a{+}1)t(1{-}t)+bt^2+d(1{-}t)^2$"),
    ]
    for ax, (title, cells, formula) in zip(axes, panels):
        for i in range(4):
            for j in range(4):
                w = cells.get((i, j), 0.0)
                col = "#ffffff" if w == 0 else plt.matplotlib.colors.to_hex(
                    np.array(plt.matplotlib.colors.to_rgb("#dbe8f9")) * (1 - min(1, w / 0.45)) +
                    np.array(plt.matplotlib.colors.to_rgb(BLUE)) * min(1, w / 0.45))
                ax.add_patch(Rectangle((j, 3 - i), 1, 1, facecolor=col, edgecolor=GRID, linewidth=0.8))
                k = (j - i) % 4
                ax.text(j + 0.5, 3 - i + 0.5, ["a", "b", "c", "d"][k], ha="center", va="center", fontsize=8,
                        color="white" if w > 0.3 else (INK if w > 0 else MUTED), style="italic")
        for j in range(4):
            ax.text(j + 0.5, 4.12, f"${j}$", ha="center", va="bottom", fontsize=7.5, color=MUTED)
            ax.text(-0.15, 3 - j + 0.5, f"${j}$", ha="right", va="center", fontsize=7.5, color=MUTED)
        ax.set_xlim(-0.6, 4.05); ax.set_ylim(-1.25, 4.9); ax.set_aspect("equal"); ax.axis("off")
        ax.text(2, 4.55, title, ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(2, -0.25, formula, ha="center", va="top", fontsize=8, color=INK)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(HERE / "fig_product_families.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------
# Figure 3: phase diagram
# ------------------------------------------------------------------------------------------
def figure_phase():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.75), sharey=True, gridspec_kw={"width_ratios": [2, 1.15]})
    specs = [(axes[0], classI, np.linspace(0, 2 * np.pi, 721), r"class I", [0, 0.5, 1, 1.5, 2]),
             (axes[1], classII, np.linspace(0, np.pi, 361), r"class II", [0, 0.5, 1])]
    for ax, fam, grid, name, ticks in specs:
        u = np.array([L4(fam(t)) for t in grid])
        lmax = np.array([max(fam(t)[0] + 1, max(fam(t)[1:])) for t in grid])
        xg = grid / np.pi
        ax.fill_between(xg, 0.9, u, color="#f6c9b6", lw=0)
        ax.fill_between(xg, u, lmax, color="#bcd6f5", lw=0)
        ax.fill_between(xg, lmax, 2.62, color="#cdeee0", lw=0)
        ax.plot(xg, u, color=ORANGE, lw=1.6)
        ax.plot(xg, lmax, color=AQUA, lw=1.6)
        ax.set_xlim(xg[0], xg[-1]); ax.set_ylim(0.9, 2.62)
        ax.set_xticks(ticks); ax.set_xlabel(r"$\theta/\pi$")
        ax.set_title(name, fontsize=9, pad=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=3, width=0.6)
    a0, a1 = axes
    a0.set_ylabel(r"mirror constant $\mu$")
    a0.axhline(4 / 3, color=INK, lw=0.8, ls=(0, (4, 2)))
    a0.plot(0, 4 / 3, marker="o", ms=5, color=INK, clip_on=False, zorder=6)
    a0.text(0.06, 4 / 3 - 0.03, r"$\mu=\frac{4}{3}$ used in the audited paper for $W[1,1,1,0]$", va="top", fontsize=7.8, color=INK)
    a0.text(1.0, 1.12, "not block positive (not a witness)", ha="center", fontsize=8, color=INK)
    a0.text(1.45, 1.9, "decomposable witness", ha="center", fontsize=8, color=INK)
    a0.text(0.5, 2.4, "positive semidefinite", ha="center", fontsize=8, color=INK)
    a0.text(1.98, 1.53, r"$u_W=\frac{3}{2}$", ha="right", va="bottom", fontsize=8.5, color=INK)
    a0.text(1.1, 2.19, r"$\lambda_{\max}(W)$", ha="right", va="bottom", fontsize=8.5, color=INK)
    a1.text(0.5, 1.12, "not block positive", ha="center", fontsize=8, color=INK)
    a1.text(0.17, 1.6, "decomp.\nwitness", ha="center", fontsize=8, color=INK)
    a1.text(0.62, 2.2, "positive semidefinite", ha="center", fontsize=8, color=INK)
    a1.text(0.86, 1.3, r"$u_W=\lambda_{\max}$", ha="left", va="bottom", fontsize=8.5, color=INK, rotation=0)
    fig.subplots_adjust(wspace=0.14)
    fig.savefig(HERE / "fig_phase_diagram.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_negative_region()
    figure_families()
    figure_phase()
    print("wrote fig_negative_region.pdf, fig_product_families.pdf, fig_phase_diagram.pdf")

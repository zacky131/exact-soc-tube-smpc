"""Schematic of plan-dependent tightening under multiplicative noise.

Scalar tube system (illustrative, not one of the paper's examples):
  x+ = a x + b u + (a1 x + b1 u) theta + w,  u = v + K (x - z)
  nominal z+ = a z + b v,  Var(e+) = (Phi0^2 + Phi1^2) Var(e) + (a1 z + b1 v)^2 + W
Constraint P(x_t <= bmax) >= 0.9, Cantelli factor kappa = 3.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

a, b, a1, b1, K, W = 1.1, 1.0, 0.10, 0.25, -0.6, 0.01
Phi0, Phi1 = a + b * K, a1 + b1 * K
kappa, bmax, T = 3.0, 4.0, 12


def plan_from_path(r):
    """Inputs v_t that make the nominal state follow the path r exactly."""
    return np.array([(r[t + 1] - a * r[t]) / b for t in range(len(r) - 1)])


def variances(r, v, coupling=True):
    s = [0.0]
    for t in range(len(v)):
        c = (a1 * r[t] + b1 * v[t]) ** 2 if coupling else 0.0
        s.append((Phi0**2 + Phi1**2) * s[-1] + c + W)
    return np.array(s)


def path(level, rise):
    t = np.arange(T + 1)
    return level * np.clip(t / rise, 0, 1) ** 1.0


# (a) aggressive plan accepted by the coupling-agnostic design
r_a = path(3.55, 2)
v_a = plan_from_path(r_a)
sig_bel = np.sqrt(variances(r_a, v_a, coupling=False))
sig_true_a = np.sqrt(variances(r_a, v_a, coupling=True))

# (b) proposed: exact plan-dependent tightening; largest level that stays feasible
rise_b = 5
lo, hi = 0.0, 4.0
for _ in range(60):
    mid = (lo + hi) / 2
    r = path(mid, rise_b)
    s = np.sqrt(variances(r, plan_from_path(r)))
    lo, hi = (mid, hi) if np.all(r + kappa * s <= bmax + 1e-12) else (lo, mid)
r_b = path(lo, rise_b)
sig_b = np.sqrt(variances(r_b, plan_from_path(r_b)))

# ---------------------------------------------------------------- plotting
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
BLUE, GREEN, PURPLE, RED = "#2b5d9b", "#5a9e3c", "#9b59a6", "#c0392b"
ymin = -1.6


def panel(ax, r, sig_tube, title, sig_true=None):
    t = np.arange(T + 1)
    w = 0.62
    for k in t:
        ax.add_patch(Rectangle((k - w / 2, ymin), w, bmax - ymin, facecolor="none",
                               edgecolor=BLUE, alpha=0.35, hatch="////", lw=0))
        ax.plot([k - w / 2, k + w / 2], [bmax, bmax], color=BLUE, lw=1.6)
        tb = bmax - kappa * sig_tube[k]
        ax.plot([k - w / 2, k + w / 2], [tb, tb], color=PURPLE, lw=1.6)
        ax.add_patch(Ellipse((k, r[k]), 0.5, max(2 * kappa * sig_tube[k], 0.06), fill=False,
                             ls=(0, (1, 1.5)), ec=GREEN, lw=1.3))
        if sig_true is not None and r[k] + kappa * sig_true[k] > bmax + 1e-9:
            ax.add_patch(Ellipse((k, r[k]), 0.5, 2 * kappa * sig_true[k], fill=False,
                                 ls="--", ec=RED, lw=1.1))
    for k in t[:-1]:
        ax.plot([k + w / 2, k + 1 - w / 2], [bmax - kappa * sig_tube[k], bmax - kappa * sig_tube[k + 1]],
                color=PURPLE, ls=":", lw=1.0)
    ax.plot(t, r, "k--", lw=1.2, marker="o", ms=3.5)
    ax.set_xlim(-0.6, T + 0.6)
    ax.set_ylim(ymin, 5.6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(r"prediction step $t$")
    ax.set_title(title, fontsize=9.5)


fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
panel(axs[0], r_a, sig_bel, r"(a) Coupling ignored: tube fixed offline", sig_true=sig_true_a)
panel(axs[1], r_b, sig_b, r"(b) Proposed: tube depends on the plan (exact SOC)")
axs[0].set_ylabel(r"$h^\top x_t$")
axs[0].annotate("believed safe,\nactually violates", xy=(2.25, r_a[2] + kappa * sig_true_a[2] - 0.3),
                xytext=(3.6, 5.15), fontsize=8, color=RED, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
axs[1].annotate("tightening follows the plan:\nlarger inputs, wider tube", xy=(5.0, r_b[5] + kappa * sig_b[5] + 0.05),
                xytext=(5.6, 5.15), fontsize=8, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.8))

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [Line2D([], [], color=BLUE, lw=1.6, label=r"constraint $h^\top x\leq b$"),
           Line2D([], [], color=GREEN, ls=(0, (1, 1.5)), lw=1.3, label=r"tube $\pm\kappa\sigma_t$ used by design"),
           Line2D([], [], color=PURPLE, lw=1.6, label=r"tightened bound $b-\kappa\sigma_t$"),
           Line2D([], [], color="k", ls="--", marker="o", ms=3.5, lw=1.2, label=r"nominal plan $h^\top z_t$"),
           Line2D([], [], color=RED, ls="--", lw=1.1, label=r"true tube (with coupling)")]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, frameon=False,
           bbox_to_anchor=(0.5, -0.02), handlelength=2.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("fig_tube.pdf", bbox_inches="tight")
fig.savefig("fig_tube.png", dpi=220, bbox_inches="tight")
print("level a", r_a[-1], "level b", r_b[-1])
print("max violation a (true)", np.max(r_a + kappa * sig_true_a - bmax))

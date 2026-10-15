"""Figures of the paper, read only from the stored result files; written to figures/.

Fig. 1 (fig_mechanism.pdf): first-step plans, SOC vs coupling-agnostic design (results/EXP009_illustrations, part a).
Fig. 2: block diagram (TikZ, in the manuscript source; not generated here).
Fig. 3 (fig_region.pdf): feasible initial states, benchmark of Section 6.1, N = 10, SOC vs LMI with eps = 0.9
  (results/EXP003_pilot_v4_scaled/part1_grid.csv).
Fig. 4 (fig_trajectories.pdf): closed-loop sample paths and violation frequency (results/EXP009_illustrations, part b).
Fig. 5 (fig_coupling.pdf): worst-step violation vs initial transient r, SOC vs coupling-agnostic design, upper limits of
  the 95% Clopper-Pearson interval (results/EXP002_mechanism_pilot_v4).
Fig. 6 (fig_timing.pdf): median solve times (results/EXP003b_final).
Usage: python3 scripts/make_figures_scl.py            (all figures of the paper)
       python3 scripts/make_figures_scl.py two-sided  (Fig. 5 with two-sided intervals, as in the first submission)
"""
import csv, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"; OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#6b6a66", "#d9d8d4"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.family": "serif", "pdf.fonttype": 42})


def fig_region():
    rows = list(csv.DictReader(open(ROOT / "results/EXP003_pilot_v4_scaled/part1_grid.csv")))
    x1 = np.array([float(r["x1"]) for r in rows]); x2 = np.array([float(r["x2"]) for r in rows])
    cx = np.array([r["cx_class"] for r in rows]); fs = np.array([r["fs0.9_class"] for r in rows])
    both = (cx == "verified_feasible") & (fs == "verified_feasible")
    cx_only = (cx == "verified_feasible") & (fs == "certified_infeasible")
    fs_unres = (cx == "verified_feasible") & (fs == "unresolved")
    infeas = cx == "certified_infeasible"
    assert not np.any((fs == "verified_feasible") & (cx != "verified_feasible")), "containment violated in data"
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.scatter(x1[infeas], x2[infeas], s=10, marker="x", color=GRID, linewidths=0.8, label="infeasible (both)")
    ax.scatter(x1[both], x2[both], s=14, marker="o", color=BLUE, edgecolors="white", linewidths=0.4, label="feasible: SOC and LMI")
    ax.scatter(x1[cx_only], x2[cx_only], s=22, marker="s", color=ORANGE, edgecolors="white", linewidths=0.4, label="feasible: SOC only")
    if fs_unres.any():
        ax.scatter(x1[fs_unres], x2[fs_unres], s=22, marker="D", facecolors="none", edgecolors=AQUA, linewidths=1.0, label="SOC feasible, LMI unresolved")
    xs = np.linspace(-6, 10, 2)
    ax.plot(xs, 2.3 + 2.0 * xs, color=MUTED, lw=1.0, ls="--")  # h'x = 1 with h = [-2, 1]/2.3  <=>  x2 = 2.3 + 2 x1
    ax.set_xlim(-6.6, 10.6); ax.set_ylim(-6.6, 10.6)
    ax.set_xlabel(r"$x_{0,1}$"); ax.set_ylabel(r"$x_{0,2}$")
    ax.text(-5.6, 9.2, r"$h^\top x=1$", color=MUTED, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(); fig.savefig(OUT / "fig_region.pdf"); plt.close(fig)
    return dict(both=int(both.sum()), cx_only=int(cx_only.sum()), fs_unresolved=int(fs_unres.sum()), infeasible=int(infeas.sum()), total=len(rows))


def cp(rate, n):
    k = int(round(rate * n))
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def fig_coupling(upper_only=False, name="fig_coupling.pdf"):
    """upper_only=True (revision-v7, R1-W3): bars show only the upper limit of the 95 % Clopper-Pearson interval."""
    s = json.load(open(ROOT / "results/EXP002_mechanism_pilot_v4/summary.json"))
    n = 2000
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.35), sharey=True)
    for ax, sc in zip(axes, (1.0, 1.5)):
        cells = sorted([c for c in s["cells"] if c["s"] == sc], key=lambda c: c["r"])
        for meth, col, mk, ls, lab, off in (("CX-A", BLUE, "o", "-", "SOC design", -0.3), ("CA-A", ORANGE, "s", "--", "coupling-agnostic", 0.3)):
            rr, vv, lo, hi, inf = [], [], [], [], []
            for c in cells:
                d = c[meth]
                if not d.get("feasible_at_k0"):
                    inf.append(c["r"]); continue
                l, h = cp(d["max_k"], n)
                rr.append(c["r"] + off); vv.append(100 * d["max_k"]); lo.append(0.0 if upper_only else 100 * (d["max_k"] - l)); hi.append(100 * (h - d["max_k"]))
            ax.errorbar(rr, vv, yerr=[lo, hi], color=col, marker=mk, ms=4, lw=1.2, ls=ls, capsize=1.5, elinewidth=0.8, label=lab)
            for r in inf:
                ax.text(r, 1.0, "infeas.", color=col, fontsize=6, ha="center", rotation=90, va="bottom")
        ax.axhline(10, color=MUTED, lw=0.8, ls=":")
        ax.set_title(f"$\\beta={sc:g}$", color=INK); ax.set_xlabel("initial transient $r$")
        ax.set_xticks([0, 4, 8, 12, 16, 20]); ax.grid(axis="y", color=GRID, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("worst-step violation [%]")
    axes[0].text(0, 10.6, "design risk $1-p$", color=MUTED, fontsize=6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT / name); plt.close(fig)


def fig_mechanism():
    """Benchmark A, x0 = (0,0,16): first-step plans; Cantelli bound with true vs believed std; MC 90% quantile."""
    d = json.load(open(ROOT / "results/EXP009_illustrations/data.json"))["a"]
    k, b, N = d["kappa"], d["b"], d["N"]; t = np.arange(N + 1)
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.5), sharey=True)
    for ax, name, title in zip(axes, ("CX", "CA"), ("SOC design", "coupling-agnostic design")):
        x = d[name]; hz = np.array(x["hz"])
        ax.plot(t, hz, color=MUTED, lw=1.2, label=r"nominal $h^\top z_t$")
        ax.plot(t, hz + k * np.array(x["std_true"]), color=ORANGE, lw=1.4, ls="--", label=r"true bound $h^\top z_t+\kappa\sigma_t$")
        if name == "CA":
            ax.plot(t, hz + k * np.array(x["std_believed"]), color=BLUE, lw=1.4, ls="-.", label="bound believed by design")
        else:
            ax.plot([], [], color=BLUE, lw=1.4, ls="-.", label="bound believed by design")
        ax.plot(t, x["mc_q90"], color=AQUA, marker="o", ms=3, lw=0, label="Monte Carlo 90% quantile of $h^\\top x_t$")
        ax.axhline(b, color=INK, lw=0.8, ls=":")
        ax.set_title(title, color=INK); ax.set_xlabel("prediction step $t$")
        ax.grid(axis="y", color=GRID, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("$x_1$"); axes[0].text(0.3, b + 0.6, r"constraint $x_1\leq 12$", color=INK, fontsize=6)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.8)); fig.savefig(OUT / "fig_mechanism.pdf"); plt.close(fig)


def fig_trajectories():
    """Benchmark D, x0 = (0, 1.5): 50 closed-loop paths of h'x_k and per-step violation frequency (2000 runs)."""
    d = json.load(open(ROOT / "results/EXP009_illustrations/data.json"))["b"]
    meth = (("CX-A", "SOC scheme", BLUE), ("CX-DF", "SOC scheme, binary initialisation", AQUA), ("NOM", "certainty-equivalent MPC", ORANGE))
    fig = plt.figure(figsize=(7.0, 3.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.8])
    for i, (m, lab, col) in enumerate(meth):
        ax = fig.add_subplot(gs[0, i])
        for pth in d[m]["paths"]:
            ax.plot(np.arange(len(pth)), pth, color=col, lw=0.5, alpha=0.45)
        ax.axhline(1.0, color=INK, lw=0.9, ls=":")
        ax.set_title(lab, color=INK); ax.set_ylim(-1.2, 1.6)
        ax.set_xlabel("time step $k$")
        if i == 0:
            ax.set_ylabel("$h^\\top x_k$"); ax.text(30, 1.07, "$h^\\top x=1$", color=INK, fontsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    ax = fig.add_subplot(gs[1, :])
    markers = {"CX-A": "o", "CX-DF": "D", "NOM": "s"}
    for m, lab, col in meth:
        v = 100 * np.array(d[m]["viol_freq"]); k = np.arange(len(v))
        ax.plot(k, v, color=col, lw=1.2, marker=markers[m], ms=2.5, label=lab)
    ax.axhline(10, color=MUTED, lw=0.8, ls=":"); ax.text(0.5, 11, "admissible level $1-p$", color=MUTED, fontsize=6)
    ax.set_yscale("symlog", linthresh=0.5, linscale=0.4); ax.set_ylim(0, 70)
    ax.set_ylabel("violation freq. [%]"); ax.set_xlabel("time step $k$")
    ax.set_yticks([0, 0.5, 1, 5, 10, 50]); ax.set_yticklabels(["0", "0.5", "1", "5", "10", "50"])
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_trajectories.pdf"); plt.close(fig)


def fig_timing():
    """Median Clarabel solve time vs horizon N (EXP003b final, 100 instances per size)."""
    s = json.load(open(ROOT / "results/EXP003b_final/summary.json"))["B"]
    meth = (("CX", "SOC, full horizon", BLUE, "o", "-"), ("DM3", "SOC, dual mode $N_c=3$", AQUA, "D", "-."),
            ("DM5", "SOC, dual mode $N_c=5$", YELLOW, "^", ":"), ("FS0.5", "LMI formulation", ORANGE, "s", "--"))
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.4), sharey=True)
    for ax, n in zip(axes, (4, 10)):
        rows = sorted([r for r in s if r["n"] == n], key=lambda r: r["N"])
        for m, lab, col, mk, ls in meth:
            ax.plot([r["N"] for r in rows], [r[m]["p50"] for r in rows], color=col, marker=mk, ms=3.5, ls=ls, lw=1.2, label=lab)
        ax.set_yscale("log"); ax.set_title(f"$n={n}$", color=INK); ax.set_xlabel("horizon $N$"); ax.set_xticks([10, 20, 30])
        ax.grid(axis="y", color=GRID, lw=0.5, which="both")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("median solve time [ms]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.8)); fig.savefig(OUT / "fig_timing.pdf"); plt.close(fig)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "two-sided":
    fig_coupling(upper_only=False, name="fig_coupling_two_sided.pdf"); print("fig_coupling_two_sided.pdf written"); sys.exit(0)
if __name__ == "__main__":
    print(fig_region()); fig_coupling(upper_only=True); fig_mechanism(); fig_trajectories(); fig_timing(); print("figures written to", OUT)

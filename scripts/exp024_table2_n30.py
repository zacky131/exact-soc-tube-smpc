"""EXP024 — Table 2 (Section 6.3) rerun with the full horizon N = 30 (revision-v8).

Closed loop of Algorithm 1 (CX-A, TubeController mode "A") on benchmark D from x0 = (-0.1, 1.2) with the terminal
ingredients of Assumption 3 (designed as in EXP010/EXP016: best over the (lambda, H0) grid, 2 % margin), for the
Cantelli factor kappa = 3 and the normal quantile kappa = Phi^-1(0.9) = 1.28, and theta ~ Gaussian / Laplace /
Student-t5 / Student-t3 (unit variance, cxsmpc.benchmarks.theta_sampler). 2000 runs, run length T = 40, Table 1 seeds:
theta_r = theta_sampler(kind)(rng(7000 + 1000 * ix + r), T) with ix = 0, i.e. exactly rng.standard_normal(T) for
the Gaussian case, so the (Gaussian, Cantelli) cell must reproduce the Table 1 row "SOC scheme, terminal, N = 30"
at x0 = (-0.1, 1.2) (EXP022, row CX-A-terminal-N30).

The closed-loop code is that of scripts/exp010_terminal_rerun.py run_cell (unchanged logic); the only additions are
the theta distribution and a record of terminal-constraint activity: for every verified solve, the terminal state
constraint z_N' Ef z_N <= 1 and the trace constraint tr(S Sigma_N) <= gamma_bar are counted as active if they hold
with a relative slack below ACTIVE_TOL.

Output: results/EXP024_table2_n30/summary.json
Usage:  python3 scripts/exp024_table2_n30.py [--runs 2000] [--workers 8]
"""
import argparse, importlib.util, json, pathlib, sys, warnings
from multiprocessing import Pool

import numpy as np
from scipy.stats import beta, norm

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ACTIVE_TOL = 1e-3
CFG = dict(N=30, Nc=None, p=0.9, seed=7000, ix=0, x0=[-0.1, 1.2], terminal_margin=0.02, T=40, runs=2000,
           kappas=["cantelli", "normal"], tails=["gauss", "laplace", "t5", "t3"], active_tol=ACTIVE_TOL)

_spec = importlib.util.spec_from_file_location("exp010", str(ROOT / "scripts" / "exp010_terminal_rerun.py"))
e10 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e10)


class ActivityProbe:
    """Wraps a CXProblem; records terminal-constraint activity of every verified solve."""

    def __init__(self, pr, Ef, S, gamma):
        self.pr, self.model, self.Ef, self.S, self.gamma = pr, pr.model, Ef, S, gamma
        self.count = dict(solves=0, zN_active=0, trace_active=0, either_active=0)

    def solve(self, z0, Sigma0, **kw):
        r = self.pr.solve(z0, Sigma0, **kw)
        if r["class"] == "verified_feasible":
            m = self.model; z, v = r["z"], r["v"]; Sig = np.array(Sigma0, float)
            for t in range(z.shape[1] - 1):
                xi = np.r_[z[:, t], v[:, t]]
                Sig = m.L(Sig) + m.D(np.outer(xi, xi)) + m.W
            a1 = float(z[:, -1] @ self.Ef @ z[:, -1]) >= 1 - ACTIVE_TOL
            a2 = float(np.trace(self.S @ Sig)) >= (1 - ACTIVE_TOL) * self.gamma
            self.count["solves"] += 1; self.count["zN_active"] += a1; self.count["trace_active"] += a2
            self.count["either_active"] += a1 or a2
        return r


def run_cell(task):
    kap, tail, runs = task
    e10.setup()
    from cxsmpc.benchmarks import bench_D, theta_sampler
    from cxsmpc.closedloop import TubeController
    from cxsmpc.ocp import CXProblem
    D = bench_D(); md = D.meta; h = md["h"]
    con = dict(kind="state", h=h, b=1.0, p=CFG["p"])
    if kap == "normal":
        con["kappa"] = float(norm.ppf(CFG["p"]))
    td = e10.best_terminal(D, [con], h, CFG["terminal_margin"])
    term = dict(rho_L=float(D.rho_L()), lam=float(td["lam"]), alpha=float(td["c"]), gamma=float(td["gamma"]))
    pr = CXProblem(D, CFG["N"], md["Q"], md["R"], md["P"], [con], Ef=td["Ef"], trace_terminal=(td["S"], td["gamma"]))
    x0 = np.array(CFG["x0"], float)
    if pr.solve(x0, np.zeros((2, 2)))["class"] != "verified_feasible":
        return dict(kappa=kap, tail=tail, status="infeasible_at_x0", terminal=term)
    probe = ActivityProbe(pr, td["Ef"], td["S"], td["gamma"])
    ctrl = TubeController(probe, "A"); sampler = theta_sampler(tail)
    T = CFG["T"]; viol = np.zeros(T + 1); costs = []; logs = {}
    for r in range(runs):
        th = sampler(np.random.default_rng(CFG["seed"] + 1000 * CFG["ix"] + r), T)  # gauss: = rng.standard_normal(T)
        x = x0.copy(); ctrl.reset(x); J = 0.0
        for k in range(T):
            u, ok = ctrl.step(x, k)
            J += float(x @ md["Q"] @ x + u @ md["R"] @ u)
            x = D.A[0] @ x + D.B[0] @ u + (D.A[1] @ x + D.B[1] @ u) * th[k]
            viol[k + 1] += h @ x > 1.0
        costs.append(J)
        for kk, vv in ctrl.log.items():
            logs[kk] = logs.get(kk, 0) + vv
    vf = viol / runs; kmax = int(round(vf.max() * runs))
    return dict(kappa=kap, tail=tail, status="ok", runs=runs, T=T, terminal=term,
                max_k_violation=float(vf.max()), argmax_k=int(vf.argmax()),
                max_k_cp95_upper=float(beta.ppf(0.975, kmax + 1, runs - kmax)) if kmax < runs else 1.0,
                cost_mean=float(np.mean(costs)), cost_std=float(np.std(costs, ddof=1)), logs=logs,
                activity=probe.count, viol_freq=vf.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=CFG["runs"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "EXP024_table2_n30"))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from cxsmpc.provenance import write_provenance
    write_provenance(out, dict(CFG, runs=a.runs), __file__)
    tasks = [(k, t, a.runs) for k in CFG["kappas"] for t in CFG["tails"]]
    res = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_cell, tasks):
            res.append(r)
            print("done", r["kappa"], r["tail"], r["status"],
                  {k: r[k] for k in ("max_k_violation", "max_k_cp95_upper", "cost_mean") if k in r}, flush=True)
    (out / "summary.json").write_text(json.dumps(dict(config=dict(CFG, runs=a.runs), cells=res), indent=1))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

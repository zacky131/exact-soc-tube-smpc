"""EXP010 — closed loop on benchmark D WITH the terminal ingredients of Assumption 3 (the scheme covered by Theorems 2-3).

Prepared for the revision (reviewer request: Table 1 / Fig. 3 rerun with terminal constraints). Results enter the paper
as ledger record E010 (added after the freeze).

Protocol (matches EXP001 final / EXP009 part (b) so that results are directly comparable with Table 1 and Fig. 3):
  benchmark D, N = 10, p = 0.9, theta ~ N(0, 1), matched noise sequences
  theta_r = rng(7000 + 1000 * ix + r).standard_normal(T) for the ix-th initial state (identical to EXP001 / Table 1), methods CX-A (Algorithm 1) and CX-DF (binary initialisation),
  terminal ingredients designed as in EXP004 (trace terminal, 2 % margin, best over the (lambda, H0) grid).
  Initial states at which the k = 0 problem is not verified feasible are reported as "infeasible_at_x0" (no run).

Same-length comparison for Section 6.3 (reviewer request): run with --T 40 and compare the x0 = (-0.1, 1.2) cell
with Table 1 (same T, same noise), or add --no-terminal to rerun the terminal-free scheme with any T.

Usage (also --N and --Nc for the dual-mode variant with a longer horizon):
  python scripts/exp010_terminal_rerun.py                       # full run (2000 runs x 4 x0 x 2 methods, T = 40)
  python scripts/exp010_terminal_rerun.py --runs 20 --T 10 --out /tmp/exp010_smoke   # smoke test
"""
import argparse, json, pathlib, sys, warnings
from multiprocessing import Pool

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def setup(obj_scale=1e-3, fallback_max_iters=20000):
    import cxsmpc.ocp as ocp
    ocp.OBJ_SCALE = obj_scale; ocp.FALLBACK_MAX_ITERS = fallback_max_iters


def best_terminal(mdl, cons, h, margin):
    """Same terminal design as EXP002/EXP004 (best over the (lambda, H0) grid, largest terminal level)."""
    from cxsmpc.terminal import design_terminal
    best = None
    for lamf in (0.1, 0.25, 0.5, 0.75):
        for H0 in (np.eye(mdl.n), np.outer(h, h) + 1e-3 * np.eye(mdl.n)):
            d = design_terminal(mdl, cons, lam=mdl.rho_L() + lamf * (1 - mdl.rho_L()), H0=H0, margin=margin)
            if d and (best is None or d["c"] > best["c"]):
                best = d
    return best


def run_cell(task):
    method, ix, x0, use_terminal, cfg = task
    setup()
    from cxsmpc.benchmarks import bench_D
    from cxsmpc.closedloop import TubeController
    from cxsmpc.ocp import CXProblem, DualModeCXProblem
    D = bench_D(); md = D.meta; h = md["h"]
    cons = [dict(kind="state", h=h, b=1.0, p=cfg["p"])]
    kw = {}
    if use_terminal:
        td = best_terminal(D, cons, h, cfg["terminal_margin"])
        if td is None:
            return dict(method=method, x0=x0, terminal=use_terminal, status="no terminal set")
        kw = dict(Ef=td["Ef"], trace_terminal=(td["S"], td["gamma"]))
    if cfg.get("Nc") is not None:  # dual-mode variant (Proposition 4), exact with the trace terminal collapsed on z_Nc
        pr = DualModeCXProblem(D, cfg["N"], md["Q"], md["R"], md["P"], cons, Nc=cfg["Nc"], **kw)
    else:
        pr = CXProblem(D, cfg["N"], md["Q"], md["R"], md["P"], cons, **kw)
    if pr.solve(np.array(x0, float), np.zeros((2, 2)))["class"] != "verified_feasible":
        return dict(method=method, x0=x0, terminal=use_terminal, status="infeasible_at_x0")
    ctrl = TubeController(pr, "A" if method == "CX-A" else "DF")
    T, runs = cfg["T"], cfg["runs"]
    viol = np.zeros(T + 1); costs = []; logs = {}; paths = []
    for r in range(runs):
        th = np.random.default_rng(cfg["seed"] + 1000 * ix + r).standard_normal(T)  # = EXP001 seeds
        x = np.array(x0, float); ctrl.reset(x); J = 0.0; hv = [float(h @ x)]
        for k in range(T):
            u, ok = ctrl.step(x, k)
            J += float(x @ md["Q"] @ x + u @ md["R"] @ u)
            x = D.A[0] @ x + D.B[0] @ u + (D.A[1] @ x + D.B[1] @ u) * th[k]
            hv.append(float(h @ x)); viol[k + 1] += h @ x > 1.0
        costs.append(J)
        if r < cfg["paths"]:
            paths.append(hv)
        for kk, vv in ctrl.log.items():
            logs[kk] = logs.get(kk, 0) + vv
    vf = viol / runs; kmax = int(round(vf.max() * runs))
    return dict(method=method, x0=x0, terminal=use_terminal, status="ok", runs=runs, T=T,
                max_k_violation=float(vf.max()), argmax_k=int(vf.argmax()),
                max_k_cp95_upper=float(beta.ppf(0.975, kmax + 1, runs - kmax)) if kmax < runs else 1.0,
                cost_mean=float(np.mean(costs)), cost_std=float(np.std(costs, ddof=1)) if runs > 1 else 0.0,
                viol_freq=vf.tolist(), paths=paths, logs=logs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--paths", type=int, default=50)
    ap.add_argument("--methods", default="CX-A,CX-DF")
    ap.add_argument("--no-terminal", action="store_true", help="also run the terminal-free scheme (same-length comparison)")
    ap.add_argument("--terminal", choices=("yes", "no", "both"), default=None,
                    help="which variant to run (default: yes, or both with --no-terminal)")
    ap.add_argument("--out", default=str(ROOT / "results" / "EXP010_terminal_rerun"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--N", type=int, default=10, help="prediction horizon (Table 1: 10)")
    ap.add_argument("--Nc", type=int, default=None, help="dual-mode free moves (Proposition 4); default: full horizon")
    a = ap.parse_args()
    cfg = dict(N=a.N, Nc=a.Nc, p=0.9, seed=7000, terminal_margin=0.02, T=a.T, runs=a.runs, paths=a.paths,
               x0=[[-0.1, 1.2], [-0.1, 1.3], [-0.1, 1.4], [0.0, 1.5]])
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from cxsmpc.provenance import write_provenance
    write_provenance(out, cfg, __file__)
    mode = a.terminal or ("both" if a.no_terminal else "yes")
    flags = {"yes": [True], "no": [False], "both": [True, False]}[mode]
    tasks = [(m, ix, x0, f, cfg) for f in flags for m in a.methods.split(",") for ix, x0 in enumerate(cfg["x0"])]
    with Pool(a.workers) as pool:
        res = pool.map(run_cell, tasks)
    (out / "summary.json").write_text(json.dumps(dict(config=cfg, cells=res)))
    for r in res:
        print(r["method"], r["x0"], "terminal" if r["terminal"] else "no-terminal", r["status"],
              {k: r[k] for k in ("max_k_violation", "max_k_cp95_upper", "cost_mean") if k in r})
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

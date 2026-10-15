"""EXP025 — terminal-constraint activity at N = 30 for all four initial states of Table 1 (revision-v9).

Reruns the Table 1 rows "SOC scheme, terminal, N = 30" (Algorithm 1, CX-A) and "SOC scheme, binary init., terminal,
N = 30" (CX-DF) with the closed-loop code of scripts/exp010_terminal_rerun.py run_cell (Table 1 seeds
7000 + 1000 ix + r, 2000 runs, T = 40, Gaussian theta, Cantelli kappa = 3), and records for every applied verified
plan whether the terminal-set constraint z_N' Ef z_N <= 1 and the trace bound tr(S Sigma_N) <= gamma_bar are active
(relative slack below 1e-3, as in EXP024), per closed-loop step k. Each terminal cell must reproduce the Table 1 row
(EXP022) exactly. CX-DF without terminal constraints at N = 30 is added as the cost reference for binary
initialisation (for Algorithm 1 the reference is EXP017).

Output: results/EXP025_terminal_activity_n30/summary.json
Usage:  python3 scripts/exp025_terminal_activity_n30.py [--runs 2000] [--workers 12]
"""
import argparse, importlib.util, json, pathlib, sys, warnings
from multiprocessing import Pool

import numpy as np
from scipy.stats import beta

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
X0 = [[-0.1, 1.2], [-0.1, 1.3], [-0.1, 1.4], [0.0, 1.5]]
CFG = dict(N=30, p=0.9, seed=7000, terminal_margin=0.02, T=40, runs=2000, x0=X0, active_tol=1e-3)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "scripts" / file))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


e10 = _load("exp010", "exp010_terminal_rerun.py")
e24 = _load("exp024", "exp024_table2_n30.py")


def run_cell(task):
    method, ix, use_terminal, runs = task
    e10.setup()
    from cxsmpc.benchmarks import bench_D
    from cxsmpc.closedloop import TubeController
    from cxsmpc.ocp import CXProblem
    D = bench_D(); md = D.meta; h = md["h"]
    cons = [dict(kind="state", h=h, b=1.0, p=CFG["p"])]
    x0 = np.array(X0[ix], float); T = CFG["T"]
    kw, td = {}, None
    if use_terminal:
        td = e10.best_terminal(D, cons, h, CFG["terminal_margin"])
        kw = dict(Ef=td["Ef"], trace_terminal=(td["S"], td["gamma"]))
    pr = CXProblem(D, CFG["N"], md["Q"], md["R"], md["P"], cons, **kw)
    if pr.solve(x0, np.zeros((2, 2)))["class"] != "verified_feasible":
        return dict(method=method, x0=X0[ix], terminal=use_terminal, status="infeasible_at_x0")
    probe = e24.ActivityProbe(pr, td["Ef"], td["S"], td["gamma"]) if use_terminal else pr
    ctrl = TubeController(probe, "A" if method == "CX-A" else "DF")
    viol = np.zeros(T + 1); costs = []; logs = {}
    zN_k = np.zeros(T, int); tr_k = np.zeros(T, int); solves_k = np.zeros(T, int)
    for r in range(runs):
        th = np.random.default_rng(CFG["seed"] + 1000 * ix + r).standard_normal(T)  # = Table 1 seeds
        x = x0.copy(); ctrl.reset(x); J = 0.0
        for k in range(T):
            if use_terminal:
                c0 = dict(probe.count)
            u, ok = ctrl.step(x, k)
            if use_terminal:
                zN_k[k] += probe.count["zN_active"] - c0["zN_active"]
                tr_k[k] += probe.count["trace_active"] - c0["trace_active"]
                solves_k[k] += probe.count["solves"] - c0["solves"]
            J += float(x @ md["Q"] @ x + u @ md["R"] @ u)
            x = D.A[0] @ x + D.B[0] @ u + (D.A[1] @ x + D.B[1] @ u) * th[k]
            viol[k + 1] += h @ x > 1.0
        costs.append(J)
        for kk, vv in ctrl.log.items():
            logs[kk] = logs.get(kk, 0) + vv
    vf = viol / runs; kmax = int(round(vf.max() * runs))
    out = dict(method=method, x0=X0[ix], terminal=use_terminal, status="ok", runs=runs, T=T,
               max_k_violation=float(vf.max()), argmax_k=int(vf.argmax()),
               max_k_cp95_upper=float(beta.ppf(0.975, kmax + 1, runs - kmax)) if kmax < runs else 1.0,
               cost_mean=float(np.mean(costs)), cost_std=float(np.std(costs, ddof=1)), logs=logs)
    if use_terminal:
        out.update(activity=probe.count, zN_active_per_k=zN_k.tolist(), trace_active_per_k=tr_k.tolist(),
                   verified_solves_per_k=solves_k.tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=CFG["runs"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "results" / "EXP025_terminal_activity_n30"))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from cxsmpc.provenance import write_provenance
    write_provenance(out, dict(CFG, runs=a.runs), __file__)
    tasks = [(m, ix, True, a.runs) for m in ("CX-A", "CX-DF") for ix in range(4)] + [("CX-DF", ix, False, a.runs) for ix in range(4)]
    res = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_cell, tasks):
            res.append(r)
            print("done", r["method"], r["x0"], r["terminal"], r["status"], {k: r[k] for k in ("max_k_violation", "cost_mean") if k in r}, flush=True)
    (out / "summary.json").write_text(json.dumps(dict(config=dict(CFG, runs=a.runs), cells=res), indent=1))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

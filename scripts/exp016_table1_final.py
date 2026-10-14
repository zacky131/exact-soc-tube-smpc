"""EXP016 — final run of ALL Table 1 rows from one code version (revision-v4).

Every row of Table 1 is produced by this single invocation, with the noise seeds of Table 1
(theta_r = rng(7000 + 1000 * ix + r).standard_normal(40) for the ix-th initial state), 2000 runs per row, T = 40:
  via scripts/exp001_closed_loop.py run_task (unchanged):
    NOM   certainty-equivalent MPC (N = 10)
    CX-A  SOC scheme, no terminal constraints (N = 10)
    CX-DF SOC scheme, binary initialisation, no terminal constraints (N = 10)
    FS-DF LMI formulation [FS16], binary initialisation, eps = 0.5 (N = 10), 2000 runs
    M24-IF indirect-feedback cost [Mark et al. 2024] (N = 10), only to re-check the 'identical trajectories' claim
  via scripts/exp010_terminal_rerun.py run_cell (unchanged):
    CX-A / CX-DF with terminal ingredients, full horizon N = 10
    CX-A / CX-DF with terminal ingredients, dual mode N = 30, Nc = 5
Output: results/EXP016_table1_final/summary.json (one row per method and initial state) and per-run costs for the
methods run through exp001 (used for the M24-IF identity check).

Usage:  python scripts/exp016_table1_final.py [--runs 2000] [--workers 8]
"""
import argparse, importlib.util, json, pathlib, sys, warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "scripts" / file))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


e1 = _load("exp001", "exp001_closed_loop.py")
e10 = _load("exp010", "exp010_terminal_rerun.py")
X0 = [[-0.1, 1.2], [-0.1, 1.3], [-0.1, 1.4], [0.0, 1.5]]


def run(task):
    kind, method, ix, runs = task
    if kind in ("exp001", "exp001-eps0.9"):
        eps = 0.9 if kind == "exp001-eps0.9" else 0.5
        e1.CFG["runs"] = runs; e1.CFG["runs_FS"] = runs; e1.CFG["fs_eps"] = eps
        r = e1.run_task(("E1", method, ix))
        method = method if eps == 0.5 else f"{method}-eps0.9"
        if r.get("infeasible_at_x0"):
            return dict(row=method, x0=X0[ix], status="not_initialised", k0_class=r["k0_class"])
        vc = np.array(r["viol_counts"]); n = r["runs"]; k = int(vc.max())
        return dict(row=method, x0=X0[ix], status="ok", runs=n, max_k_violation=float(k / n), argmax_k=int(vc.argmax()),
                    max_k_cp95_upper=e1.cp_upper(k, n), cost_mean=float(np.mean(r["costs"])),
                    cost_std=float(np.std(r["costs"], ddof=1)), logs=r["logs"], costs=[float(c) for c in r["costs"]])
    N, Nc = {"term10": (10, None), "term30": (30, 5), "term30full": (30, None)}[kind]
    cfg = dict(N=N, Nc=Nc, p=0.9, seed=7000, terminal_margin=0.02, T=40, runs=runs, paths=0, x0=X0)
    c = e10.run_cell((method, ix, X0[ix], True, cfg))
    row = f"{method}-terminal-N{N}" + (f"-Nc{Nc}" if Nc else "")
    if c["status"] != "ok":
        return dict(row=row, x0=X0[ix], status=c["status"])
    return dict(row=row, x0=X0[ix], status="ok", runs=c["runs"], max_k_violation=c["max_k_violation"], argmax_k=c["argmax_k"],
                max_k_cp95_upper=c["max_k_cp95_upper"], cost_mean=c["cost_mean"], cost_std=c["cost_std"], logs=c["logs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lmi-t0", action="store_true", help="revision-v7: add the LMI sensitivity rows FS-DF-T0 (eps 0.5) and eps = 0.9 rows")
    ap.add_argument("--n30", choices=("dm5", "full"), default="dm5",
                    help="N = 30 terminal rows: dual mode Nc = 5 (revision-v4) or full horizon (revision-v6)")
    ap.add_argument("--out", default=str(ROOT / "results" / "EXP016_table1_final"))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from cxsmpc.provenance import write_provenance
    write_provenance(out, dict(runs=a.runs, T=40, seed=7000, x0=X0, fs_eps=0.5, N_main=10, N30_rows=a.n30, lmi_t0=a.lmi_t0), __file__)
    # longest tasks (LMI, dual mode N = 30) first
    tasks = [("exp001", "FS-DF", ix, a.runs) for ix in range(4)]
    if a.lmi_t0:
        tasks += [("exp001", "FS-DF-T0", ix, a.runs) for ix in range(4)]
        tasks += [("exp001-eps0.9", m, ix, a.runs) for m in ("FS-DF", "FS-DF-T0") for ix in range(4)]
    tasks += [("term30" if a.n30 == "dm5" else "term30full", m, ix, a.runs) for m in ("CX-A", "CX-DF") for ix in range(4)]
    tasks += [("exp001", m, ix, a.runs) for m in ("M24-IF", "CX-A", "CX-DF", "NOM") for ix in range(4)]
    tasks += [("term10", m, ix, a.runs) for m in ("CX-A", "CX-DF") for ix in range(4)]
    res = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(run, tasks):
            res.append(r); print("done", r["row"], r["x0"], r["status"], flush=True)
    # identity check: M24-IF vs CX-A per-run costs
    ident = {}
    for ix in range(4):
        A = next((r for r in res if r["row"] == "CX-A" and r["x0"] == X0[ix] and r["status"] == "ok"), None)
        B = next((r for r in res if r["row"] == "M24-IF" and r["x0"] == X0[ix] and r["status"] == "ok"), None)
        if A and B:
            ident[str(X0[ix])] = float(np.max(np.abs(np.array(A["costs"]) - np.array(B["costs"]))))
    rows = [{k: v for k, v in r.items() if k != "costs"} for r in res]
    (out / "summary.json").write_text(json.dumps(dict(rows=rows, m24_vs_cxa_max_abs_cost_diff=ident), indent=1))
    print("M24-IF vs CX-A max |cost diff| per x0:", ident)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

"""EXP003b (PILOT) — computational remedies for the O(N^2) exact formulation.

Methods: CX (exact, full horizon), DM(Nc) dual-mode exact CX with free moves t<Nc and v=Kz afterwards,
HY(Ne) exact head + per-direction trace-chain tail, FS (FS16 fixed-K SDP, eps=0.5; region part also eps=0.9).
Part A: feasible region on benchmark D (N=30, 17x17 grid, reset init), verified 3-way classification.
Part B: Clarabel solve time on random systems (n=2..10, N=10,20,30; 5 systems x 4 states), run alone.
Path constraints only, no terminal sets (identical for all methods). Solutions verified a posteriori.
"""
import csv, json, pathlib, sys, warnings
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import cxsmpc.ocp as ocp  # noqa: E402
from cxsmpc.benchmarks import bench_C, bench_D  # noqa: E402
from cxsmpc.ocp import CXProblem, DualModeCXProblem, FSProblem, HybridProblem  # noqa: E402
from cxsmpc.provenance import write_provenance  # noqa: E402

import cxsmpc.ocp as _ocp_scale
_ocp_scale.OBJ_SCALE = 1e-3  # v4/v2: objective scaling for all methods (see RETRACTION in results log)
warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "EXP003b_final"
CFG = dict(obj_scale=1e-3, A=dict(N=30, grid=np.linspace(-6, 10, 17).tolist(), Nc=[3, 5, 10], Ne=[3], eps=[0.5, 0.9]),
           B=dict(n=[2, 4, 6, 8, 10], N=[10, 20, 30], q=2, systems=10, states=10, Nc=[3, 5], Ne=[], eps=0.5, p=0.9,
                  n_state_cons=2, input_bound=2.0, x0_norm=0.3, seed=3000),
           fallback_max_iters=50000, verify_tol=ocp.VERIFY_TOL, solver="CLARABEL")
ocp.FALLBACK_MAX_ITERS = CFG["fallback_max_iters"]


def methods(model, N, Q, R, P, cons, Nc, Ne, eps_list):
    ms = {"CX": lambda: CXProblem(model, N, Q, R, P, [dict(c) for c in cons])}
    for k in Nc:
        ms[f"DM{k}"] = (lambda k=k: DualModeCXProblem(model, N, Q, R, P, [dict(c) for c in cons], Nc=k))
    for k in Ne:
        ms[f"HY{k}"] = (lambda k=k: HybridProblem(model, N, Q, R, P, [dict(c) for c in cons], Ne=k))
    for e in eps_list:
        ms[f"FS{e}"] = (lambda e=e: FSProblem(model, N, Q, R, P, [dict(c) for c in cons], eps=e))
    return ms


def partA(summ):
    c = CFG["A"]
    D = bench_D(); md = D.meta
    cons = [dict(kind="state", h=md["h"], b=md["b"], p=md["p"])]
    builders = methods(D, c["N"], md["Q"], md["R"], md["P"], cons, c["Nc"], c["Ne"], c["eps"])
    probs = {k: b() for k, b in builders.items() if not k.startswith("FS")}
    rows = []
    for ix, x1 in enumerate(c["grid"]):
        print(f"A row {ix + 1}/{len(c['grid'])}", flush=True)
        for x2 in c["grid"]:
            x0 = np.array([x1, x2]); row = dict(x1=x1, x2=x2)
            for k, b in builders.items():
                pr = b() if k.startswith("FS") else probs[k]  # fresh FS instance (no SCS warm start)
                r = pr.solve(x0, np.zeros((2, 2)))
                row[f"{k}_class"], row[f"{k}_cost"] = r["class"], r["cost"]
            rows.append(row)
    with open(OUT / "partA_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    G = len(rows); keys = list(builders)
    area = {k: {cl: sum(r[f"{k}_class"] == cl for r in rows) / G for cl in ("verified_feasible", "certified_infeasible", "unresolved")} for k in keys}
    # exactness/nesting checks on verified points: DM, HY, FS feasible => CX feasible; J_CX <= J_other
    checks = {}
    for k in keys:
        if k == "CX":
            continue
        both = [r for r in rows if r[f"{k}_class"] == "verified_feasible"]
        checks[k] = dict(verified=len(both),
                         other_feasible_CX_infeasible=sum(r["CX_class"] == "certified_infeasible" for r in both),
                         cost_ratio_other_over_CX_median=float(np.median([r[f"{k}_cost"] / r["CX_cost"] for r in both if r["CX_class"] == "verified_feasible" and r["CX_cost"] > 1e-9])) if both else None,
                         cost_below_CX=sum(r["CX_class"] == "verified_feasible" and r[f"{k}_cost"] < r["CX_cost"] * (1 - 1e-5) - 1e-5 for r in both))
    summ["A"] = dict(grid_points=G, area=area, checks_vs_CX=checks)


def rand_unit(rng, n):
    h = rng.normal(size=n); return h / np.linalg.norm(h)


def partB(summ):
    c = CFG["B"]; rng = np.random.default_rng(c["seed"]); rows = []
    for n in c["n"]:
        m = max(1, n // 2)
        for N in c["N"]:
            for s in range(c["systems"]):
                mdl = bench_C(n, m, c["q"], rng)
                cons = [dict(kind="state", h=rand_unit(rng, n), b=1.0, p=c["p"]) for _ in range(c["n_state_cons"])]
                cons += [dict(kind="input", l=sg * np.eye(m)[i], b=c["input_bound"], p=c["p"]) for i in range(m) for sg in (1, -1)]
                Q, R, P = mdl.meta["Q"], mdl.meta["R"], mdl.meta["P"]
                probs = {k: b() for k, b in methods(mdl, N, Q, R, P, cons, [k for k in c["Nc"] if k < N], c["Ne"], [c["eps"]]).items()}
                for pr in probs.values():
                    pr.solve(np.zeros(n), np.zeros((n, n)))  # warm-up (compile), excluded
                for k in range(c["states"]):
                    x0 = c["x0_norm"] * rand_unit(rng, n)
                    for name, pr in probs.items():
                        r = pr.solve(x0, np.zeros((n, n)))
                        rows.append(dict(n=n, N=N, system=s, state=k, method=name, cls=r["class"], solve=r["solve_time"], solver=r["solver_used"], cost=r["cost"]))
            print(f"B done n={n} N={N}", flush=True)
    with open(OUT / "partB_timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    agg = []
    for n in c["n"]:
        for N in c["N"]:
            entry = dict(n=n, N=N)
            for name in sorted({r["method"] for r in rows if r["n"] == n and r["N"] == N}):
                sub = [r for r in rows if r["n"] == n and r["N"] == N and r["method"] == name and r["solve"] is not None]
                t = np.array([r["solve"] for r in sub]) * 1e3
                entry[name] = dict(p50=float(np.percentile(t, 50)), p99=float(np.percentile(t, 99)), samples=len(t),
                                   verified=sum(r["cls"] == "verified_feasible" for r in sub))
            agg.append(entry)
    summ["B"] = agg


def main():
    part = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    write_provenance(OUT / f"prov_{part}", CFG, __file__)
    sp = OUT / "summary.json"
    summ = json.loads(sp.read_text()) if sp.exists() else dict(label="PILOT")
    (partA if part == "A" else partB)(summ)
    sp.write_text(json.dumps(summ, indent=2, default=str))
    print("PART_DONE", part, flush=True)


if __name__ == "__main__":
    main()

"""EXP003 (PILOT) — containment/feasible region and computation scaling: CX-SMPC (SOCP) vs FS16 fixed-K (SDP).

Part 1 (benchmark D): feasibility grid of initial states for CX and FS(eps), eps in {0.1,...,0.9}; checks
         Proposition 'containment' (FS feasible => CX feasible, J_CX <= J_FS) and reports feasible-area fractions.
Part 2 (benchmark C, random systems): solve-time distributions vs (n, N), same solver (Clarabel), same constraints.
PILOT scale (fewer systems/states than the plan's full E3); results are labelled as pilot.
v3: a-posteriori verification + 3-way classification (verified_feasible / certified_infeasible / unresolved).
v2: sparse (non-condensed) formulation for both methods; solver failures retried with SCS and logged (never counted
as infeasible); solve_time reported is the PRIMARY (Clarabel) solver time.
Path constraints only (t = 0..N), no terminal set, reset initialisation (z0 = x0, Sigma0 = 0) for both methods.
"""
import csv
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cxsmpc.benchmarks import bench_C, bench_D  # noqa: E402
from cxsmpc.ocp import CXProblem, FSProblem  # noqa: E402
from cxsmpc.provenance import write_provenance  # noqa: E402

import cxsmpc.ocp as _ocp_scale
_ocp_scale.OBJ_SCALE = 1e-3  # v4/v2: objective scaling for all methods (see RETRACTION in results log)
warnings.filterwarnings("ignore", message="Solution may be inaccurate")
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "EXP003_pilot_v4_scaled"
CFG = dict(obj_scale=1e-3, part1=dict(N=10, grid=np.linspace(-6, 10, 17).tolist(), eps=[0.1, 0.3, 0.5, 0.7, 0.9]),
           part2=dict(n=[2, 4, 6, 8, 10], N=[10, 20, 30], q=2, systems=5, states=4, eps=0.5, p=0.9,
                      n_state_cons=2, input_bound=2.0, x0_norm=0.3, seed=3000),
           solver="CLARABEL", tol_cost=1e-5)


def part1(summ):
    OUT.mkdir(parents=True, exist_ok=True)
    c = CFG["part1"]
    D = bench_D()
    md = D.meta
    cons = [dict(kind="state", h=md["h"], b=md["b"], p=md["p"])]
    cx = CXProblem(D, c["N"], md["Q"], md["R"], md["P"], cons)
    rows = []
    for ix, x1 in enumerate(c["grid"]):
        print(f"part1 row {ix + 1}/{len(c['grid'])}", flush=True)
        for x2 in c["grid"]:
            x0 = np.array([x1, x2])
            r = cx.solve(x0, np.zeros((2, 2)))
            row = dict(x1=x1, x2=x2, cx_class=r["class"], cx_cost=r["cost"], cx_status=r["status"],
                       cx_solver=r["solver_used"], cx_resid=r["max_residual"])
            for e in c["eps"]:
                fs = FSProblem(D, c["N"], md["Q"], md["R"], md["P"], cons, eps=e)  # fresh instance: no warm-start carry-over
                rf = fs.solve(x0, np.zeros((2, 2)))
                row.update({f"fs{e}_class": rf["class"], f"fs{e}_cost": rf["cost"], f"fs{e}_status": rf["status"],
                            f"fs{e}_solver": rf["solver_used"], f"fs{e}_resid": rf["max_residual"]})
            rows.append(row)
    with open(OUT / "part1_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    G = len(rows)
    def frac(key):
        cl = [r[key] for r in rows]
        return dict(verified_feasible=sum(x == "verified_feasible" for x in cl) / G,
                    certified_infeasible=sum(x == "certified_infeasible" for x in cl) / G,
                    unresolved=sum(x == "unresolved" for x in cl) / G)
    area = {"CX": frac("cx_class")} | {f"FS(eps={e})": frac(f"fs{e}_class") for e in c["eps"]}
    viol_feas = viol_cost = 0
    ratios = {}
    for e in c["eps"]:
        rr = []
        for r in rows:
            fv, cv = r[f"fs{e}_class"] == "verified_feasible", r["cx_class"]
            if fv and cv == "certified_infeasible":
                viol_feas += 1
            if fv and cv == "verified_feasible":
                rr.append(r["cx_cost"] / r[f"fs{e}_cost"] if r[f"fs{e}_cost"] > 0 else 1.0)
                if r["cx_cost"] > r[f"fs{e}_cost"] * (1 + CFG["tol_cost"]) + CFG["tol_cost"]:
                    viol_cost += 1
        ratios[f"FS(eps={e})"] = dict(n=len(rr), median=float(np.median(rr)), min=float(np.min(rr)), max=float(np.max(rr))) if rr else None
    summ["part1"] = dict(grid_points=G, classification="verified_feasible / certified_infeasible / unresolved "
                         "(a-posteriori check of exact Cantelli + FS own LMI/constraints, tol 1e-6)",
                         area_fractions=area,
                         containment_violations_feasibility_verified=viol_feas,
                         containment_violations_cost_verified=viol_cost,
                         cost_ratio_CX_over_FS_on_joint_verified=ratios)


def rand_unit(rng, n):
    h = rng.normal(size=n)
    return h / np.linalg.norm(h)


def part2(summ):
    c = CFG["part2"]
    rng = np.random.default_rng(c["seed"])
    rows = []
    for n in c["n"]:
        m = max(1, n // 2)
        for N in c["N"]:
            for s in range(c["systems"]):
                mdl = bench_C(n, m, c["q"], rng)
                cons = [dict(kind="state", h=rand_unit(rng, n), b=1.0, p=c["p"]) for _ in range(c["n_state_cons"])]
                cons += [dict(kind="input", l=sgn * np.eye(m)[i], b=c["input_bound"], p=c["p"]) for i in range(m) for sgn in (1, -1)]
                Q, R, P = mdl.meta["Q"], mdl.meta["R"], mdl.meta["P"]
                cx = CXProblem(mdl, N, Q, R, P, cons)
                fs = FSProblem(mdl, N, Q, R, P, cons, eps=c["eps"])
                x_warm = np.zeros(n)
                cx.solve(x_warm, np.zeros((n, n)))  # warm-up: excludes one-off canonicalisation from timing
                fs.solve(x_warm, np.zeros((n, n)))
                for k in range(c["states"]):
                    x0 = c["x0_norm"] * rand_unit(rng, n)
                    r1 = cx.solve(x0, np.zeros((n, n)))
                    r2 = fs.solve(x0, np.zeros((n, n)))
                    rows.append(dict(n=n, m=m, N=N, system=s, state=k,
                                     cx_status=r1["status"], cx_solver=r1["solver_used"], fs_solver=r2["solver_used"],
                                     cx_class=r1["class"], fs_class=r2["class"], cx_solve=r1["solve_time"], cx_wall=r1["wall_time"], cx_iters=r1["iters"], cx_cost=r1["cost"],
                                     fs_status=r2["status"], fs_solve=r2["solve_time"], fs_wall=r2["wall_time"], fs_iters=r2["iters"], fs_cost=r2["cost"]))
            print(f"done n={n} N={N}", flush=True)
    with open(OUT / "part2_timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    agg = []
    for n in c["n"]:
        for N in c["N"]:
            sub = [r for r in rows if r["n"] == n and r["N"] == N]
            both = [r for r in sub if r["cx_solve"] is not None and r["fs_solve"] is not None]
            cxt = np.array([r["cx_solve"] for r in both]) * 1e3
            fst = np.array([r["fs_solve"] for r in both]) * 1e3
            pct = lambda a: {f"p{q}": float(np.percentile(a, q)) for q in (50, 95, 99)} | dict(mean=float(a.mean()), max=float(a.max()))
            agg.append(dict(n=n, N=N, samples=len(both),
                            cx_ms=pct(cxt), fs_ms=pct(fst),
                            median_ratio_fs_over_cx=float(np.median(fst / cxt)),
                            cx_classes={k: sum(r["cx_class"] == k for r in sub) for k in ("verified_feasible", "certified_infeasible", "unresolved")},
                            fs_classes={k: sum(r["fs_class"] == k for r in sub) for k in ("verified_feasible", "certified_infeasible", "unresolved")},
                            containment_violations_verified=sum(r["fs_class"] == "verified_feasible" and r["cx_class"] == "certified_infeasible" for r in sub),
                            cost_violations_verified=sum(r["fs_class"] == "verified_feasible" and r["cx_class"] == "verified_feasible"
                                                         and r["cx_cost"] > r["fs_cost"] * (1 + CFG["tol_cost"]) + CFG["tol_cost"] for r in sub)))
    summ["part2"] = agg


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    write_provenance(OUT / f"prov_{part}", CFG, __file__)
    sp = OUT / "summary.json"
    summ = json.loads(sp.read_text()) if sp.exists() else dict(label="PILOT — reduced sample sizes; not the full E3")
    if part in ("all", "part1"):
        part1(summ)
        print(json.dumps(summ["part1"], indent=1), flush=True)
        sp.write_text(json.dumps(summ, indent=2, default=str))
    if part in ("all", "part2"):
        part2(summ)
        sp.write_text(json.dumps(summ, indent=2, default=str))
    print("PART_DONE", part, flush=True)


if __name__ == "__main__":
    main()

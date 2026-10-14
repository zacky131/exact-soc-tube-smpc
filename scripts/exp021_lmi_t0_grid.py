"""EXP021 — Fig. 2 feasibility grid with the LMI sensitivity comparator (revision-v7, reviewer R1-W1).

Same grid, horizon, constraint, solver protocol and a-posteriori verification as EXP003 v4 part 1
(scripts/exp003_equivalence_scaling.py): benchmark D, N = 10, 17 x 17 grid on [-6, 10]^2, reset initialisation
(Sigma0 = 0), eps in {0.1, 0.3, 0.5, 0.7, 0.9}, a fresh problem instance per point. The only change: the LMI
formulation imposes the exact deterministic constraint at t = 0 (FSProblem(t0_exact=True)), since the variance at
t = 0 is zero for a reset; the literal [FS16] classes and the SOC classes are read from EXP003 v4 for comparison.

Output: results/EXP021_lmi_t0_grid/{grid.csv, summary.json}
Usage:  python scripts/exp021_lmi_t0_grid.py [--workers 8]
"""
import argparse, csv, json, pathlib, sys, warnings
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = dict(obj_scale=1e-3, fallback_max_iters=20000, N=10, grid=np.linspace(-6, 10, 17).tolist(), eps=[0.1, 0.3, 0.5, 0.7, 0.9])


def row(x1):
    import cxsmpc.ocp as ocp
    ocp.OBJ_SCALE = CFG["obj_scale"]; ocp.FALLBACK_MAX_ITERS = CFG["fallback_max_iters"]
    from cxsmpc.benchmarks import bench_D
    from cxsmpc.ocp import FSProblem
    D = bench_D(); md = D.meta
    cons = [dict(kind="state", h=md["h"], b=1.0, p=0.9)]
    out = []
    for x2 in CFG["grid"]:
        rec = dict(x1=x1, x2=x2)
        for e in CFG["eps"]:
            fs = FSProblem(D, CFG["N"], md["Q"], md["R"], md["P"], cons, eps=e, t0_exact=True)  # fresh instance
            r = fs.solve(np.array([x1, x2]), np.zeros((2, 2)))
            rec[f"fsT0_{e}_class"] = r["class"]; rec[f"fsT0_{e}_status"] = r["status"]; rec[f"fsT0_{e}_solver"] = r["solver_used"]
            rec[f"fsT0_{e}_resid"] = r["max_residual"]
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "EXP021_lmi_t0_grid"))
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from cxsmpc.provenance import write_provenance
    write_provenance(out, CFG, __file__)
    with Pool(a.workers) as pool:
        rows = [r for rr in pool.map(row, CFG["grid"]) for r in rr]
    ref = {(float(r["x1"]), float(r["x2"])): r for r in csv.DictReader(open(ROOT / "results/EXP003_pilot_v4_scaled/part1_grid.csv"))}
    for r in rows:
        q = ref[(float(r["x1"]), float(r["x2"]))]
        r["cx_class"] = q["cx_class"]
        for e in CFG["eps"]:
            r[f"fs_{e}_class_literal"] = q[f"fs{e}_class"]
    with open(out / "grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    G = len(rows); summ = dict(grid_points=G, soc_verified=sum(r["cx_class"] == "verified_feasible" for r in rows), eps={})
    for e in CFG["eps"]:
        k = f"fsT0_{e}_class"; lit = f"fs_{e}_class_literal"
        summ["eps"][str(e)] = dict(
            t0_exact=dict(verified=sum(r[k] == "verified_feasible" for r in rows), infeasible=sum(r[k] == "certified_infeasible" for r in rows),
                          unresolved=sum(r[k] == "unresolved" for r in rows)),
            literal=dict(verified=sum(r[lit] == "verified_feasible" for r in rows), infeasible=sum(r[lit] == "certified_infeasible" for r in rows),
                         unresolved=sum(r[lit] == "unresolved" for r in rows)),
            t0_feasible_where_soc_certified_infeasible=sum(r[k] == "verified_feasible" and r["cx_class"] == "certified_infeasible" for r in rows),
            soc_only_vs_t0=sum(r["cx_class"] == "verified_feasible" and r[k] != "verified_feasible" for r in rows))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1)); print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

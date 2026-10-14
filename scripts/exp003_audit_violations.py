"""Audit EXP003 pilot v2 part-1 containment 'violations': re-solve flagged points, recompute the EXACT covariance of the
FS plan and test the exact Cantelli constraints and the FS plan's own LMI residual. Writes results/EXP003_pilot_v2/violation_audit.csv."""
import csv, json, pathlib, sys, warnings
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cxsmpc.benchmarks import bench_D
from cxsmpc.ocp import CXProblem, FSProblem, kappa
warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "EXP003_pilot_v2"
rows = list(csv.DictReader(open(OUT / "part1_grid.csv")))
D = bench_D(); md = D.meta; h = md["h"]; k = kappa(md["p"]); N = 10
cons = [dict(kind="state", h=h, b=1.0, p=md["p"])]
cx = CXProblem(D, N, md["Q"], md["R"], md["P"], cons)
T = lambda s: s == "True"
out = []
for r in rows:
    for e in ["0.1", "0.3", "0.5", "0.7", "0.9"]:
        fsf, cxf = T(r[f"fs{e}_feas"]), T(r["cx_feas"])
        costv = fsf and cxf and float(r["cx_cost"]) > float(r[f"fs{e}_cost"]) * (1 + 1e-5) + 1e-5
        if not ((fsf and not cxf) or costv):
            continue
        x0 = np.array([float(r["x1"]), float(r["x2"])])
        fs = FSProblem(D, N, md["Q"], md["R"], md["P"], cons, eps=float(e))
        rf = fs.solve(x0, np.zeros((2, 2)))
        rec = dict(x1=x0[0], x2=x0[1], eps=e, kind="feasibility" if not cxf else "cost",
                   fs_status=rf["status"], fs_primary=rf["status_primary"][:25], fs_solver=rf["solver_used"])
        if rf["feasible"]:
            Sig = D.cov_recursion(np.zeros((2, 2)), rf["z"], rf["v"])
            exact = max(h @ rf["z"][:, t] + k * np.sqrt(max(h @ Sig[t] @ h, 0)) - 1.0 for t in range(N + 1))
            lmi = min(np.linalg.eigvalsh(fs.X[t + 1].value - D.L(fs.X[t].value if t else np.zeros((2, 2))) - D.D(np.outer(np.r_[rf["z"][:, t], rf["v"][:, t]], np.r_[rf["z"][:, t], rf["v"][:, t]]))).min() for t in range(N))
            fsc = max(h @ rf["z"][:, t] - (1 - float(e) / 2 - k**2 / (2 * float(e)) * h @ (fs.X[t].value if t else np.zeros((2, 2))) @ h) for t in range(N + 1))
            rec.update(fs_plan_exact_cantelli_maxviol=exact, fs_lmi_min_eig=lmi, fs_own_constraint_maxviol=fsc)
        rc = cx.solve(x0, np.zeros((2, 2)))
        rec.update(cx_status=rc["status"], cx_primary=rc["status_primary"][:25], cx_cost=rc["cost"], fs_cost=rf["cost"])
        out.append(rec)
with open(OUT / "violation_audit.csv", "w", newline="") as f:
    keys = sorted({kk for o in out for kk in o})
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
print(json.dumps(out, indent=1, default=str))

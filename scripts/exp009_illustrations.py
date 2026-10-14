"""EXP009 — data for illustrative figures of the SCL manuscript (L1, numerical simulation).

(a) Mechanism (benchmark A, s = 1, x1 <= 12, x0 = (0,0,16), N = 15, p = 0.9, trace terminal as in EXP002 v4):
    first-step plans of the exact (CX) and coupling-agnostic (CA) designs; along each plan:
    nominal h'z_t, TRUE predicted std sqrt(h'Sigma_t h) (exact recursion, true model), CA-believed std,
    and open-loop Monte Carlo (20000 samples, plan + tube feedback, true AP002 noise) empirical 90% quantile of h'x_t
    and empirical violation probability P(h'x_t > b).
(b) Closed loop (benchmark D, x0 = (0, 1.5), N = 10, T = 40, as EXP001 final): 50 sample paths of h'x_k and
    per-step violation frequency over 2000 runs for Thm-3 scheme (CX-A), binary initialisation (CX-DF), and
    certainty-equivalent MPC (NOM); matched noise sequences.
"""
import json, pathlib, sys, warnings
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "EXP009_illustrations"
CFG = dict(a=dict(s=1.0, b=12.0, r=16.0, N=15, p=0.9, mc=20000, seed=21000),
           b=dict(x0=[0.0, 1.5], N=10, T=40, runs=2000, paths=50, p=0.9, seed=7000 + 3000),
           obj_scale=1e-3, fallback_max_iters=20000)


def setup():
    import cxsmpc.ocp as ocp
    ocp.OBJ_SCALE = CFG["obj_scale"]; ocp.FALLBACK_MAX_ITERS = CFG["fallback_max_iters"]


def part_a():
    setup()
    import importlib.util
    spec = importlib.util.spec_from_file_location("e2", str(pathlib.Path(__file__).parent / "exp002_mechanism.py"))
    e2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e2)
    from cxsmpc.ocp import CXProblem, kappa
    c = CFG["a"]
    A, CA, P = e2.models(c["s"])
    h = np.array([1.0, 0, 0]); cons = [dict(kind="state", h=h, b=c["b"], p=c["p"])]
    x0 = np.array([0.0, 0.0, c["r"]]); out = dict(kappa=kappa(c["p"]), b=c["b"], N=c["N"])
    for name, mdl in (("CX", A), ("CA", CA)):
        td = e2.best_terminal(mdl, cons, h)
        pr = CXProblem(mdl, c["N"], np.eye(3), np.eye(1), P, cons, Ef=td["Ef"], trace_terminal=(td["S"], td["gamma"]))
        r = pr.solve(x0, np.zeros((3, 3)))
        assert r["class"] == "verified_feasible", (name, r["class"])
        Sig_true = A.cov_recursion(np.zeros((3, 3)), r["z"], r["v"])
        Sig_bel = mdl.cov_recursion(np.zeros((3, 3)), r["z"], r["v"])
        rng = np.random.default_rng(c["seed"])  # same noise for both plans
        X = np.tile(x0, (c["mc"], 1)); q90 = [float(h @ x0)]; pv = [0.0]
        for t in range(c["N"]):
            U = r["v"][:, t] + (X - r["z"][:, t]) @ A.K.T
            X = A.sample_true(X, U, rng)
            q90.append(float(np.quantile(X @ h, c["p"]))); pv.append(float(np.mean(X @ h > c["b"])))
        out[name] = dict(hz=[float(h @ r["z"][:, t]) for t in range(c["N"] + 1)],
                         std_true=[float(np.sqrt(h @ S @ h)) for S in Sig_true],
                         std_believed=[float(np.sqrt(h @ S @ h)) for S in Sig_bel],
                         mc_q90=q90, mc_viol=pv)
    return out


def run_b(method):
    setup()
    from cxsmpc.benchmarks import bench_D
    from cxsmpc.closedloop import NominalMPC, TubeController
    from cxsmpc.ocp import CXProblem
    c = CFG["b"]; D = bench_D(); md = D.meta; h = md["h"]
    cons = [dict(kind="state", h=h, b=1.0, p=c["p"])]
    if method == "NOM":
        ctrl = NominalMPC(D, c["N"], md["Q"], md["R"], md["P"], cons)
    else:
        ctrl = TubeController(CXProblem(D, c["N"], md["Q"], md["R"], md["P"], cons), "A" if method == "CX-A" else "DF")
    viol = np.zeros(c["T"] + 1); paths = []; logs = {}
    for r in range(c["runs"]):
        th = np.random.default_rng(c["seed"] + r).standard_normal(c["T"])
        x = np.array(c["x0"]); ctrl.reset(x); hv = [float(h @ x)]
        for k in range(c["T"]):
            u, ok = ctrl.step(x, k)
            x = D.A[0] @ x + D.B[0] @ u + (D.A[1] @ x + D.B[1] @ u) * th[k]
            hv.append(float(h @ x)); viol[k + 1] += h @ x > 1.0
        if r < c["paths"]:
            paths.append(hv)
        for kk, vv in ctrl.log.items():
            logs[kk] = logs.get(kk, 0) + vv
    return method, dict(paths=paths, viol_freq=(viol / c["runs"]).tolist(), logs=logs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from cxsmpc.provenance import write_provenance
    write_provenance(OUT, CFG, __file__)
    with Pool(4) as pool:
        ra = pool.apply_async(part_a)
        rb = pool.map(run_b, ["CX-A", "CX-DF", "NOM"])
        data = dict(a=ra.get(), b=dict(rb))
    (OUT / "data.json").write_text(json.dumps(data))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

"""EXP002 (PILOT) — mechanism: does modelling the nominal-covariance coupling matter? (claim C004)

Benchmark A (AP002, W = I, K from AP002), constraint x1 <= 6 (SOURCE: ours), p = 0.9, N = 10, T = 30.
Grid: x0 = [0, 0, r], r in {0, 2, 4, 6, 8} x parameter-noise scale s in {0.5, 1, 1.5} (Sp -> s Sp).
Methods: CX-A (exact coupled covariance, Thm A), CA-A (coupling-agnostic design: Sigma+ = L(Sigma) + W, otherwise
identical), NOM (certainty-equivalent; s = 1 only, fewer runs). The TRUE AP002 noise (incl. p2^2 term) is simulated.
Theorem-A plans are deterministic -> solved once per cell, closed loop simulated vectorized over runs.
"""
import json, pathlib, sys, warnings
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "EXP002_mechanism_pilot_v4"
CFG = dict(shift_fallback="if predict-init solve is not verified, apply the shifted previous plan (Theorem rf candidate)", N=15, T=30, b=12.0, h=[1, 0, 0], p=0.9, r=[0, 4, 8, 12, 16, 20], s=[0.5, 1.0, 1.5],
           terminal_margin=0.02, terminal="trace terminal (Assumption T, strict margin 2%) designed per method with its OWN design model; best over lam/H0 grid", runs=2000, runs_nom=500,
           seed=9000, obj_scale=1e-3, fallback_max_iters=20000, Q="I3", R="1", P="terminal_weight(K) for each s")


def models(s):
    import cxsmpc.ocp as ocp
    ocp.OBJ_SCALE = CFG["obj_scale"]; ocp.FALLBACK_MAX_ITERS = CFG["fallback_max_iters"]
    from cxsmpc.benchmarks import bench_A
    from cxsmpc.model import CouplingAgnosticModel, terminal_weight
    A = bench_A(Sp_scale=s)
    CA = CouplingAgnosticModel(A.A, A.B, A.W, A.K, name="A_CA")
    P = terminal_weight(A, np.eye(3), np.eye(1))
    return A, CA, P


def det_plan(problem, model_design, x0, T):
    """Theorem-A deterministic nominal: reset at k=0, predict afterwards (covariance propagated by the DESIGN model)."""
    n = model_design.n
    z0, S0 = np.array(x0, float), np.zeros((n, n))
    from cxsmpc.closedloop import shifted_plan
    plan, status, last = [], [], None
    for k in range(T):
        r = problem.solve(z0, S0)
        status.append(r["class"])
        if r["class"] != "verified_feasible":
            if last is None:
                plan.append(None); continue
            r = shifted_plan(model_design, *last)  # Theorem rf candidate (logged as unverified step)
        last = (r["z"], r["v"])
        xi = np.r_[r["z"][:, 0], r["v"][:, 0]]
        plan.append((r["z"][:, 0].copy(), r["v"][:, 0].copy()))
        z0, S0 = r["z"][:, 1].copy(), model_design.L(S0) + model_design.D(np.outer(xi, xi)) + model_design.W
    return plan, status


def simulate(Atrue, plan, x0, runs, seed, h, b):
    rng = np.random.default_rng(seed)
    X = np.tile(np.asarray(x0, float), (runs, 1)); T = len(plan)
    viol = np.zeros(T + 1); J = np.zeros(runs); fallback = 0
    for k in range(T):
        if plan[k] is None:
            U = X @ Atrue.K.T; fallback += 1
        else:
            z, v = plan[k]; U = v + (X - z) @ Atrue.K.T
        J += np.sum(X * X, axis=1) + np.sum(U * U, axis=1)
        X = Atrue.sample_true(X, U, rng)
        viol[k + 1] = np.mean(X @ h > b)
    return viol, J, fallback


def best_terminal(mdl, cons, h):
    from cxsmpc.terminal import design_terminal
    best = None
    for lamf in (0.1, 0.25, 0.5, 0.75):
        for H0 in (np.eye(mdl.n), np.outer(h, h) + 1e-3 * np.eye(mdl.n)):
            d = design_terminal(mdl, cons, lam=mdl.rho_L() + lamf * (1 - mdl.rho_L()), H0=H0)
            if d and (best is None or d["c"] > best["c"]):
                best = d
    return best


def cell(task):
    s, r = task
    from cxsmpc.ocp import CXProblem
    A, CA, P = models(s)
    h = np.array(CFG["h"], float); cons = [dict(kind="state", h=h, b=CFG["b"], p=CFG["p"])]
    x0 = [0.0, 0.0, float(r)]
    out = dict(s=s, r=r)
    for name, mdl in (("CX-A", A), ("CA-A", CA)):
        td = best_terminal(mdl, cons, h)
        if td is None:
            out[name] = dict(feasible_at_k0=False, terminal="none exists"); continue
        pr = CXProblem(mdl, CFG["N"], np.eye(3), np.eye(1), P, cons, Ef=td["Ef"], trace_terminal=(td["S"], td["gamma"]))
        plan, st = det_plan(pr, mdl, x0, CFG["T"])
        if plan[0] is None:
            out[name] = dict(feasible_at_k0=False); continue
        viol, J, fb = simulate(A, plan, x0, CFG["runs"], CFG["seed"] + int(100 * s) + r, h, CFG["b"])
        # what the design model believes vs the true predicted variance along h at each step (diagnostic of under-tightening)
        resid = max(problem_residual_exact(A, pr, plan_k) for plan_k in [None]) if False else None
        out[name] = dict(feasible_at_k0=True, steps_shift_candidate=sum(c != "verified_feasible" for c in st), fallback_steps=fb,
                         viol=viol.tolist(), max_k=float(viol.max()), time_avg=float(viol[1:].mean()),
                         cost_mean=float(J.mean()), cost_std=float(J.std(ddof=1)))
    return out


def nom_cell(r):
    from cxsmpc.closedloop import NominalMPC
    A, CA, P = models(1.0)
    h = np.array(CFG["h"], float); cons = [dict(kind="state", h=h, b=CFG["b"], p=CFG["p"])]
    ctrl = NominalMPC(A, CFG["N"], np.eye(3), np.eye(1), P, cons)
    rng = np.random.default_rng(CFG["seed"] + 77 + r)
    viol = np.zeros(CFG["T"] + 1)
    for run in range(CFG["runs_nom"]):
        x = np.array([0.0, 0.0, float(r)]); ctrl.reset(x)
        for k in range(CFG["T"]):
            u, ok = ctrl.step(x, k)
            x = A.sample_true(x[None, :], np.atleast_2d(u), rng)[0]
            viol[k + 1] += x @ h > CFG["b"]
    viol /= CFG["runs_nom"]
    return dict(r=r, max_k=float(viol.max()), time_avg=float(viol[1:].mean()), infeasible=ctrl.log["infeasible"])


def cp_upper(rate, n, a=0.05):
    from scipy.stats import beta
    k = int(round(rate * n))
    return float(beta.ppf(1 - a / 2, k + 1, n - k)) if k < n else 1.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from cxsmpc.provenance import write_provenance
    write_provenance(OUT, CFG, __file__)
    tasks = [(s, r) for s in CFG["s"] for r in CFG["r"]]
    with Pool(14) as pool:
        cells = pool.map(cell, tasks)
        noms = pool.map(nom_cell, CFG["r"])
    for c in cells:
        for m in ("CX-A", "CA-A"):
            if c[m].get("feasible_at_k0"):
                c[m]["max_k_cp95_upper"] = cp_upper(c[m]["max_k"], CFG["runs"])
    (OUT / "summary.json").write_text(json.dumps(dict(label="PILOT", cells=cells, nominal_s1=noms), indent=2))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

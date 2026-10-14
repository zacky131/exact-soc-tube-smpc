"""EXP004 (ablations, E4) + EXP005 (failure envelope, E5) — PILOT.

Deterministic-nominal plans (Theorem A, reset at k=0, prediction afterwards, shifted-candidate fallback), terminal
ingredients (trace terminal, 2% strict margin) designed per variant, vectorized closed-loop Monte Carlo (2000 runs).
A2: exact expected cost vs nominal cost.          A4: Cantelli kappa vs Gaussian quantile kappa_G = Phi^-1(p).
A5: K = AP002 gain vs multiplicative-noise LQR gain (benchmark A).
A3: Theorem A vs DF (binary) vs reset-only on D (per-run solves, 500 runs).
E5-tails: theta ~ Gaussian / Laplace / t5 / t3 (unit variance) on D, Cantelli vs Gaussian kappa.
E5-Cp: D with (A1, B1) scaled by c until rho(L) -> 1.
Benchmarks: A (x1 <= 12, x0 = [0,0,r]) and D (h'x <= 1, x0 from FS16/M24). OBJ_SCALE = 1e-3.
"""
import json, pathlib, sys, warnings
from multiprocessing import Pool

import numpy as np
from scipy.stats import beta, norm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "EXP004_005_ablation_envelope_pilot"
CFG = dict(runs=2000, runs_perrun=500, T=30, p=0.9, seed=11000, obj_scale=1e-3, fallback_max_iters=20000, terminal_margin=0.02,
           A=dict(N=15, b=12.0, r=[8, 16], s=1.0), D=dict(N=10, x0=[[-0.1, 1.2], [-0.1, 1.4], [0.0, 1.5]]),
           tails=["gauss", "laplace", "t5", "t3"], cp_scale=[1.0, 1.5, 2.0, 2.5, 3.0])


def setup():
    import cxsmpc.ocp as ocp
    ocp.OBJ_SCALE = CFG["obj_scale"]; ocp.FALLBACK_MAX_ITERS = CFG["fallback_max_iters"]


def best_terminal(mdl, cons, h):
    from cxsmpc.terminal import design_terminal
    best = None
    for lamf in (0.1, 0.25, 0.5, 0.75):
        for H0 in (np.eye(mdl.n), np.outer(h, h) + 1e-3 * np.eye(mdl.n)):
            d = design_terminal(mdl, cons, lam=mdl.rho_L() + lamf * (1 - mdl.rho_L()), H0=H0, margin=CFG["terminal_margin"])
            if d and (best is None or d["c"] > best["c"]):
                best = d
    return best


def det_plan(pr, mdl, x0, T):
    from cxsmpc.closedloop import shifted_plan
    n = mdl.n; z0, S0 = np.array(x0, float), np.zeros((n, n)); plan, shifts, last = [], 0, None
    for k in range(T):
        r = pr.solve(z0, S0)
        if r["class"] != "verified_feasible":
            if last is None:
                return None, shifts
            r = shifted_plan(mdl, *last); shifts += 1
        last = (r["z"], r["v"])
        xi = np.r_[r["z"][:, 0], r["v"][:, 0]]
        plan.append((r["z"][:, 0].copy(), r["v"][:, 0].copy()))
        z0, S0 = r["z"][:, 1].copy(), mdl.L(S0) + mdl.D(np.outer(xi, xi)) + mdl.W
    return plan, shifts


def simulate(mdl, plan, x0, step, rng, h, b, Q, R, runs, T):
    X = np.tile(np.asarray(x0, float), (runs, 1)); viol = np.zeros(T + 1); J = np.zeros(runs)
    for k in range(T):
        z, v = plan[k]; U = v + (X - z) @ mdl.K.T
        J += np.einsum("ij,jk,ik->i", X, Q, X) + np.einsum("ij,jk,ik->i", U, R, U)
        X = step(X, U, rng); viol[k + 1] = np.mean(X @ h > b)
    return viol, J


def generic_step(mdl, sampler):
    def f(X, U, rng):
        th = sampler(rng, (X.shape[0], mdl.q)); Xn = X @ mdl.A[0].T + U @ mdl.B[0].T
        for i in range(1, mdl.q + 1):
            Xn += (X @ mdl.A[i].T + U @ mdl.B[i].T) * th[:, [i - 1]]
        return Xn
    return f


def summarize(viol, J, runs, extra):
    k = int(round(viol.max() * runs))
    return dict(max_k=float(viol.max()), cp95_upper=float(beta.ppf(0.975, k + 1, runs - k)) if k < runs else 1.0,
                time_avg=float(viol[1:].mean()), cost_mean=float(J.mean()), cost_std=float(J.std(ddof=1)), **extra)


def bench(name, Kchoice="default", cscale=1.0):
    from cxsmpc.benchmarks import bench_A, bench_D, theta_sampler
    from cxsmpc.model import MultModel, mult_lqr, terminal_weight
    if name == "A":
        A = bench_A(Sp_scale=CFG["A"]["s"])
        if Kchoice == "lqr":
            K, _ = mult_lqr(A.A, A.B, np.eye(3), np.eye(1)); s_ = A.sample_true
            A = MultModel(A.A, A.B, A.W, K); A.sample_true = s_
        return A, np.eye(3), np.eye(1), terminal_weight(A, np.eye(3), np.eye(1)), np.array([1.0, 0, 0]), CFG["A"]["b"]
    D = bench_D(); md = D.meta
    if cscale != 1.0:
        D2 = MultModel([D.A[0], cscale * D.A[1]], [D.B[0], cscale * D.B[1]], D.W, D.K)
        if D2.rho_L() >= 1:
            return None
        D = D2
    return D, md["Q"], md["R"], terminal_weight(D, md["Q"], md["R"]), md["h"], 1.0


def task_det(t):
    """t = (group, bench, variant, x0, noise, Kchoice, cscale)"""
    setup()
    from cxsmpc.benchmarks import theta_sampler
    from cxsmpc.ocp import CXProblem
    group, bn, variant, x0, noise, Kchoice, cscale = t
    B = bench(bn, Kchoice, cscale)
    if B is None:
        return dict(task=t, status="not MSS (rho(L) >= 1)")
    mdl, Q, R, P, h, b = B
    con = dict(kind="state", h=h, b=b, p=CFG["p"])
    if variant == "gaussk":
        con["kappa"] = float(norm.ppf(CFG["p"]))
    td = best_terminal(mdl, [con], h)
    if td is None:
        return dict(task=t, status="no terminal set")
    N = CFG[bn]["N"]
    pr = CXProblem(mdl, N, Q, R, P, [con], Ef=td["Ef"], trace_terminal=(td["S"], td["gamma"]),
                   cost_mode="nominal" if variant == "nomcost" else "exact")
    plan, shifts = det_plan(pr, mdl, x0, CFG["T"])
    if plan is None:
        return dict(task=t, status="infeasible at x0", rho_L=mdl.rho_L())
    step = mdl.sample_true if (bn == "A") else generic_step(mdl, theta_sampler(noise))
    rng = np.random.default_rng(CFG["seed"] + (7 if bn == "A" else 0) + int(round(1000 * sum(abs(np.asarray(x0)))) ))  # deterministic; matched across variants/noises
    viol, J = simulate(mdl, plan, x0, step, rng, h, b, Q, R, CFG["runs"], CFG["T"])
    return dict(task=t, status="ok", rho_L=mdl.rho_L(), **summarize(viol, J, CFG["runs"], dict(shift_steps=shifts)))


def task_A3(t):
    setup()
    from cxsmpc.closedloop import TubeController
    from cxsmpc.ocp import CXProblem
    mode, x0 = t
    mdl, Q, R, P, h, b = bench("D")
    pr = CXProblem(mdl, CFG["D"]["N"], Q, R, P, [dict(kind="state", h=h, b=b, p=CFG["p"])])
    ctrl = TubeController(pr, mode); viol = np.zeros(CFG["T"] + 1); J = []; logs = {}
    for r in range(CFG["runs_perrun"]):
        th = np.random.default_rng(CFG["seed"] + 500 + r).standard_normal(CFG["T"])
        x = np.array(x0, float); ctrl.reset(x); Jr = 0.0
        for k in range(CFG["T"]):
            u, ok = ctrl.step(x, k); Jr += x @ Q @ x + u @ R @ u
            x = mdl.A[0] @ x + mdl.B[0] @ u + (mdl.A[1] @ x + mdl.B[1] @ u) * th[k]; viol[k + 1] += h @ x > b
        J.append(Jr)
        for kk, vv in ctrl.log.items():
            logs[kk] = logs.get(kk, 0) + vv
    viol /= CFG["runs_perrun"]
    return dict(task=("A3", mode, x0), status="ok", **summarize(viol, np.array(J), CFG["runs_perrun"], dict(logs=logs)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from cxsmpc.provenance import write_provenance
    write_provenance(OUT, CFG, __file__)
    det = []
    xA = [[0.0, 0.0, float(r)] for r in CFG["A"]["r"]]
    for x0 in xA:
        for v in ("exact", "nomcost", "gaussk"):
            det.append(("A2/A4", "A", v, x0, "true", "default", 1.0))
        det.append(("A5", "A", "exact", x0, "true", "lqr", 1.0))
    for x0 in CFG["D"]["x0"]:
        for v in ("exact", "nomcost", "gaussk"):
            det.append(("A2/A4", "D", v, x0, "gauss", "default", 1.0))
        for nz in CFG["tails"][1:]:
            for v in ("exact", "gaussk"):
                det.append(("E5-tails", "D", v, x0, nz, "default", 1.0))
        for c in CFG["cp_scale"][1:]:
            det.append(("E5-Cp", "D", "exact", x0, "gauss", "default", c))
    a3 = [(m, x0) for m in ("A", "DF", "R") for x0 in CFG["D"]["x0"]]
    with Pool(14) as pool:
        r_det = pool.map(task_det, det)
        r_a3 = pool.map(task_A3, a3)
    (OUT / "summary.json").write_text(json.dumps(dict(label="PILOT", det=r_det, A3=r_a3), indent=2, default=str))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

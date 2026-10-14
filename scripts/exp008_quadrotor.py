"""EXP008 (PILOT, L1) — illustrative quadrotor altitude benchmark Q (see cxsmpc.benchmarks.bench_Q for sources/assumptions).

Climb from z0 below the set-point (at rest) with an overshoot ceiling z - z_ref <= 0.2 m and thrust-deviation
limits as chance constraints (p = 0.9). Ts = 0.05 s (20 Hz), N = 30 (1.5 s; N = 20 infeasible for the exact scheme from z0 <= -2 m), T = 80 (4 s).
Methods (deterministic nominal, reset at k=0, prediction afterwards, shifted-candidate fallback, trace terminal):
  CX-A (exact coupled covariance), CA-A (coupling-agnostic design), FS-A (FS16 fixed-K, eps = 0.5, same protocol),
  NOM (certainty-equivalent, per-run solves, fewer runs).
Simulation: true multiplicative + additive noise; PHYSICAL thrust saturation applied (not in the design model) and
counted. Evidence level L1 (linear model, numerical simulation) — NOT a flight or physics-engine result.
"""
import json, pathlib, sys, warnings
from multiprocessing import Pool

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "EXP008_quadrotor_pilot"
CFG = dict(N=30, T=80, p=0.9, ceiling=0.2, z0=[-0.5, -1.0, -2.0, -3.0], noise=dict(base=dict(sig_eta=0.2, sig_xi=0.3),
           high=dict(sig_eta=0.35, sig_xi=0.5)), runs=2000, runs_nom=300, seed=13000, obj_scale=1e-3,
           fallback_max_iters=20000, fs_eps=0.5, terminal_margin=0.02)


def setup():
    import cxsmpc.ocp as ocp
    ocp.OBJ_SCALE = CFG["obj_scale"]; ocp.FALLBACK_MAX_ITERS = CFG["fallback_max_iters"]


def constraints(md):
    return [dict(kind="state", h=np.array([1.0, 0.0]), b=CFG["ceiling"], p=CFG["p"]),
            dict(kind="input", l=np.array([1.0]), b=md["u_max"], p=CFG["p"]),
            dict(kind="input", l=np.array([-1.0]), b=-md["u_min"], p=CFG["p"])]


def best_terminal(mdl, cons):
    from cxsmpc.terminal import design_terminal
    best = None
    for lamf in (0.1, 0.25, 0.5, 0.75):
        for H0 in (np.eye(2), np.diag([1.0, 1e-2])):
            d = design_terminal(mdl, cons, lam=mdl.rho_L() + lamf * (1 - mdl.rho_L()), H0=H0, margin=CFG["terminal_margin"])
            if d and (best is None or d["c"] > best["c"]):
                best = d
    return best


def det_plan(pr, mdl, x0, T):
    from cxsmpc.closedloop import shifted_plan
    z0, S0 = np.array(x0, float), np.zeros((2, 2)); plan, shifts, last = [], 0, None
    for k in range(T):
        r = pr.solve(z0, S0)
        if r["class"] != "verified_feasible":
            if last is None:
                return None, shifts
            r = shifted_plan(mdl, *last); shifts += 1
        last = (r["z"], r["v"]); xi = np.r_[r["z"][:, 0], r["v"][:, 0]]
        plan.append((r["z"][:, 0].copy(), r["v"][:, 0].copy()))
        z0, S0 = r["z"][:, 1].copy(), mdl.L(S0) + mdl.D(np.outer(xi, xi)) + mdl.W
    return plan, shifts


def simulate(Mtrue, md, plan_or_ctrl, x0, runs, rng, vectorized=True):
    T = CFG["T"]; X = np.tile(np.asarray(x0, float), (runs, 1))
    viol_z = np.zeros(T + 1); viol_u = np.zeros(T); sat = np.zeros(T); J = np.zeros(runs)
    Lw = np.linalg.cholesky(Mtrue.W + 1e-18 * np.eye(2))
    for k in range(T):
        if vectorized:
            z, v = plan_or_ctrl[k]; U = v + (X - z) @ Mtrue.K.T
        else:
            U = np.array([plan_or_ctrl.step(X[i], k)[0] for i in range(runs)]).reshape(runs, 1)
        viol_u[k] = np.mean((U[:, 0] > md["u_max"]) | (U[:, 0] < md["u_min"]))
        Us = np.clip(U, md["u_min"], md["u_max"]); sat[k] = np.mean(Us[:, 0] != U[:, 0])
        J += np.einsum("ij,jk,ik->i", X, md["Q"], X) + np.einsum("ij,jk,ik->i", Us, md["R"], Us)
        th = rng.standard_normal((runs, 2))
        Xn = X @ Mtrue.A[0].T + Us @ Mtrue.B[0].T
        for i in (1, 2):
            Xn += (X @ Mtrue.A[i].T + Us @ Mtrue.B[i].T) * th[:, [i - 1]]
        X = Xn + rng.standard_normal((runs, 2)) @ Lw.T
        viol_z[k + 1] = np.mean(X[:, 0] > CFG["ceiling"])
    return viol_z, viol_u, sat, J, X


def summ(viol_z, viol_u, sat, J, Xend, runs):
    k = int(round(viol_z.max() * runs))
    return dict(ceiling_max_k=float(viol_z.max()), ceiling_cp95_upper=float(beta.ppf(0.975, k + 1, runs - k)) if k < runs else 1.0,
                ceiling_time_avg=float(viol_z[1:].mean()), input_viol_max_k=float(viol_u.max()), saturation_frac=float(sat.mean()),
                cost_mean=float(J.mean()), cost_std=float(J.std(ddof=1)), final_abs_z_mean=float(np.abs(Xend[:, 0]).mean()))


def task(t):
    setup()
    from cxsmpc.benchmarks import bench_Q
    from cxsmpc.model import CouplingAgnosticModel
    from cxsmpc.ocp import CXProblem, FSProblem
    method, z0, nz = t
    Qm = bench_Q(**CFG["noise"][nz]); md = Qm.meta; cons = constraints(md); x0 = [z0, 0.0]
    rng = np.random.default_rng(CFG["seed"] + int(100 * abs(z0)) + (1 if nz == "high" else 0))
    if method == "NOM":
        from cxsmpc.closedloop import NominalMPC
        ctrl = NominalMPC(Qm, CFG["N"], md["Q"], md["R"], md["P"], cons); ctrl.reset(x0)
        out = simulate(Qm, md, ctrl, x0, CFG["runs_nom"], rng, vectorized=False)
        return dict(task=t, status="ok", **summ(*out, CFG["runs_nom"]))
    design = Qm if method in ("CX-A", "FS-A") else CouplingAgnosticModel(Qm.A, Qm.B, Qm.W, Qm.K)
    td = best_terminal(design, cons)
    if td is None:
        return dict(task=t, status="no terminal set")
    if method == "FS-A":
        pr = FSProblem(Qm, CFG["N"], md["Q"], md["R"], md["P"], cons, eps=CFG["fs_eps"], Ef=td["Ef"])
    else:
        pr = CXProblem(design, CFG["N"], md["Q"], md["R"], md["P"], cons, Ef=td["Ef"], trace_terminal=(td["S"], td["gamma"]))
    plan, shifts = det_plan(pr, design, x0, CFG["T"])
    if plan is None:
        return dict(task=t, status="infeasible at x0")
    out = simulate(Qm, md, plan, x0, CFG["runs"], rng)
    return dict(task=t, status="ok", shift_steps=shifts, **summ(*out, CFG["runs"]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from cxsmpc.provenance import write_provenance
    write_provenance(OUT, CFG, __file__)
    tasks = [(m, z0, nz) for m in ("CX-A", "CA-A", "FS-A", "NOM") for z0 in CFG["z0"] for nz in CFG["noise"]]
    with Pool(8) as pool:
        res = pool.map(task, tasks)
    (OUT / "summary.json").write_text(json.dumps(dict(label="PILOT (L1, illustrative parameters)", results=res), indent=2, default=str))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

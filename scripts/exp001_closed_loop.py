"""EXP001 + EXP013 (PILOT) — closed-loop Monte Carlo on benchmark D.

E1  (K = LQ-optimal): NOM, CX-A (Thm A), CX-DF (binary init, exact), FS-DF (FS16), M24-IF (exact realised covariance,
     conditional-mean cost; by Prop. ifopt identical to CX-A for the LQ gain — checked here).
E13 (K = non-optimal, LQR designed with R/10): CX-A vs M24-IF, regulation and tracking (reference beyond constraint).
     Covariance-bias diagnostic for M24-IF: C_k = mean e_k e_k' vs Sbar_k = mean Sigma_{0|k} along h.
Thm C: feasibility check of the constant robust-coupling tightening on D (margin vs slack), no closed loop run.
All: 4 initial states x 500 runs x T=40, theta ~ N(0,1), matched seeds per (x0, run) across methods,
OBJ_SCALE=1e-3 for every method, SCS fallback 20000 iterations, solutions verified a posteriori.
"""
import csv, json, pathlib, sys, warnings
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "EXP001_closed_loop_final"
CFG = dict(protocol_v2="shifted-candidate fallback in TubeController; cell reported as infeasible_at_x0 if the k=0 problem is not verified feasible (no closed loop run)", N=10, T=40, runs=2000, runs_FS=500, x0=[[-0.1, 1.2], [-0.1, 1.3], [-0.1, 1.4], [0.0, 1.5]], p=0.9, seed=7000,
           obj_scale=1e-3, fallback_max_iters=20000, fs_eps=0.5, K_alt_R_factor=0.1, ref_tracking=[-1.0, 1.5])


def setup():
    import cxsmpc.ocp as ocp
    ocp.OBJ_SCALE = CFG["obj_scale"]; ocp.FALLBACK_MAX_ITERS = CFG["fallback_max_iters"]


def build(exp, method):
    setup()
    from cxsmpc.benchmarks import bench_D
    from cxsmpc.closedloop import IFController, IFProblem, NominalMPC, TubeController
    from cxsmpc.model import mult_lqr, terminal_weight
    from cxsmpc.ocp import CXProblem, FSProblem
    D = bench_D(); md = D.meta
    Q, R = md["Q"], md["R"]
    if exp == "E13":
        Kalt, _ = mult_lqr(D.A, D.B, Q, CFG["K_alt_R_factor"] * R)
        D = type(D)(D.A, D.B, D.W, Kalt, name="D_Kalt"); D.meta = md
        P = terminal_weight(D, Q, R)  # Assumption P for the non-optimal K (with the true cost weights)
    else:
        P = md["P"]
    cons = [dict(kind="state", h=md["h"], b=1.0, p=CFG["p"])]
    N = CFG["N"]
    ref = CFG["ref_tracking"] if method.endswith("-TRK") else None
    if method == "NOM":
        return D, NominalMPC(D, N, Q, R, P, cons)
    if method.startswith("CX-A"):
        if ref is not None:  # Theorem-A scheme with tracking cost: same IF problem with e fixed to 0 (no measurement in plan)
            pr = IFProblem(D, N, Q, R, P, cons, ref=ref)
            return D, _ZeroEIF(pr)
        return D, TubeController(CXProblem(D, N, Q, R, P, cons), "A")
    if method == "CX-DF":
        return D, TubeController(CXProblem(D, N, Q, R, P, cons), "DF")
    if method == "FS-DF":
        return D, TubeController(FSProblem(D, N, Q, R, P, cons, eps=CFG["fs_eps"]), "DF")
    if method == "FS-DF-T0":  # revision-v7 (R1-W1): exact constraint at t = 0 for reset solves (Sigma0 = 0)
        from cxsmpc.ocp import FSResetExact
        return D, TubeController(FSResetExact(D, N, Q, R, P, cons, eps=CFG["fs_eps"]), "DF")
    if method.startswith("M24-IF"):
        return D, IFController(IFProblem(D, N, Q, R, P, cons, ref=ref))
    raise ValueError(method)


class _ZeroEIF:
    """Deterministic-nominal scheme with the IF problem's cost evaluated at e = 0 (Theorem A with a tracking cost)."""

    def __init__(self, pr):
        from cxsmpc.closedloop import IFController
        self.inner = IFController(pr)

    def reset(self, x0):
        self.inner.reset(x0); self.log = self.inner.log

    def step(self, x, k):
        c = self.inner
        c.pr.e.value = np.zeros(c.m.n)
        res = c.pr.solve(c.z0, c.S0)
        c.Sigma_hist.append(c.S0.copy()); c.z_hist.append(c.z0.copy())
        if res["class"] != "verified_feasible":
            c.log["unresolved" if res["class"] == "unresolved" else "infeasible"] += 1
            return c.m.K @ x, False
        xi = np.r_[res["z"][:, 0], res["v"][:, 0]]
        u = res["v"][:, 0] + c.m.K @ (x - c.z0)
        c.z0, c.S0 = res["z"][:, 1].copy(), c.m.L(c.S0) + c.m.D(np.outer(xi, xi)) + c.m.W
        return u, True

    @property
    def Sigma_hist(self):
        return self.inner.Sigma_hist

    @property
    def z_hist(self):
        return self.inner.z_hist


def run_task(task):
    exp, method, ix = task
    D, ctrl = build(exp, method)
    md = D.meta; h = md["h"]; T = CFG["T"]
    x0 = np.array(CFG["x0"][ix], float)
    if method != "NOM":
        pr0 = ctrl.pr if hasattr(ctrl, "pr") else ctrl.inner.pr
        if hasattr(pr0, "e"):
            pr0.e.value = np.zeros(D.n)
        r0 = pr0.solve(x0, np.zeros((D.n, D.n)))
        if r0["class"] != "verified_feasible":
            return dict(exp=exp, method=method, x0=CFG["x0"][ix], infeasible_at_x0=True, k0_class=r0["class"])
    viol = np.zeros(T + 1, int); costs = []; logs = {}; per_run = []
    ee = np.zeros((T, 2, 2)); Sb = np.zeros((T, 2, 2)); has_hist = False
    nruns = CFG["runs_FS"] if method.startswith("FS-DF") else CFG["runs"]
    for r in range(nruns):
        rng = np.random.default_rng(CFG["seed"] + 1000 * ix + r)  # matched across methods
        th = rng.standard_normal(T)
        x = x0.copy(); ctrl.reset(x); J = 0.0; maxh = h @ x; nv = 0
        for k in range(T):
            u, ok = ctrl.step(x, k)
            if hasattr(ctrl, "z_hist") and len(ctrl.z_hist) > k:
                e = x - ctrl.z_hist[k]; ee[k] += np.outer(e, e); Sb[k] += ctrl.Sigma_hist[k]; has_hist = True
            J += x @ md["Q"] @ x + u @ md["R"] @ u
            x = D.A[0] @ x + D.B[0] @ u + (D.A[1] @ x + D.B[1] @ u) * th[k]
            v = h @ x > 1.0; viol[k + 1] += v; nv += v; maxh = max(maxh, h @ x)
        costs.append(J); per_run.append(dict(run=r, J=J, n_viol=nv, max_hx=maxh))
        for kk, vv in ctrl.log.items():
            logs[kk] = logs.get(kk, 0) + vv
    out = dict(exp=exp, method=method, x0=CFG["x0"][ix], runs=nruns, viol_counts=viol.tolist(), costs=costs, logs=logs, per_run=per_run)
    if has_hist:
        R_ = nruns
        out["bias_h"] = [float(h @ (ee[k] / R_ - Sb[k] / R_) @ h) for k in range(T)]
        out["var_h_true"] = [float(h @ (ee[k] / R_) @ h) for k in range(T)]
    return out


def thmC_check():
    setup()
    from cxsmpc.benchmarks import bench_D
    from cxsmpc.ocp import kappa
    D = bench_D(); md = D.meta; h = md["h"]; res = []
    for r in (0.25, 0.5, 1.0, 1.5):
        best = None
        for a in np.logspace(-2, 1.5, 60):
            Phis = [D.Phi[0]] + [np.sqrt(1 + a) * P for P in D.Phi[1:]]
            M = sum(np.kron(P, P) for P in Phis)
            if max(abs(np.linalg.eigvals(M))) >= 1:
                continue
            rhs = D.W + (1 + 1 / a) * D.D(r**2 * np.eye(4))
            S = np.linalg.solve(np.eye(4) - M, rhs.reshape(-1, order="F")).reshape(2, 2, order="F")
            mg = kappa(CFG["p"]) * np.sqrt(h @ S @ h)
            best = mg if best is None else min(best, mg)
        res.append(dict(Xi_radius=r, min_margin_over_alpha=best,
                        slack_at_x0=[1.0 - float(h @ np.array(x)) for x in CFG["x0"]],
                        x0_norms=[float(np.linalg.norm(x)) for x in CFG["x0"]]))
    return res


def cp_upper(k, n, a=0.05):
    from scipy.stats import beta
    return float(beta.ppf(1 - a / 2, k + 1, n - k)) if k < n else 1.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from cxsmpc.provenance import write_provenance
    write_provenance(OUT, CFG, __file__)
    tasks = [("E1", m, i) for m in ("NOM", "CX-A", "CX-DF", "FS-DF", "M24-IF") for i in range(4)]
    tasks += [("E13", m, i) for m in ("CX-A", "M24-IF", "CX-A-TRK", "M24-IF-TRK") for i in range(4)]
    with Pool(min(14, len(tasks))) as pool:
        results = []
        for r in pool.imap_unordered(run_task, tasks):
            results.append(r); print(f"done {r['exp']} {r['method']} x0={r['x0']}", flush=True)
    summ = dict(label="PILOT", thmC_feasibility=thmC_check(), table=[])
    with open(OUT / "per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["exp", "method", "x0", "run", "J", "n_viol", "max_hx"])
        w.writeheader()
        for r in results:
            for pr in r.get("per_run", []):
                w.writerow(dict(exp=r["exp"], method=r["method"], x0=r["x0"], **pr))
    for r in sorted(results, key=lambda r: (r["exp"], r["method"], r["x0"])):
        if r.get("infeasible_at_x0"):
            summ["table"].append(dict(exp=r["exp"], method=r["method"], x0=r["x0"], infeasible_at_x0=True, k0_class=r["k0_class"]))
            continue
        n = r["runs"]
        vc = np.array(r["viol_counts"]); kmax = int(vc.argmax())
        row = dict(exp=r["exp"], method=r["method"], x0=r["x0"], runs=n,
                   max_k_violation_rate=float(vc.max() / n), max_k_cp95_upper=cp_upper(int(vc.max()), n), argmax_k=kmax,
                   time_avg_violation_rate=float(vc[1:].sum() / (n * CFG["T"])),
                   cost_mean=float(np.mean(r["costs"])), cost_std=float(np.std(r["costs"], ddof=1)),
                   cost_median=float(np.median(r["costs"])), logs=r["logs"])
        if "bias_h" in r:
            row["bias_h_max_abs"] = float(np.max(np.abs(r["bias_h"])))
            row["bias_h_rel_max"] = float(np.max(np.abs(np.array(r["bias_h"])) / np.maximum(np.array(r["var_h_true"]), 1e-12)))
            row["bias_h"] = r["bias_h"]
        summ["table"].append(row)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

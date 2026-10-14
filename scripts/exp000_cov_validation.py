"""EXP000 — covariance-predictor validation and solution audit (MUST_RUN, supports C001 prerequisite).

For CX-SMPC plans generated from random feasible initial states:
  (1) Monte-Carlo empirical covariance of e_t = x_t - z_t under the TRUE noise model vs the exact recursion;
      standardized per-entry deviations using batch standard errors (correct predictor -> |z| mostly < 3);
  (2) audit: kernel-predicted h'Sigma_t h == recursion; Cantelli constraints satisfied by the returned plan;
      solver objective == independently recomputed expected cost.
Also: replication of AP002 stationary covariance (Cor. 1) with AP002's gain.
"""
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cxsmpc.benchmarks import bench_A, bench_D, theta_sampler  # noqa: E402
from cxsmpc.kernels import DirectionKernel  # noqa: E402
from cxsmpc.model import terminal_weight  # noqa: E402
from cxsmpc.ocp import CXProblem, kappa  # noqa: E402
from cxsmpc.provenance import write_provenance  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "EXP000"
CFG = dict(N=10, n_plans=20, M=20000, n_batches=20, seed_plans=1000, seed_mc=2000,
           noise_D=["gauss", "t5", "rademacher"], p=0.9,
           bench_A_constraint=dict(h=[1, 0, 0], b=3.0, note="SOURCE: ours (AP002 has no constraints)"),
           bench_A_cost=dict(Q="I3", R="1", P="terminal_weight(K)"))


def simulate(model, z, v, x0, M, rng, step):
    N = v.shape[1]
    X = np.tile(x0, (M, 1))
    E = [X - z[:, 0]]
    for t in range(N):
        U = v[:, t] + (X - z[:, t]) @ model.K.T
        X = step(X, U, rng)
        E.append(X - z[:, t + 1])
    return E


def step_generic(model, sampler):
    def f(X, U, rng):
        M = X.shape[0]
        th = sampler(rng, (M, model.q))
        Xn = X @ model.A[0].T + U @ model.B[0].T
        for i in range(1, model.q + 1):
            Xn += (X @ model.A[i].T + U @ model.B[i].T) * th[:, [i - 1]]
        if np.any(model.W):
            Xn += rng.multivariate_normal(np.zeros(model.n), model.W, size=M)
        return Xn
    return f


def exact_cost(model, Q, R, P, z, v, Sig):
    N = v.shape[1]
    K = model.K
    J = sum(z[:, t] @ Q @ z[:, t] + v[:, t] @ R @ v[:, t] + np.trace((Q + K.T @ R @ K) @ Sig[t]) for t in range(N))
    return J + z[:, N] @ P @ z[:, N] + np.trace(P @ Sig[N])


def run_bench(name, model, Q, R, P, cons, x0_sampler, noise_list, writer, summ):
    N, M, nb = CFG["N"], CFG["M"], CFG["n_batches"]
    prob = CXProblem(model, N, Q, R, P, cons)
    rng_p = np.random.default_rng(CFG["seed_plans"])
    plans, tries = [], 0
    while len(plans) < CFG["n_plans"] and tries < 2000:
        tries += 1
        x0 = x0_sampler(rng_p)
        r = prob.solve(x0, np.zeros((model.n, model.n)))
        if r["feasible"]:
            plans.append((x0, r))
    audit = dict(kernel_max_rel_err=0.0, cantelli_max_violation=-np.inf, cost_max_rel_err=0.0,
                 statuses={}, n_plans=len(plans), tries=tries)
    for x0, r in plans:
        audit["statuses"][r["status"]] = audit["statuses"].get(r["status"], 0) + 1
        Sig = model.cov_recursion(np.zeros((model.n, model.n)), r["z"], r["v"])
        for c in cons:
            ker = DirectionKernel(model, c["h"], N)
            h = np.asarray(c["h"], float)
            for t in range(N + 1):
                pred = sum(float(np.sum(ker.P[t] * 0)) for _ in [0]) + ker.wconst[t] + sum(
                    float(np.sum((ker.R[t - 1 - j] @ np.concatenate([r["z"][:, j], r["v"][:, j]])) ** 2)) for j in range(t))
                true = h @ Sig[t] @ h
                audit["kernel_max_rel_err"] = max(audit["kernel_max_rel_err"], abs(pred - true) / max(1.0, abs(true)))
                lhs = h @ r["z"][:, t] + kappa(c["p"]) * np.sqrt(max(true, 0.0))
                audit["cantelli_max_violation"] = max(audit["cantelli_max_violation"], lhs - c["b"])
        Jx = exact_cost(model, Q, R, P, r["z"], r["v"], Sig)
        audit["cost_max_rel_err"] = max(audit["cost_max_rel_err"], abs(Jx - r["cost"]) / max(1.0, abs(Jx)))
    summ[name] = dict(audit=audit, noise={})
    for noise in noise_list:
        step = model.sample_true if noise == "true_AP002" else step_generic(model, theta_sampler(noise))
        zs_all, rel_all = [], []
        for ip, (x0, r) in enumerate(plans):
            rng = np.random.default_rng(CFG["seed_mc"] + ip)  # matched across noise kinds per plan
            E = simulate(model, r["z"], r["v"], x0, M, rng, step)
            Sig = model.cov_recursion(np.zeros((model.n, model.n)), r["z"], r["v"])
            for t in range(1, N + 1):
                e = E[t]
                C = e.T @ e / M  # mean is 0 in theory; second moment = covariance
                bat = np.stack([eb.T @ eb / len(eb) for eb in np.array_split(e, nb)])
                se = bat.std(axis=0, ddof=1) / np.sqrt(nb)
                iu = np.triu_indices(model.n)
                # SE floor: entries with (near-)zero MC variance (e.g. Rademacher theta, theta^2 == 1) are
                # checked by absolute error instead of a z-score (z would be numerically meaningless)
                floor = 1e-9 * max(1.0, np.abs(Sig[t]).max())
                degenerate = se <= floor
                assert np.all(np.abs(C - Sig[t])[degenerate] <= 1e-6 * max(1.0, np.abs(Sig[t]).max())), "degenerate-entry mismatch"
                zsc = ((C - Sig[t]) / np.where(degenerate, np.inf, se))[iu]
                rel = np.linalg.norm(C - Sig[t]) / np.linalg.norm(Sig[t])
                mean_z = np.abs(e.mean(axis=0)) / (e.std(axis=0, ddof=1) / np.sqrt(M))
                zs_all.extend(np.abs(zsc).tolist())
                rel_all.append(rel)
                writer.writerow(dict(bench=name, noise=noise, plan=ip, t=t, rel_frob_err=rel,
                                     max_abs_z=float(np.max(np.abs(zsc))), max_abs_mean_z=float(mean_z.max())))
        zs = np.array(zs_all)
        from scipy.stats import t as tdist
        summ[name]["noise"][noise] = dict(n_entries=int(zs.size), frac_abs_z_gt3=float(np.mean(zs > 3)),
                                          frac_abs_z_gt2=float(np.mean(zs > 2)), max_abs_z=float(zs.max()),
                                          reference_t19=dict(P_abs_gt3=float(2 * tdist.sf(3, nb - 1)), P_abs_gt2=float(2 * tdist.sf(2, nb - 1))),
                                          median_rel_frob_err=float(np.median(rel_all)),
                                          p95_rel_frob_err=float(np.percentile(rel_all, 95)))


def main():
    write_provenance(OUT, CFG, __file__)
    summ = {}
    # AP002 replication (Cor. 1 with AP002 gain)
    A = bench_A()
    S_inf = A.stationary_cov()
    summ["AP002_stationary_cov"] = dict(computed=np.round(S_inf, 3).tolist(),
                                        reported_in_AP002=[[1.79, 0, 0.06], [0, "1.17 (text extraction shows 0.17; >=1 required since W=I)", 0.26], [0.06, 0.26, 1.87]],
                                        rho_L=A.rho_L())
    with open(OUT / "per_step.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bench", "noise", "plan", "t", "rel_frob_err", "max_abs_z", "max_abs_mean_z"])
        w.writeheader()
        QA, RA = np.eye(3), np.eye(1)
        PA = terminal_weight(A, QA, RA)
        consA = [dict(kind="state", h=np.array(CFG["bench_A_constraint"]["h"], float), b=CFG["bench_A_constraint"]["b"], p=CFG["p"])]
        run_bench("A", A, QA, RA, PA, consA, lambda r: r.uniform(-3, 3, 3), ["true_AP002"], w, summ)
        D = bench_D()
        md = D.meta
        consD = [dict(kind="state", h=md["h"], b=md["b"], p=md["p"])]
        run_bench("D", D, md["Q"], md["R"], md["P"], consD, lambda r: r.uniform([-1, 0], [1, 2]), CFG["noise_D"], w, summ)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps(summ, indent=2, default=str))


if __name__ == "__main__":
    main()

"""EXP014 — nnz of the conic problem for ALL timed instances of the Table 3 sizes (revision-v2, item 4).

Replays the random-number sequence of scripts/exp003b_hybrid.py partB exactly (same seed, same order of bench_C,
constraint and initial-state draws), so the systems below are the very instances whose solve times are reported in
Fig. 5 and Table 3 (10 random systems per (n, N); problem size does not depend on the initial state). For the sizes
(n, N) in Table 3, the conic data passed to Clarabel are extracted with CVXPY's get_problem_data after setting the
parameter values (no solve), and nnz(A), the number of variables and of constraint rows are recorded per instance.

Usage:  python scripts/exp014_problem_sizes_all.py [--out results/EXP014_problem_sizes_all]
"""
import argparse, json, pathlib, sys, warnings

import cvxpy as cp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = {(4, 10), (4, 30), (10, 10), (10, 30)}  # sizes reported in Table 3


def rand_unit(rng, n):
    h = rng.normal(size=n); return h / np.linalg.norm(h)


def conic_size(pr, n):
    pr.z0.value = np.zeros(n); S0 = np.zeros((n, n)); pr.Sig0.value = S0; pr._set_extra(S0)
    data, _, _ = pr.prob.get_problem_data(cp.CLARABEL)
    A = data["A"]
    return dict(nnz=int(A.nnz), variables=int(A.shape[1]), rows=int(A.shape[0]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "results" / "EXP014_problem_sizes_all"))
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import cxsmpc.ocp as ocp
    ocp.OBJ_SCALE = 1e-3
    from cxsmpc.benchmarks import bench_C
    from cxsmpc.ocp import CXProblem, DualModeCXProblem, FSProblem
    c = dict(n=[2, 4, 6, 8, 10], N=[10, 20, 30], q=2, systems=10, states=10, p=0.9, n_state_cons=2, input_bound=2.0,
             x0_norm=0.3, seed=3000, eps=0.5)  # = EXP003b CFG["B"]
    rng = np.random.default_rng(c["seed"]); rows = []
    for n in c["n"]:
        m = max(1, n // 2)
        for N in c["N"]:
            for s in range(c["systems"]):
                mdl = bench_C(n, m, c["q"], rng)
                cons = [dict(kind="state", h=rand_unit(rng, n), b=1.0, p=c["p"]) for _ in range(c["n_state_cons"])]
                cons += [dict(kind="input", l=sg * np.eye(m)[i], b=c["input_bound"], p=c["p"]) for i in range(m) for sg in (1, -1)]
                if (n, N) in TARGET:
                    Q, R, P = mdl.meta["Q"], mdl.meta["R"], mdl.meta["P"]
                    probs = {"CX": CXProblem(mdl, N, Q, R, P, [dict(cc) for cc in cons]),
                             "DM3": DualModeCXProblem(mdl, N, Q, R, P, [dict(cc) for cc in cons], Nc=3),
                             "FS0.5": FSProblem(mdl, N, Q, R, P, [dict(cc) for cc in cons], eps=c["eps"])}
                    for name, pr in probs.items():
                        rows.append(dict(n=n, N=N, system=s, method=name, **conic_size(pr, n)))
                for _ in range(c["states"]):
                    rand_unit(rng, n)  # consume the initial-state draws exactly as in EXP003b
            if (n, N) in TARGET:
                print(f"done n={n} N={N}", flush=True)
    summ = {}
    for (n, N) in sorted(TARGET):
        for name in ("CX", "DM3", "FS0.5"):
            v = [r["nnz"] for r in rows if r["n"] == n and r["N"] == N and r["method"] == name]
            summ[f"n={n}|N={N}|{name}"] = dict(min=min(v), max=max(v), median=float(np.median(v)), count=len(v),
                                               spread_pct=100.0 * (max(v) - min(v)) / min(v))
    (out / "sizes_all.json").write_text(json.dumps(dict(config=c, rows=rows, summary=summ), indent=1))
    for k, v in summ.items():
        print(k, v)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

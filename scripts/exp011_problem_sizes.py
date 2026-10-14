"""EXP011 — problem sizes behind the timing study of Section 6.4 (reviewer request: report sizes next to solve times).

For each (n, N) of the EXP003b timing grid, one random system is generated with the same generator and constraint set
(bench_C, q = 2, m = max(1, n // 2), 2 state + 2m input chance constraints). The conic problem passed to Clarabel is
extracted with CVXPY's get_problem_data, and the script records the number of variables (columns of A), the number of
constraint rows, nnz(A), and the cone dimensions (SOC sizes, PSD sizes) for the full-horizon SOC formulation (CX),
the dual-mode variant (DM3, DM5) and the LMI formulation (FS0.5).

Problem sizes depend only on the dimensions and the sparsity pattern, not on the random values (up to the numerical
rank of the kernel factors), so one system per size suffices. No number from this script is in the manuscript.

Usage:  python scripts/exp011_problem_sizes.py [--out results/EXP011_problem_sizes]
"""
import argparse, csv, json, pathlib, sys, warnings

import cvxpy as cp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def rand_unit(rng, n):
    h = rng.normal(size=n); return h / np.linalg.norm(h)


def sizes(pr):
    data, _, _ = pr.prob.get_problem_data(cp.CLARABEL)
    A = data["A"]; dims = data["dims"]
    soc = list(getattr(dims, "soc", []) or []); psd = list(getattr(dims, "psd", []) or [])
    return dict(variables=int(A.shape[1]), rows=int(A.shape[0]), nnz=int(A.nnz),
                n_soc=len(soc), soc_rows=int(sum(soc)), max_soc=int(max(soc)) if soc else 0,
                n_psd=len(psd), max_psd=int(max(psd)) if psd else 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "results" / "EXP011_problem_sizes"))
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import cxsmpc.ocp as ocp
    ocp.OBJ_SCALE = 1e-3
    from cxsmpc.benchmarks import bench_C
    from cxsmpc.ocp import CXProblem, DualModeCXProblem, FSProblem
    cfg = dict(n=[2, 4, 6, 8, 10], N=[10, 20, 30], q=2, p=0.9, n_state_cons=2, input_bound=2.0, x0_norm=0.3, seed=3000, eps=0.5, Nc=[3, 5])
    rng = np.random.default_rng(cfg["seed"]); rows = []
    for n in cfg["n"]:
        m = max(1, n // 2)
        for N in cfg["N"]:
            mdl = bench_C(n, m, cfg["q"], rng)
            cons = [dict(kind="state", h=rand_unit(rng, n), b=1.0, p=cfg["p"]) for _ in range(cfg["n_state_cons"])]
            cons += [dict(kind="input", l=sg * np.eye(m)[i], b=cfg["input_bound"], p=cfg["p"]) for i in range(m) for sg in (1, -1)]
            Q, R, P = mdl.meta["Q"], mdl.meta["R"], mdl.meta["P"]
            probs = {"CX": CXProblem(mdl, N, Q, R, P, [dict(c) for c in cons]),
                     "FS0.5": FSProblem(mdl, N, Q, R, P, [dict(c) for c in cons], eps=cfg["eps"])}
            for k in cfg["Nc"]:
                if k < N:
                    probs[f"DM{k}"] = DualModeCXProblem(mdl, N, Q, R, P, [dict(c) for c in cons], Nc=k)
            x0 = cfg["x0_norm"] * rand_unit(rng, n)
            for name, pr in probs.items():
                pr.solve(x0, np.zeros((n, n)))  # sets parameter values before extracting the conic data
                rows.append(dict(n=n, m=m, N=N, method=name, **sizes(pr)))
            print(f"done n={n} N={N}", flush=True)
    with open(out / "sizes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out / "sizes.json").write_text(json.dumps(dict(config=cfg, rows=rows), indent=1))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

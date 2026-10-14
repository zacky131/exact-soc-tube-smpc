"""EXP023 — selected terminal-design parameters (revision-v7, reviewer R1-W5: reproducible numerical record).

Re-runs the deterministic terminal design used in the experiments (best over the grid lambda = rho(L) + f (1 - rho(L)),
f in {0.1, 0.25, 0.5, 0.75}, H0 in {I, hh' + 1e-3 I}, 2 % margin; the design with the largest terminal level alpha
is kept) and records the selected (f, lambda, H0, alpha, gamma_bar) for every design that appears in the paper:
  benchmark D (Section 6.1): Cantelli kappa = 3 (Table 1, Table 2 Cantelli column) and normal quantile kappa = 1.28
  (Table 2); example of [MF2026] (Section 6.2, Figs. 1 and 5): beta in {1, 1.5}, SOC and coupling-agnostic designs.
Output: results/EXP023_terminal_params/params.json
"""
import importlib.util, json, pathlib, sys, warnings

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def select(mdl, cons, h, margin=0.02):
    from cxsmpc.terminal import design_terminal
    best = None
    for lamf in (0.1, 0.25, 0.5, 0.75):
        for hname, H0 in (("I", np.eye(mdl.n)), ("hh'+1e-3 I", np.outer(h, h) + 1e-3 * np.eye(mdl.n))):
            lam = mdl.rho_L() + lamf * (1 - mdl.rho_L())
            d = design_terminal(mdl, cons, lam=lam, H0=H0, margin=margin)
            if d and (best is None or d["c"] > best["c"]):
                best = dict(f=lamf, lam=float(lam), H0=hname, alpha=float(d["c"]), gamma=float(d["gamma"]), rho_L=float(mdl.rho_L()), c=d["c"])
    if best:
        best.pop("c")
    return best


def main():
    import cxsmpc.ocp as ocp
    ocp.OBJ_SCALE = 1e-3
    from cxsmpc.benchmarks import bench_D
    out = {}
    D = bench_D(); md = D.meta; h = md["h"]
    for name, kap in (("D, Cantelli kappa=3", None), ("D, normal quantile kappa=1.28", float(norm.ppf(0.9)))):
        con = dict(kind="state", h=h, b=1.0, p=0.9)
        if kap is not None:
            con["kappa"] = kap
        out[name] = select(D, [con], h)
    spec = importlib.util.spec_from_file_location("e2", str(ROOT / "scripts" / "exp002_mechanism.py"))
    e2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e2)
    hA = np.array([1.0, 0, 0])
    for beta in (1.0, 1.5):
        A, CA, _ = e2.models(beta)
        cons = [dict(kind="state", h=hA, b=12.0, p=0.9)]
        out[f"MF2026 beta={beta}, SOC design"] = select(A, cons, hA)
        out[f"MF2026 beta={beta}, coupling-agnostic design"] = select(CA, cons, hA)
    o = ROOT / "results" / "EXP023_terminal_params"; o.mkdir(parents=True, exist_ok=True)
    (o / "params.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

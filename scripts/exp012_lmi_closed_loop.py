"""EXP012 — closed loop of the LMI formulation [FS16] (binary initialisation) on benchmark D with 2000 runs.

Revision request: Table 1 used 500 runs for the LMI row (EXP001, runs_FS = 500) and only eps = 0.5. This script reruns
the LMI row with the same number of runs as the SOC rows (2000) for eps = 0.5 (Table 1) and eps = 0.9 (the value with
the largest feasible set in Fig. 2). It reuses run_task of scripts/exp001_closed_loop.py unchanged, so the controller,
the noise (seed 7000 + 1000 * ix + r, matched with all Table 1 rows) and the verification protocol are identical;
the first 500 runs for eps = 0.5 reproduce EXP001.

Usage:  python scripts/exp012_lmi_closed_loop.py [--runs 2000] [--eps 0.5,0.9] [--workers 4]
"""
import argparse, importlib.util, json, pathlib, sys, warnings
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("exp001", str(ROOT / "scripts" / "exp001_closed_loop.py"))
e1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e1)


def run(task):
    eps, ix, runs = task
    e1.CFG["fs_eps"] = eps; e1.CFG["runs_FS"] = runs
    r = e1.run_task(("E1", "FS-DF", ix))
    r["eps"] = eps
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--eps", default="0.5,0.9")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "EXP012_lmi_closed_loop"))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = dict(e1.CFG, runs_FS=a.runs, eps_list=[float(e) for e in a.eps.split(",")])
    from cxsmpc.provenance import write_provenance
    write_provenance(out, cfg, __file__)
    tasks = [(eps, ix, a.runs) for eps in cfg["eps_list"] for ix in range(len(e1.CFG["x0"]))]
    with Pool(a.workers) as pool:
        res = pool.map(run, tasks)
    table = []
    for r in res:
        if r.get("infeasible_at_x0"):
            table.append(dict(eps=r["eps"], x0=r["x0"], infeasible_at_x0=True, k0_class=r["k0_class"])); continue
        n = r["runs"]; vc = np.array(r["viol_counts"])
        table.append(dict(eps=r["eps"], x0=r["x0"], runs=n, max_k_violation_rate=float(vc.max() / n),
                          max_k_cp95_upper=e1.cp_upper(int(vc.max()), n), argmax_k=int(vc.argmax()),
                          cost_mean=float(np.mean(r["costs"])), cost_std=float(np.std(r["costs"], ddof=1)),
                          first500_cost_mean=float(np.mean(r["costs"][:500])), logs=r["logs"]))
    (out / "summary.json").write_text(json.dumps(dict(config={k: v for k, v in cfg.items() if k != "protocol_v2"}, table=table), indent=1))
    for t in table:
        print(t)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()

"""Build EVIDENCE_LEDGER.yaml from result summaries (values are read from files, never typed by hand)."""
import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(p):
    p = ROOT / p
    return json.loads(p.read_text()) if p.exists() else None


ev = {}
ev["E000"] = dict(type="analytical", evidence_level="L0",
                  source=dict(file="theory/cx_smpc_theory.pdf", commit=None), experiment=dict(id="THEORY_P1"),
                  metrics=dict(proved_results="kernel lemma, exact SOC thm, expected cost, containment vs FS16, recursive feasibility, Thm A, Thm C, Cor rfC, Thm perfA, Thm perfC",
                               independent_review="pending (theory/PROOF_REVIEW_CHECKLIST.md)"),
                  scope=dict(environment="analytical"), verified=False)
s0 = load("results/EXP000/summary.json")
if s0:
    m = {"AP002_stationary_cov_computed": s0["AP002_stationary_cov"]["computed"]}
    for b in ("A", "D"):
        a = s0[b]["audit"]
        m[f"{b}_audit_kernel_max_rel_err"] = a["kernel_max_rel_err"]
        m[f"{b}_audit_cantelli_max_violation"] = a["cantelli_max_violation"]
        m[f"{b}_audit_cost_max_rel_err"] = a["cost_max_rel_err"]
        for nz, r in s0[b]["noise"].items():
            m[f"{b}_{nz}_median_rel_frob_err"] = r["median_rel_frob_err"]
            m[f"{b}_{nz}_frac_abs_z_gt3"] = r["frac_abs_z_gt3"]
    ev["E001"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP000/summary.json", commit=None), experiment=dict(id="EXP000"),
                      protocol=dict(simulator="numpy Monte Carlo", controller="CX-SMPC plans (reset init)", scenario="benchmarks A (AP002) and D (Primbs-Sung)"),
                      metrics=m, scope=dict(environment="numerical simulation", trials=20, mc_samples_per_plan=20000), verified=False)
s0b = load("results/EXP000/check_M1e5/summary.json")
if s0b:
    ev["E002"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP000/check_M1e5/summary.json", commit=None), experiment=dict(id="EXP000"),
                      metrics={f"D_{nz}_{k}": r[k] for nz, r in s0b["D"]["noise"].items() for k in ("median_rel_frob_err", "frac_abs_z_gt3")},
                      scope=dict(environment="numerical simulation", trials=20, mc_samples_per_plan=100000), verified=False)
s3 = load("results/EXP003_pilot_v4_scaled/summary.json")  # v4: objective scaling (supersedes v3)
if s3:
    p1 = s3["part1"]
    m = dict(containment_violations_feasibility_verified=p1["containment_violations_feasibility_verified"],
             containment_violations_cost_verified=p1["containment_violations_cost_verified"],
             area_fractions=p1["area_fractions"])
    for row in s3["part2"]:
        key = f"n{row['n']}_N{row['N']}"
        m[f"{key}_cx_ms_p50"], m[f"{key}_cx_ms_p99"] = row["cx_ms"]["p50"], row["cx_ms"]["p99"]
        m[f"{key}_fs_ms_p50"], m[f"{key}_fs_ms_p99"] = row["fs_ms"]["p50"], row["fs_ms"]["p99"]
        m[f"{key}_median_ratio_fs_over_cx"] = row["median_ratio_fs_over_cx"]
    ev["E003"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP003_pilot_v4_scaled/summary.json", commit=None), experiment=dict(id="EXP003"),
                      protocol=dict(simulator="none (optimization only)", controller="CX-SMPC SOCP vs FS16 fixed-K SDP, sparse formulation, Clarabel (SCS fallback), a-posteriori verified",
                                    scenario="part1: benchmark D 17x17 grid, eps sweep; part2: random systems n=2..10, N=10..30, 5 systems x 4 states"),
                      metrics=m, scope=dict(environment="numerical, AMD Ryzen 7 5800H, non-RT Linux", trials=20, label="PILOT"), verified=False)
s4 = load("results/EXP003b_hybrid_pilot_v2_scaled/summary.json")  # v2: objective scaling (part A region)
s4f = load("results/EXP003b_final/summary.json")  # FINAL timing (100 per config)
if s4:
    m = dict(partA_area=s4["A"]["area"], partA_checks_vs_CX=s4["A"]["checks_vs_CX"])
    for row in (s4f["B"] if s4f else s4["B"]):
        for meth in ("CX", "DM3", "DM5", "HY3", "FS0.5"):
            if meth in row:
                m[f"n{row['n']}_N{row['N']}_{meth}_ms_p50"] = row[meth]["p50"]
                m[f"n{row['n']}_N{row['N']}_{meth}_verified"] = row[meth]["verified"]
    ev["E004"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP003b_final/summary.json", commit=None), experiment=dict(id="EXP003b"),
                      source_partA=dict(file="results/EXP003b_hybrid_pilot_v2_scaled/summary.json"),
                      protocol=dict(simulator="none (optimization only)", controller="CX vs dual-mode DM(Nc) vs trace-chain HY(Ne) vs FS16; Clarabel; verified",
                                    scenario="A: benchmark D N=30 17x17 grid (scaled pilot); B FINAL: random systems n=2..10, N=10..30, 10 systems x 10 states"),
                      metrics=m, scope=dict(environment="numerical, AMD Ryzen 7 5800H, non-RT Linux", trials=100, label="FINAL (B) / PILOT (A)"), verified=False)
s5 = load("results/EXP001_closed_loop_final/summary.json")  # FINAL (2000 runs; FS 500)
if s5:
    m = {"thmC_feasibility": s5["thmC_feasibility"]}
    for r in s5["table"]:
        key = f"{r['exp']}_{r['method']}_x0={r['x0'][0]},{r['x0'][1]}"
        if r.get("infeasible_at_x0"):
            m[key] = "infeasible at x0 (" + r["k0_class"] + ")"; continue
        m[key] = dict(runs=r["runs"], max_k_violation=r["max_k_violation_rate"], cp95_upper=r["max_k_cp95_upper"],
                      time_avg_violation=r["time_avg_violation_rate"], cost_mean=r["cost_mean"], cost_std=r["cost_std"],
                      unresolved=r["logs"].get("unresolved", 0), infeasible=r["logs"].get("infeasible", 0))
    ev["E005"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP001_closed_loop_final/summary.json", commit=None), experiment=dict(id="EXP001+EXP013"),
                      protocol=dict(simulator="numpy closed-loop Monte Carlo", controller="NOM, CX-A, CX-DF, FS16-DF, M24-IF (+E13 non-optimal K, tracking)",
                                    scenario="benchmark D, 4 x0, T=40, theta~N(0,1)"),
                      metrics=m, scope=dict(environment="numerical simulation", trials=2000, trials_FS=500, seeds="matched per (x0, run)", label="FINAL"), verified=False)
s6 = load("results/EXP002_mechanism_pilot_v4/summary.json")
if s6:
    m = {}
    for c in s6["cells"]:
        for meth in ("CX-A", "CA-A"):
            d = c[meth]
            m[f"s{c['s']}_r{c['r']}_{meth}"] = (dict(max_k_violation=d["max_k"], cp95_upper=d["max_k_cp95_upper"], time_avg=d["time_avg"],
                                                      cost_mean=d["cost_mean"], shift_steps=d["steps_shift_candidate"], fallback_steps=d["fallback_steps"])
                                                 if d.get("feasible_at_k0") else "declared infeasible at k=0")
    m["nominal_s1"] = s6["nominal_s1"]
    ev["E006"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP002_mechanism_pilot_v4/summary.json", commit=None), experiment=dict(id="EXP002"),
                      protocol=dict(simulator="vectorized closed-loop MC, true AP002 noise", controller="CX-A (exact coupling) vs CA-A (coupling-agnostic) vs NOM",
                                    scenario="benchmark A, x1<=12, x0=[0,0,r], r=0..20, Sp scale s=0.5..1.5, N=15, T=30, trace terminal"),
                      metrics=m, scope=dict(environment="numerical simulation", trials=2000, seeds="matched per cell", label="PILOT"), verified=False)
s7 = load("results/EXP004_005_ablation_envelope_pilot/summary.json")
if s7:
    m = {}
    for r in s7["det"]:
        g, bn, v, x0, nz, Kc, c = r["task"]
        m[f"{g}|{bn}|{v}|x0={x0}|{nz}|K={Kc}|c={c}"] = ({k: r[k] for k in ("max_k", "cp95_upper", "time_avg", "cost_mean", "cost_std", "shift_steps")}
                                                         if r["status"] == "ok" else r["status"])
    for r in s7["A3"]:
        m[f"A3|{r['task'][1]}|x0={r['task'][2]}"] = dict(max_k=r["max_k"], cp95_upper=r["cp95_upper"], cost_mean=r["cost_mean"], logs=r["logs"])
    ev["E007"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP004_005_ablation_envelope_pilot/summary.json", commit=None), experiment=dict(id="EXP004+EXP005"),
                      protocol=dict(simulator="vectorized closed-loop MC (det. nominal) + per-run (A3)", controller="CX-A variants: exact/nominal cost, Cantelli/Gaussian kappa, K choice; A/DF/reset-only",
                                    scenario="benchmarks A (x1<=12) and D, trace terminal; tails Laplace/t5/t3; noise scale x1.5-3"),
                      metrics=m, scope=dict(environment="numerical simulation", trials=2000, label="PILOT"), verified=False)
s8 = load("results/EXP008_quadrotor_pilot/summary.json")
if s8:
    m = {}
    for r in s8["results"]:
        meth, z0, nz = r["task"]
        m[f"{meth}|z0={z0}|{nz}"] = ({k: r[k] for k in ("ceiling_max_k", "ceiling_cp95_upper", "input_viol_max_k", "saturation_frac", "cost_mean", "final_abs_z_mean")}
                                     if r["status"] == "ok" else r["status"])
    ev["E008"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP008_quadrotor_pilot/summary.json", commit=None), experiment=dict(id="EXP008"),
                      protocol=dict(simulator="vectorized linear closed-loop MC with physical thrust saturation", controller="CX-A, CA-A, FS-A, NOM",
                                    scenario="ILLUSTRATIVE quadrotor altitude (mass/thrust from NanoBench; drag and noise levels ASSUMED), climbs 0.5-3 m"),
                      metrics=m, scope=dict(environment="numerical simulation (L1), not physics engine, not flight", trials=2000, label="PILOT"), verified=False)
s9p = ROOT / "results/EXP009_illustrations/data.json"
if s9p.exists():
    d9 = json.load(open(s9p))
    a9, b9 = d9["a"], d9["b"]
    m9 = {f"a|{k}": dict(true_bound_max=max(z + a9["kappa"] * s for z, s in zip(a9[k]["hz"], a9[k]["std_true"])),
                         believed_bound_max=max(z + a9["kappa"] * s for z, s in zip(a9[k]["hz"], a9[k]["std_believed"])),
                         mc_q90_max=max(a9[k]["mc_q90"]), mc_viol_max=max(a9[k]["mc_viol"])) for k in ("CX", "CA")}
    m9.update({f"b|{k}": dict(viol_max_k=max(v["viol_freq"]), argmax_k=int(max(range(len(v["viol_freq"])), key=v["viol_freq"].__getitem__)), logs=v["logs"])
               for k, v in b9.items()})
    ev["E009"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP009_illustrations/data.json", commit=None), experiment=dict(id="EXP009"),
                      protocol=dict(simulator="(a) open-loop MC of predicted closed loop, 20000 samples; (b) closed-loop MC, 2000 runs, matched noise",
                                    controller="(a) CX vs coupling-agnostic first-step plans; (b) CX-A, CX-DF, NOM",
                                    scenario="(a) benchmark A, s=1, x0=(0,0,16), N=15; (b) benchmark D, x0=(0,1.5), N=10, T=40"),
                      metrics=m9, scope=dict(environment="numerical simulation (L1)", trials=2000,
                                             label="ILLUSTRATION ONLY, added after the 2026-10-03 freeze; (b) reproduces E1 final"), verified=False)
s10p = ROOT / "results/EXP010_terminal_rerun/summary.json"
if s10p.exists():
    d10 = json.load(open(s10p))
    m10 = {f"{c['method']}|x0={c['x0']}|{'terminal' if c['terminal'] else 'no-terminal'}":
           ({k: c[k] for k in ("max_k_violation", "max_k_cp95_upper", "argmax_k", "cost_mean", "cost_std", "logs")} if c["status"] == "ok" else c["status"])
           for c in d10["cells"]}
    ev["E010"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP010_terminal_rerun/summary.json", commit=None), experiment=dict(id="EXP010"),
                      protocol=dict(simulator="closed-loop MC, benchmark D, matched noise identical to EXP001 (Table 1)",
                                    controller="CX-A, CX-DF WITH trace-terminal ingredients (Assumption 3)", scenario="4 initial states of Table 1, N=10, T=40"),
                      metrics=m10, scope=dict(environment="numerical simulation (L1)", trials=d10["config"]["runs"],
                                              label="REVISION run, added after the 2026-10-03 freeze"), verified=False)
s11p = ROOT / "results/EXP011_problem_sizes/sizes.json"
if s11p.exists():
    d11 = json.load(open(s11p))
    ev["E011"] = dict(type="structural_measurement", evidence_level="L0",
                      source=dict(file="results/EXP011_problem_sizes/sizes.json", commit=None), experiment=dict(id="EXP011"),
                      protocol=dict(simulator="CVXPY get_problem_data (Clarabel conic form)", controller="CX, DM3, DM5, FS0.5",
                                    scenario="EXP003b timing grid, one random system per (n, N)"),
                      metrics={f"n={r['n']}|N={r['N']}|{r['method']}": {k: r[k] for k in ("variables", "rows", "nnz", "n_soc", "n_psd", "max_psd")} for r in d11["rows"]},
                      scope=dict(environment="problem data only, no solve statistics", trials=1, label="REVISION, added after the freeze"), verified=False)
s12p = ROOT / "results/EXP012_lmi_closed_loop/summary.json"
if s12p.exists():
    d12 = json.load(open(s12p))
    ev["E012"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP012_lmi_closed_loop/summary.json", commit=None), experiment=dict(id="EXP012"),
                      protocol=dict(simulator="closed-loop MC via run_task of exp001 (identical protocol and noise)",
                                    controller="FS-DF (LMI formulation, binary init.), eps = 0.5 and 0.9", scenario="4 initial states of Table 1, N=10, T=40"),
                      metrics={f"eps={t['eps']}|x0={t['x0']}": ({k: t[k] for k in ("max_k_violation_rate", "max_k_cp95_upper", "cost_mean", "cost_std", "first500_cost_mean")}
                                                              if not t.get("infeasible_at_x0") else f"not initialised ({t['k0_class']})") for t in d12["table"]},
                      scope=dict(environment="numerical simulation (L1)", trials=d12["config"]["runs_FS"], label="REVISION run, added after the freeze"), verified=False)
for _N in (20, 30):
    _p = ROOT / f"results/EXP013_terminal_dualmode_N{_N}/summary.json"
    if _p.exists():
        _d = json.load(open(_p))
        ev.setdefault("E013", dict(type="simulation_result", evidence_level="L1",
                                   source=dict(file="results/EXP013_terminal_dualmode_N{20,30}/summary.json", commit=None), experiment=dict(id="EXP013"),
                                   protocol=dict(simulator="closed-loop MC, benchmark D, noise identical to Table 1",
                                                 controller="CX-A, CX-DF, dual mode Nc=3, WITH trace-terminal ingredients", scenario="4 initial states of Table 1, T=40"),
                                   metrics={}, scope=dict(environment="numerical simulation (L1)", trials=_d["config"]["runs"],
                                                          label="REVISION-v2 run, added after the freeze; not in the paper"), verified=False))
        for c in _d["cells"]:
            ev["E013"]["metrics"][f"N={_N}|{c['method']}|x0={c['x0']}"] = ({k: c[k] for k in ("max_k_violation", "max_k_cp95_upper", "cost_mean", "cost_std", "logs")}
                                                                          if c["status"] == "ok" else c["status"])
_p14 = ROOT / "results/EXP014_problem_sizes_all/sizes_all.json"
if _p14.exists():
    _d14 = json.load(open(_p14))
    ev["E014"] = dict(type="structural_measurement", evidence_level="L0",
                      source=dict(file="results/EXP014_problem_sizes_all/sizes_all.json", commit=None), experiment=dict(id="EXP014"),
                      protocol=dict(simulator="CVXPY get_problem_data, the 10 timed EXP003b instances per size", controller="CX, DM3, FS0.5",
                                    scenario="Table 3 sizes (n, N) in {4, 10} x {10, 30}"),
                      metrics=_d14["summary"], scope=dict(environment="problem data only", trials=10, label="REVISION-v2, added after the freeze"), verified=False)
_p15 = ROOT / "results/EXP015_terminal_N30_Nc5/summary.json"
if _p15.exists():
    _d15 = json.load(open(_p15))
    ev["E015"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP015_terminal_N30_Nc5/summary.json", commit=None), experiment=dict(id="EXP015"),
                      protocol=dict(simulator="closed-loop MC, benchmark D, noise identical to Table 1",
                                    controller="CX-A, CX-DF, dual mode N=30, Nc=5, WITH trace-terminal ingredients", scenario="4 initial states of Table 1, T=40"),
                      metrics={f"{c['method']}|x0={c['x0']}": ({k: c[k] for k in ("max_k_violation", "max_k_cp95_upper", "cost_mean", "cost_std", "logs")}
                                                              if c["status"] == "ok" else c["status"]) for c in _d15["cells"]},
                      scope=dict(environment="numerical simulation (L1)", trials=_d15["config"]["runs"], label="REVISION-v3 run, added after the freeze; not in the paper"), verified=False)
_p16 = ROOT / "results/EXP016_table1_final/summary.json"
if _p16.exists():
    _d16 = json.load(open(_p16))
    ev["E016"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP016_table1_final/summary.json", commit=None), experiment=dict(id="EXP016"),
                      protocol=dict(simulator="closed-loop MC, benchmark D, Table 1 seeds, one invocation of scripts/exp016_table1_final.py",
                                    controller="all Table 1 rows (NOM, CX-A, CX-DF, FS-DF eps=0.5, M24-IF check, terminal N=10, terminal dual mode N=30 Nc=5)",
                                    scenario="4 initial states, T=40"),
                      metrics=dict(rows={f"{r['row']}|x0={r['x0']}": ({k: r[k] for k in ("max_k_violation", "max_k_cp95_upper", "cost_mean", "cost_std", "logs")}
                                                                    if r["status"] == "ok" else r["status"]) for r in _d16["rows"]},
                                   m24_vs_cxa_max_abs_cost_diff=_d16["m24_vs_cxa_max_abs_cost_diff"]),
                      scope=dict(environment="numerical simulation (L1)", trials=2000, label="REVISION-v4 final Table 1 run, added after the freeze"), verified=False)
_p17 = ROOT / "results/EXP016_noterminal_N30_Nc5/summary.json"
if _p17.exists():
    _d17 = json.load(open(_p17))
    ev["E017"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP016_noterminal_N30_Nc5/summary.json", commit=None), experiment=dict(id="EXP016-ref"),
                      protocol=dict(simulator="closed-loop MC, benchmark D, Table 1 seeds", controller="CX-A, CX-DF, dual mode N=30 Nc=5, WITHOUT terminal constraints",
                                    scenario="4 initial states, T=40 (reference for cost ratios; not in Table 1)"),
                      metrics={f"{c['method']}|x0={c['x0']}": {k: c[k] for k in ("max_k_violation", "max_k_cp95_upper", "cost_mean", "cost_std", "logs")} for c in _d17["cells"]},
                      scope=dict(environment="numerical simulation (L1)", trials=_d17["config"]["runs"], label="REVISION-v4 reference run, added after the freeze"), verified=False)
_p18 = ROOT / "results/EXP017_full_N30/summary.json"
if _p18.exists():
    _d18 = json.load(open(_p18))
    ev["E018"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP017_full_N30/summary.json", commit=None), experiment=dict(id="EXP017"),
                      protocol=dict(simulator="closed-loop MC, benchmark D, Table 1 seeds", controller="CX-A full horizon N=30, with and without trace-terminal ingredients",
                                    scenario="4 initial states, T=40 (horizon vs dual-mode decomposition; not in Table 1)"),
                      metrics={f"x0={c['x0']}|{'terminal' if c['terminal'] else 'no-terminal'}": ({k: c[k] for k in ("max_k_violation", "max_k_cp95_upper", "cost_mean", "cost_std", "logs")}
                                                                                                 if c["status"] == "ok" else c["status"]) for c in _d18["cells"]},
                      scope=dict(environment="numerical simulation (L1)", trials=_d18["config"]["runs"], label="REVISION-v5 run, added after the freeze; not in the paper"), verified=False)
_p19 = ROOT / "results/EXP019_table1_final/summary.json"
if _p19.exists():
    _d19 = json.load(open(_p19))
    ev["E019"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP019_table1_final/summary.json", commit=None), experiment=dict(id="EXP019"),
                      protocol=dict(simulator="closed-loop MC, benchmark D, Table 1 seeds, one invocation of exp016_table1_final.py --n30 full",
                                    controller="all Table 1 rows; terminal rows N=10 and full-horizon N=30", scenario="4 initial states, T=40"),
                      metrics=dict(rows={f"{r['row']}|x0={r['x0']}": ({k: r[k] for k in ("max_k_violation", "max_k_cp95_upper", "cost_mean", "cost_std", "logs")}
                                                                    if r["status"] == "ok" else r["status"]) for r in _d19["rows"]},
                                   m24_vs_cxa_max_abs_cost_diff=_d19["m24_vs_cxa_max_abs_cost_diff"]),
                      scope=dict(environment="numerical simulation (L1)", trials=2000, label="REVISION-v6 final Table 1 run, added after the freeze"), verified=False)
_p20 = ROOT / "results/EXP018_full_N30_binary/summary.json"
if _p20.exists():
    _d20 = json.load(open(_p20))
    ev["E020"] = dict(type="simulation_result", evidence_level="L1",
                      source=dict(file="results/EXP018_full_N30_binary/summary.json", commit=None), experiment=dict(id="EXP018"),
                      protocol=dict(simulator="closed-loop MC, benchmark D, Table 1 seeds", controller="CX-DF full horizon N=30 with terminal ingredients",
                                    scenario="4 initial states, T=40 (step-1 run; identical to the corresponding E019 rows)"),
                      metrics={f"x0={c['x0']}": {k: c[k] for k in ("max_k_violation", "max_k_cp95_upper", "cost_mean", "cost_std", "logs")} for c in _d20["cells"]},
                      scope=dict(environment="numerical simulation (L1)", trials=_d20["config"]["runs"], label="REVISION-v6, added after the freeze"), verified=False)
ledger = dict(project=dict(name="CX-SMPC multiplicative-noise SMPC", domain="control"), evidence=ev)
(ROOT / "EVIDENCE_LEDGER.yaml").write_text("# Generated by scripts/build_ledger.py — do not edit values by hand.\n# FROZEN 2026-10-03 (ARIS Stage 16): no edits without a new audit record.\n# Audit 2026-10-03: E009 (EXP009 figure data, illustration only) appended after the freeze; E000-E008 unchanged.\n# Audit 2026-10-04: E010-E012 (revision-v1 runs) appended after the freeze; E000-E009 unchanged.\n# Audit 2026-10-05: E013-E014 (revision-v2 runs) appended after the freeze; E000-E012 unchanged.\n# Audit 2026-10-06: E015 (revision-v3 run) appended after the freeze; E000-E014 unchanged.\n# Audit 2026-10-06: E016-E017 (revision-v4 runs) appended after the freeze; E000-E015 unchanged.\n# Audit 2026-10-06: E018 (revision-v5 run) appended after the freeze; E000-E017 unchanged.\n# Audit 2026-10-06: E019-E020 (revision-v6 runs) appended after the freeze; E000-E018 unchanged.\n" + yaml.safe_dump(ledger, sort_keys=False, width=140))
print("ledger records:", list(ev))

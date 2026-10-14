# Terminal rows with a longer horizon: dual-mode variant (Proposition 4)

- **Generated:** 2026-10-05, branch `revision-v2`, item 2.
- **Status:** **not in the paper**; the author decides whether to add these rows to Table 1.

## Protocol
- **Script:** `scripts/exp010_terminal_rerun.py --runs 2000 --T 40 --N {20,30} --Nc 3`.
- **Setup:** benchmark D, p = 0.9, the same 4 initial states as Table 1, 2000 runs of length 40.
- **Terminal ingredients (Assumption 3):** trace terminal plus ellipsoidal Z_f, 2 % margin, best (λ, H₀) on the grid; identical to the Table 1 "terminal" rows.
- **Noise:** seeds `7000 + 1000·ix + r`, identical to all Table 1 rows (EXP001, EXP010, EXP012).
- **Code change required:** `DualModeCXProblem` (`cxsmpc/ocp.py`) did not support the trace terminal constraint tr(SΣ_N) ≤ γ̄. It was added as one SOC, with the tail j ≥ N_c collapsed on z_{N_c} exactly as in Proposition 4, plus the matching a-posteriori residual. **Validation:** with N_c = N = 10, the dual-mode problem reproduces the full-horizon EXP010 results exactly (costs equal to 6 decimals, identical solver logs, 5 runs × 4 x₀ × 2 schemes).
- **Raw output:** `results/EXP013_terminal_dualmode_N20/summary.json`, `results/EXP013_terminal_dualmode_N30/summary.json` (ledger record E013).

## Results
Worst-step violation frequency in % (upper limit of the 95 % Clopper–Pearson interval) and mean closed-loop cost Σ_{k<40} ‖x_k‖²_Q + ‖u_k‖²_R. Admissible level: 10 %.

| Scheme | | x₀=(−0.1,1.2) | x₀=(−0.1,1.3) | x₀=(−0.1,1.4) | x₀=(0,1.5) |
|---|---|---|---|---|---|
| SOC scheme, terminal, N = 10 (Table 1, EXP010) | violation | 0.10 [0.36] | 0.05 [0.28] | infeasible | infeasible |
| | cost | 2509 | 4586 | – | – |
| SOC binary init., terminal, N = 10 (Table 1, EXP010) | violation | 0.15 [0.44] | 0.15 [0.44] | infeasible | infeasible |
| | cost | 2410 | 4464 | – | – |
| **SOC scheme, terminal, dual mode N = 20, N_c = 3** | violation | 0.25 [0.58] | 0.25 [0.58] | 0.65 [1.11] | 0.60 [1.05] |
| | cost | 2384 | 4590 | 8277 | 11871 |
| **SOC binary init., terminal, dual mode N = 20, N_c = 3** | violation | 0.05 [0.28] | 0.00 [0.18] | 0.10 [0.36] | 0.05 [0.28] |
| | cost | 2370 | 4596 | 8200 | 11653 |
| **SOC scheme, terminal, dual mode N = 30, N_c = 3** | violation | 0.25 [0.58] | 0.25 [0.58] | 0.65 [1.11] | 0.60 [1.05] |
| | cost | 2384 | 4590 | 8277 | 11871 |
| **SOC binary init., terminal, dual mode N = 30, N_c = 3** | violation | 0.05 [0.28] | 0.00 [0.18] | 0.10 [0.36] | 0.05 [0.28] |
| | cost | 2370 | 4596 | 8200 | 11653 |
| SOC scheme, no terminal, N = 10 (Table 1, EXP001), for reference | violation | 0.85 [1.36] | 1.05 [1.60] | 1.75 [2.43] | 2.35 [3.11] |
| | cost | 179 | 444 | 1052 | 2232 |

## Observations (all numbers from the files above)
1. **Feasibility.** With the dual-mode variant (N = 20 or 30, N_c = 3), the terminal scheme is feasible at **all four** initial states, including x₀ = (−0.1, 1.4) and (0, 1.5), where it is infeasible with N = 10 (Table 1).
2. **Constraint satisfaction.**
   - Worst-step violation is ≤ 0.65 % (upper limit ≤ 1.11 %) for Algorithm 1 and ≤ 0.10 % (≤ 0.36 %) with binary initialisation, against the admissible 10 %.
   - In all 2 × 4 × 2000 runs per horizon: 0 infeasible steps, 0 unresolved solver outputs, 0 uses of the shifted plan (solver logs in the summary files).
3. **Cost.** Mean cost relative to the terminal-free scheme with N = 10 (same noise):
   - Algorithm 1: 2384/179 = 13.3, 4590/444 = 10.3, 8277/1052 = 7.9, 11871/2232 = 5.3.
   - Binary initialisation (against the terminal-free binary row 133, 370, 1020, 2193): 17.8, 12.4, 8.0, 5.3.
   - Against the N = 10 terminal rows: cost changes by −5.0 % / +0.1 % (Algorithm 1) and −1.6 % / +3.0 % (binary) at the two common initial states. The dual-mode problem has only 3 free moves, so a longer horizon does not necessarily lower the cost.
4. **N = 20 and N = 30 give the same results.** This is expected, not a bug:
   - *Verified at x₀ = (−0.1, 1.4).* The optimal costs are 13283.508668 vs 13283.508669; the first three moves differ by at most 7.0e-6 (solver tolerance); and both terminal constraints are inactive at both horizons (zᵀE_f z − 1 = −0.22 / −0.61; tr(SΣ_N) − γ̄ = −1.02 / −1.17).
   - *Why the extra steps cost nothing.* P satisfies 𝓛\*(P) + Q + KᵀRK = P with residual 6.1e-11 (linear-quadratic P). Under v_t = K z_t, appending prediction steps therefore leaves the expected cost unchanged: stage cost plus new terminal cost equals the old terminal cost. With the terminal constraints inactive, the optimiser is the same for N = 20 and N = 30. Only the N = 20 rows would be needed in a table.
5. **Suggested wording if added to Table 1** (author decision): one row pair "SOC scheme, terminal, dual mode (N = 20, N_c = 3)", with a caption note that N = 30 gave identical values.

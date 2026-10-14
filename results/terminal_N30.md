# Terminal ingredients with the dual-mode variant, N = 30 (revision-v3, item 1)

- **Generated:** 2026-10-06.
- **Status:** **not in the paper**; Table 1 is unchanged until the author approves.

## Protocol
- **Setup:** benchmark of Section 6.1 (benchmark D), p = 0.9, the 4 initial states of Table 1, 2000 runs of length T = 40.
- **Noise:** seeds `7000 + 1000·ix + r`, identical to all Table 1 rows (EXP001, EXP010, EXP012).
- **Terminal ingredients (Assumption 3):** trace terminal plus ellipsoidal Z_f, 2 % margin, best (λ, H₀) on the grid; the same design as the Table 1 "terminal" rows.
- **Scheme:** dual-mode variant (Proposition 4) with N = 30, including the trace terminal constraint (validated: N_c = N reproduces the full-horizon problem exactly).
- **Commands:** `python scripts/exp010_terminal_rerun.py --runs 2000 --T 40 --N 30 --Nc {3|5}`.
- **Raw data:**
  - N_c = 3: `results/EXP013_terminal_dualmode_N30/summary.json` (run 2026-10-05). The protocol is identical, and all source hashes in its provenance match the current code, so it was not rerun.
  - N_c = 5: `results/EXP015_terminal_N30_Nc5/summary.json`, new. Ledger records E013 and E015.

## Results
Worst-step violation frequency in % [upper limit of the 95 % Clopper–Pearson interval] and mean closed-loop cost Σ_{k<40} ‖x_k‖²_Q + ‖u_k‖²_R. The **ratio** is the mean cost divided by that of the terminal-free SOC scheme with the same initialisation in Table 1 (N = 10, same noise): 179 / 444 / 1052 / 2232 for Algorithm 1 and 133 / 370 / 1020 / 2193 for binary initialisation. Admissible violation level: 10 %.

| Scheme | | (−0.1, 1.2) | (−0.1, 1.3) | (−0.1, 1.4) | (0, 1.5) |
|---|---|---|---|---|---|
| Algorithm 1, terminal, N = 30, N_c = 3 | violation | 0.25 [0.58] | 0.25 [0.58] | 0.65 [1.11] | 0.60 [1.05] |
| | cost | 2384 | 4590 | 8277 | 11871 |
| | ratio | 13.3 | 10.3 | 7.9 | 5.3 |
| binary init., terminal, N = 30, N_c = 3 | violation | 0.05 [0.28] | 0.00 [0.18] | 0.10 [0.36] | 0.05 [0.28] |
| | cost | 2370 | 4596 | 8200 | 11653 |
| | ratio | 17.9 | 12.4 | 8.0 | 5.3 |
| Algorithm 1, terminal, N = 30, N_c = 5 | violation | 0.25 [0.58] | 0.40 [0.79] | 0.55 [0.98] | 0.70 [1.17] |
| | cost | 1154 | 2307 | 4160 | 6234 |
| | ratio | 6.5 | 5.2 | 4.0 | 2.8 |
| binary init., terminal, N = 30, N_c = 5 | violation | 0.05 [0.28] | 0.05 [0.28] | 0.20 [0.51] | 0.10 [0.36] |
| | cost | 1128 | 2296 | 4146 | 6190 |
| | ratio | 8.5 | 6.2 | 4.1 | 2.8 |
| *Table 1: Algorithm 1, terminal, N = 10 (full horizon)* | violation | 0.10 [0.36] | 0.05 [0.28] | infeasible | infeasible |
| | cost | 2509 | 4586 | – | – |

## Summary
1. **Feasible states.** All four initial states are feasible for both N_c = 3 and N_c = 5 (with N = 10: only two). Over all 2 × 4 × 2000 runs per N_c, there were 0 infeasible steps, 0 unresolved solver outputs and 0 uses of the shifted plan.
2. **Worst-step violation.**
   - N_c = 3: ≤ 0.65 % (upper limit ≤ 1.11 %) for Algorithm 1 and ≤ 0.10 % (≤ 0.36 %) with binary initialisation.
   - N_c = 5: ≤ 0.70 % (≤ 1.17 %) and ≤ 0.20 % (≤ 0.51 %).
   - The admissible level is 10 %.
3. **Cost.**
   - N_c = 5 roughly halves the cost of N_c = 3: 1154 vs 2384, 2307 vs 4590, 4160 vs 8277, 6234 vs 11871 for Algorithm 1.
   - Ratio to the terminal-free SOC scheme: 2.8–6.5 (Algorithm 1) and 2.8–8.5 (binary initialisation) with N_c = 5, against 5.3–13.3 and 5.3–17.9 with N_c = 3.
   - With N_c = 5, Algorithm 1 at x₀ = (−0.1, 1.2) costs 1154, against 2509 for the N = 10 full-horizon terminal row of Table 1.
4. **Caveat on the ratio.** The reference is the terminal-free scheme with N = 10, so the ratio combines the effect of the terminal ingredients and of the different horizon and free-move structure. It is not a pure measure of terminal conservatism.
5. **Note.** N = 20 with N_c = 3 (`results/EXP013_terminal_dualmode_N20`) gives the same values as N = 30 with N_c = 3; see `results/terminal_dualmode.md`.

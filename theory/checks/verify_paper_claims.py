"""Numerical checks for proof steps of the SCL manuscript not covered by verify_identities.py.

Not an experiment and not a proof: random-instance checks that catch algebra/sign errors.
  C1  Thm 3(ii): terminal-part change Delta <= tr(PW) under L*(P) + Q + K'RK <= P (tube form, random Sigma_N, z_N).
  C2  Prop. 4 (dual mode): sum_{j>=Nc} xi_j' M_{t,j} xi_j == z_Nc' Mtilde_t z_Nc when v_j = K z_j.
  C3  Prop. 3 (IF with LQ gain): cross term Lambda of the conditional cost in (x0, c) vanishes for the LQ gain and
      not for a non-optimal gain (exact second moments propagated, no sampling).
  C4  Thm 2: shifted plan of a solved instance (benchmark D with terminal ingredients) satisfies all constraints of
      the shifted problem, evaluated with the exact covariance recursion.
  C5  Containment rescaling: FS16 (22a) normalised to b = 1 is the eps-family of eq. (fs-lin) with eps' = b*eps (b > 0).
"""
import pathlib, sys, warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")
from cxsmpc.model import MultModel, mult_lqr, terminal_weight  # noqa: E402

rng = np.random.default_rng(42)
ok_all = True


def report(name, ok, detail):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def rand_model(n=3, m=2, q=2, scale=0.15):
    for _ in range(100):
        A0 = rng.normal(size=(n, n)); A0 *= 0.95 / max(abs(np.linalg.eigvals(A0)))
        A = [A0] + [scale * rng.normal(size=(n, n)) for _ in range(q)]
        B = [rng.normal(size=(n, m))] + [scale * rng.normal(size=(n, m)) for _ in range(q)]
        try:
            K, P = mult_lqr(A, B, np.eye(n), np.eye(m))
        except RuntimeError:
            continue
        W = (lambda M: M @ M.T)(rng.normal(size=(n, n))) * 0.1
        return MultModel(A, B, W, K), K, P
    raise RuntimeError


# ---------------- C1
mdl, K, P = rand_model()
Q, R = np.eye(3), np.eye(2)
Sq = Q + K.T @ R @ K
worst = -np.inf
for _ in range(500):
    zN = rng.normal(size=3); SN = (lambda M: M @ M.T)(rng.normal(size=(3, 3)))
    xi = np.r_[zN, K @ zN]
    Snew = mdl.L(SN) + mdl.D(np.outer(xi, xi)) + mdl.W
    AK = mdl.A[0] + mdl.B[0] @ K
    delta = (zN @ Sq @ zN + np.trace(Sq @ SN) + (AK @ zN) @ P @ (AK @ zN) + np.trace(P @ Snew)
             - zN @ P @ zN - np.trace(P @ SN))
    worst = max(worst, delta - np.trace(P @ mdl.W))
report("C1 terminal decrease (Thm 3(ii))", worst <= 1e-8, f"max(Delta - tr(PW)) = {worst:.2e} (LQ P: equality expected)")

# ---------------- C2
from cxsmpc.kernels import DirectionKernel  # noqa: E402
N, Nc = 12, 4
h = rng.normal(size=3)
ker = DirectionKernel(mdl, h, N)
AK = mdl.A[0] + mdl.B[0] @ K
z = [rng.normal(size=3)]; v = []
for t in range(N):
    vt = rng.normal(size=2) if t < Nc else K @ z[t]
    v.append(vt); z.append(mdl.A[0] @ z[t] + mdl.B[0] @ vt)
err = 0.0
for t in range(Nc + 1, N + 1):
    lhs = sum(np.sum((ker.R[t - 1 - j] @ np.r_[z[j], v[j]]) ** 2) for j in range(Nc, t))
    Mt = np.zeros((3, 3))
    for j in range(Nc, t):
        T = np.vstack([np.eye(3), K]) @ np.linalg.matrix_power(AK, j - Nc)
        Mt += T.T @ (ker.R[t - 1 - j].T @ ker.R[t - 1 - j]) @ T
    err = max(err, abs(lhs - z[Nc] @ Mt @ z[Nc]) / max(1.0, lhs))
report("C2 dual-mode block collapse (Prop. 4)", err < 1e-10, f"max rel. error {err:.2e}")


# ---------------- C3
def cost_form(mdl, Kpol, P, Q, R, N):
    """Exact conditional cost J(x0, c) = E[sum_{t<N} |x|_Q^2 + |u|_R^2 + |x_N|_P^2], u = K x + c_t, as a quadratic form
    in y = (x0, c_0..c_{N-1}), W set to zero (constants do not affect Lambda). Built by propagating
    second moments of the augmented linear map x_t = F_t y + noise-driven parts."""
    n, m = mdl.n, mdl.m
    dim = n + N * m
    # second-moment matrix of the stacked vector s_t = [x_t; y] evolves linearly; compute J(y) = y' Hm y exactly.
    Hm = np.zeros((dim, dim))
    # E[x_t x_t'] as a linear function of Y = y y' : represent operator by its action on basis via vectorisation
    def step(Xxx, Xxy, Yyy, t):
        # x+ = A_K x + B c_t + sum_i (Phi_i x + B_i c_t) theta_i, with Phi_i = A_i + B_i K
        Ec = np.zeros((m, dim)); Ec[:, n + t * m:n + (t + 1) * m] = np.eye(m)
        Phi = [mdl.A[i] + mdl.B[i] @ Kpol for i in range(mdl.q + 1)]
        # x = (Mx) y-symbolic: track Gx (n x dim) mean map and covariance part S (from noise) as quadratic in y
        return Phi, Ec
    # Symbolic propagation: x_t = G_t y (mean part) + eta_t ; E[eta eta'] = Sig_t(Y) linear in Y.
    G = np.zeros((n, dim)); G[:, :n] = np.eye(n)
    Sig_ops = []  # list of (Lmat) such that Sig_t = sum_k Lk Y Lk'
    Phi = [mdl.A[i] + mdl.B[i] @ Kpol for i in range(mdl.q + 1)]
    for t in range(N):
        Ec = np.zeros((m, dim)); Ec[:, n + t * m:n + (t + 1) * m] = np.eye(m)
        U = Kpol @ G + Ec                      # mean input map
        Hm += G.T @ Q @ G + U.T @ R @ U
        for L in Sig_ops:                      # covariance part of stage cost: tr(Q' Sig) with Sig = L Y L'
            Hm += L.T @ (Q + Kpol.T @ R @ Kpol) @ L
        new_ops = [Phi[0] @ L for L in Sig_ops]
        for i in range(1, mdl.q + 1):
            new_ops += [Phi[i] @ L for L in Sig_ops]
            new_ops.append(mdl.A[i] @ G + mdl.B[i] @ U)   # coupling: (A_i x_mean + B_i u_mean) theta_i
        Sig_ops = new_ops
        G = mdl.A[0] @ G + mdl.B[0] @ U
    Hm += G.T @ P @ G
    for L in Sig_ops:
        Hm += L.T @ P @ L
    return Hm


mdl2, K_lq, P_lq = rand_model(n=2, m=1, q=1, scale=0.2)
Q2, R2 = np.eye(2), np.eye(1)
Nh = 5
H_lq = cost_form(mdl2, K_lq, P_lq, Q2, R2, Nh)
Lam_lq = np.abs(H_lq[:2, 2:]).max()
K_bad = 0.6 * K_lq
mdl_bad = MultModel(mdl2.A, mdl2.B, mdl2.W, K_bad)
P_bad = terminal_weight(mdl_bad, Q2, R2)
H_bad = cost_form(mdl_bad, K_bad, P_bad, Q2, R2, Nh)
Lam_bad = np.abs(H_bad[:2, 2:]).max()
report("C3 IF cross term (Prop. 3)", Lam_lq < 1e-8 and Lam_bad > 1e-4,
       f"|Lambda| LQ gain = {Lam_lq:.2e}; non-optimal gain = {Lam_bad:.2e}; x0-block equals P: {np.allclose(H_lq[:2,:2], P_lq, atol=1e-8)}")

# ---------------- C4
import cxsmpc.ocp as ocp  # noqa: E402
ocp.OBJ_SCALE = 1e-3
from cxsmpc.benchmarks import bench_D  # noqa: E402
from cxsmpc.closedloop import shifted_plan  # noqa: E402
from cxsmpc.ocp import CXProblem, kappa  # noqa: E402
from cxsmpc.terminal import design_terminal  # noqa: E402
D = bench_D(); md = D.meta
cons = [dict(kind="state", h=md["h"], b=1.0, p=0.9)]
td = design_terminal(D, cons, lam=D.rho_L() + 0.5 * (1 - D.rho_L()), H0=np.outer(md["h"], md["h"]) + 1e-3 * np.eye(2))
pr = CXProblem(D, 10, md["Q"], md["R"], md["P"], cons, Ef=td["Ef"], trace_terminal=(td["S"], td["gamma"]))
worst = -np.inf; tested = 0
for x0 in ([-0.1, 0.8], [0.3, 0.9], [-0.4, 0.5], [0.5, 1.2], [1.0, 2.0]):
    r = pr.solve(np.array(x0), np.zeros((2, 2)))
    if r["class"] != "verified_feasible":
        continue
    tested += 1
    xi0 = np.r_[r["z"][:, 0], r["v"][:, 0]]
    S1 = D.L(np.zeros((2, 2))) + D.D(np.outer(xi0, xi0)) + D.W
    sp = shifted_plan(D, r["z"], r["v"])
    Sig = D.cov_recursion(S1, sp["z"], sp["v"])
    cant = max(md["h"] @ sp["z"][:, t] + kappa(0.9) * np.sqrt(md["h"] @ Sig[t] @ md["h"]) - 1.0 for t in range(11))
    term = sp["z"][:, 10] @ td["Ef"] @ sp["z"][:, 10] - 1
    tr = np.trace(td["S"] @ Sig[10]) - td["gamma"]
    worst = max(worst, cant, term, tr)
report("C4 shifted-plan feasibility (Thm 2)", tested > 0 and worst <= 1e-6, f"{tested} instances; max constraint residual {worst:.2e}")

# ---------------- C5
b, eps, kap, a, z = 3.7, 0.4, 3.0, 0.25, 0.3
lhs1 = z / b <= 1 - eps / 2 - kap**2 * (a / b**2) / (2 * eps)       # FS16 (22a) for normalised constraint (h/b)'x <= 1
lhs2 = z <= b - (b * eps) / 2 - kap**2 * a / (2 * (b * eps))         # eq. (fs-lin) with eps' = b eps
report("C5 FS16 normalisation", lhs1 == lhs2 and np.isclose(b * (1 - eps / 2 - kap**2 * (a / b**2) / (2 * eps)), b - (b * eps) / 2 - kap**2 * a / (2 * b * eps)),
       "identical constraint for eps' = b*eps (b > 0)")

# ---------------- C6
# Remark Scope (c): theta-w correlation eta_i = E[theta_i w]. Check (1) the recursion with the cross term against Monte
# Carlo, (2) the kernel form h'Sigma_t h = c_t + sum_j (xi_j' M_tj xi_j + 2 g_tj' xi_j), g_tj = sum_i G_i' Pi_{t-1-j} eta_i,
# (3) with Sigma_0 = 0 the matrix [[c', g'], [g, M]] is PSD, so the variance is ||F xi + f||^2 (SOC form).
n6, m6, q6, N6 = 2, 1, 2, 4
A6 = [np.array([[0.9, 0.2], [0.0, 0.8]])] + [0.2 * rng.normal(size=(2, 2)) for _ in range(q6)]
B6 = [np.array([[0.0], [1.0]])] + [0.2 * rng.normal(size=(2, 1)) for _ in range(q6)]
K6 = np.array([[-0.2, -0.5]])
Lj = rng.normal(size=(q6 + n6, q6 + n6)); Cj = Lj @ Lj.T          # joint covariance of (theta, w)
Cj[:q6, :q6] = np.eye(q6) + 0 * Cj[:q6, :q6]                       # E theta theta' = I
while np.min(np.linalg.eigvalsh(Cj)) <= 1e-6:
    Cj[q6:, q6:] += np.eye(n6)
eta = [Cj[i, q6:] for i in range(q6)]; W6 = Cj[q6:, q6:]
Phi6 = [A6[i] + B6[i] @ K6 for i in range(q6 + 1)]; G6 = [np.hstack([A6[i], B6[i]]) for i in range(1, q6 + 1)]
xi6 = [rng.normal(size=n6 + m6) for _ in range(N6)]
S = np.zeros((n6, n6)); Ss = [S]
for t in range(N6):
    S = sum(P_ @ S @ P_.T for P_ in Phi6) + sum(G @ np.outer(xi6[t], xi6[t]) @ G.T for G in G6) + W6 \
        + sum(np.outer(G6[i] @ xi6[t], eta[i]) + np.outer(eta[i], G6[i] @ xi6[t]) for i in range(q6))
    Ss.append(S)
# (1) Monte Carlo
M_ = 400000; E = np.zeros((M_, n6)); Z = rng.multivariate_normal(np.zeros(q6 + n6), Cj, size=(N6, M_))
for t in range(N6):
    th, w = Z[t][:, :q6], Z[t][:, q6:]
    E = E @ Phi6[0].T + sum((E @ Phi6[i + 1].T + G6[i] @ xi6[t]) * th[:, [i]] for i in range(q6)) + w
mc_err = np.abs(np.cov(E.T) - Ss[N6]).max() / np.abs(Ss[N6]).max()
# (2) kernel form, (3) PSD factorisation
h6 = rng.normal(size=n6)
Ls = lambda Y: sum(P_.T @ Y @ P_ for P_ in Phi6)
Pi = [np.outer(h6, h6)]
for _ in range(N6):
    Pi.append(Ls(Pi[-1]))
cW = sum(np.trace(Pi[N6 - 1 - j] @ W6) for j in range(N6))
Mb = np.zeros((N6 * (n6 + m6),) * 2); g = np.zeros(N6 * (n6 + m6)); val = cW
for j in range(N6):
    Mj = sum(G.T @ Pi[N6 - 1 - j] @ G for G in G6); gj = sum(G6[i].T @ Pi[N6 - 1 - j] @ eta[i] for i in range(q6))
    sl = slice(j * (n6 + m6), (j + 1) * (n6 + m6)); Mb[sl, sl] = Mj; g[sl] = gj
    val += xi6[j] @ Mj @ xi6[j] + 2 * gj @ xi6[j]
kern_err = abs(val - h6 @ Ss[N6] @ h6) / abs(h6 @ Ss[N6] @ h6)
Big = np.block([[np.array([[cW]]), g[None, :]], [g[:, None], Mb]])
psd_min = np.linalg.eigvalsh(Big).min() / np.abs(Big).max()
report("C6 theta-w correlation (Remark Scope (c))", mc_err < 2e-2 and kern_err < 1e-10 and psd_min > -1e-10,
       f"MC rel. err {mc_err:.1e}; kernel identity {kern_err:.1e}; min eig of [[c, g'], [g, M]] (rel.) {psd_min:.1e}")

print("ALL PAPER-CLAIM CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")

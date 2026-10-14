"""Offline terminal ingredients (Assumption 'Terminal ingredients', T1-T3, theory note).

Z_f = {z : z' E z <= 1} with E = P_K / c (P_K: Lyapunov matrix of A_K, so Z_f is A_K-invariant: T1),
S, lam with L*(S) <= lam S (S = sum_k (L*/lam)^k (H0)),
gamma = (omega_bar + tr(S W)) / (1 - lam) (T2 with equality), omega_bar = lambda_max(E^{-1/2} H E^{-1/2}),
H = [I;K]' D*(S) [I;K]; c is the largest value (bisection) such that T3 holds for all state/input constraints:
  sqrt(h' E^-1 h) + kappa sqrt(gamma h' S^-1 h) <= b,  sqrt(l'K E^-1 K'l) + kappa sqrt(gamma l'K S^-1 K'l) <= d.
Returns None if no c > 0 satisfies T3 (terminal ingredients do not exist with this S, lam).
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from .ocp import kap


def design_terminal(model, constraints, lam=None, H0=None, c_max=1e6, margin=0.02):
    """margin > 0 makes T2 and T3 strict (numerical robustness of recursive feasibility under finite solver accuracy)."""
    n = model.n
    rho = model.rho_L()
    lam = (1 + rho) / 2 if lam is None else lam
    H0 = np.eye(n) if H0 is None else H0
    S = np.array(H0, float)
    for _ in range(200000):
        Sn = H0 + model.Ls(S) / lam
        if np.linalg.norm(Sn - S) < 1e-12 * np.linalg.norm(Sn):
            break
        S = Sn
    S = (Sn + Sn.T) / 2
    Sinv = np.linalg.inv(S)
    AK = model.A[0] + model.B[0] @ model.K
    PK = solve_discrete_lyapunov(AK.T, np.eye(n))  # AK' PK AK - PK = -I
    IK = np.vstack([np.eye(n), model.K])
    H = IK.T @ model.Ds(S) @ IK
    trSW = float(np.trace(S @ model.W))

    def check(c):
        E = PK / c
        Ei = np.linalg.inv(E)
        Eh = np.linalg.cholesky(Ei)
        omega = float(np.max(np.linalg.eigvalsh(Eh.T @ H @ Eh))) if np.any(H) else 0.0
        gamma = (1 + margin) * (omega + trSW) / (1 - lam)  # strict T2: lam*gamma + omega + trSW < gamma
        ok = True
        for cc in constraints:
            k = kap(cc)
            hv = np.asarray(cc["h"], float) if cc["kind"] == "state" else model.K.T @ np.asarray(cc["l"], float)
            lhs = np.sqrt(hv @ Ei @ hv) + k * np.sqrt(gamma * (hv @ Sinv @ hv))
            ok &= lhs <= (1 - margin) * cc["b"]  # strict T3
        return ok, E, gamma

    lo, hi = 0.0, c_max
    if not check(1e-12)[0]:
        return None
    for _ in range(200):
        mid = np.sqrt(max(lo, 1e-12) * hi) if lo > 0 else hi / 1e6 if hi > 1e-6 else hi / 2
        if check(mid)[0]:
            lo = mid
        else:
            hi = mid
        if hi / max(lo, 1e-300) < 1 + 1e-6:
            break
    ok, E, gamma = check(lo)
    return dict(Ef=E, S=S, lam=lam, gamma=gamma, c=lo, ok=bool(ok))

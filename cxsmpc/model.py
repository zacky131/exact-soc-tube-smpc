"""Linear system with i.i.d. state/input-multiplicative and additive noise.

x+ = A0 x + B0 u + sum_i (A_i x + B_i u) theta_i + w,  E theta = 0, E theta theta' = I, cov(w) = W.
Notation follows the paper (Sections 2 and 3).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MultModel:
    A: list  # [A0, A1, ..., Aq]
    B: list  # [B0, B1, ..., Bq]
    W: np.ndarray
    K: np.ndarray
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.A = [np.atleast_2d(np.asarray(a, float)) for a in self.A]
        self.B = [np.atleast_2d(np.asarray(b, float)) for b in self.B]
        self.K = np.atleast_2d(np.asarray(self.K, float))
        self.n = self.A[0].shape[0]
        self.m = self.B[0].shape[1]
        self.q = len(self.A) - 1
        assert len(self.B) == self.q + 1
        self.Phi = [a + b @ self.K for a, b in zip(self.A, self.B)]
        self.G = [np.hstack([self.A[i], self.B[i]]) for i in range(1, self.q + 1)]

    # covariance operators and adjoints
    def L(self, X):
        return sum(P @ X @ P.T for P in self.Phi)

    def Ls(self, Y):
        return sum(P.T @ Y @ P for P in self.Phi)

    def D(self, Y):
        return sum(G @ Y @ G.T for G in self.G)

    def Ds(self, Y):
        return sum(G.T @ Y @ G for G in self.G)

    def rho_L(self):
        M = sum(np.kron(P, P) for P in self.Phi)
        return float(np.max(np.abs(np.linalg.eigvals(M))))

    def stationary_cov(self, extra=None):
        """Solve Sigma = L(Sigma) + W (+ extra)."""
        n = self.n
        M = sum(np.kron(P, P) for P in self.Phi)
        rhs = self.W if extra is None else self.W + extra
        vec = np.linalg.solve(np.eye(n * n) - M, rhs.reshape(-1, order="F"))
        S = vec.reshape(n, n, order="F")
        return (S + S.T) / 2

    def cov_recursion(self, Sigma0, z, v):
        """Exact predicted error covariance (Lemma rec). z: (n, N+1), v: (m, N)."""
        N = v.shape[1]
        Sig = [np.array(Sigma0, float)]
        for t in range(N):
            xi = np.concatenate([z[:, t], v[:, t]])
            Sig.append(self.L(Sig[t]) + self.D(np.outer(xi, xi)) + self.W)
        return Sig


def mult_lqr(A, B, Q, R, iters=20000, tol=1e-12):
    """Optimal LQR gain for multiplicative noise (theta_0 = 1, theta_i uncorrelated unit variance).

    Returns K (u = K x) and P with sum_i Phi_i' P Phi_i + Q + K'RK = P (Assumption P with equality).
    """
    P = np.array(Q, float)
    for _ in range(iters):
        H = R + sum(b.T @ P @ b for b in B)
        Gm = sum(b.T @ P @ a for a, b in zip(A, B))
        K = -np.linalg.solve(H, Gm)
        Pn = Q + sum(a.T @ P @ a for a in A) + Gm.T @ K
        Pn = (Pn + Pn.T) / 2
        if np.linalg.norm(Pn - P) < tol * max(1.0, np.linalg.norm(P)):
            P = Pn
            break
        P = Pn
    else:
        raise RuntimeError("mult_lqr did not converge (system may not be mean-square stabilisable)")
    return K, P


def terminal_weight(model, Q, R, iters=100000, tol=1e-12):
    """P = L*(P) + Q + K'RK for the given K (requires rho(L) < 1). Satisfies Assumption P with equality."""
    S = Q + model.K.T @ R @ model.K
    P = np.array(S, float)
    for _ in range(iters):
        Pn = model.Ls(P) + S
        if np.linalg.norm(Pn - P) < tol * max(1.0, np.linalg.norm(P)):
            return (Pn + Pn.T) / 2
        P = Pn
    raise RuntimeError("terminal_weight did not converge")


class CouplingAgnosticModel(MultModel):
    """Design model of a coupling-agnostic SMPC: identical to the true model except that the nominal-covariance
    coupling term D(xi xi') is dropped (Sigma+ = L(Sigma) + W). Used only for controller design, never simulation."""

    def D(self, Y):
        return np.zeros((self.n, self.n))

    def Ds(self, Y):
        return np.zeros((self.n + self.m, self.n + self.m))

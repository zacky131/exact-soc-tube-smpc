"""Offline PSD kernels (Lemma 'PSD kernel representation' and Proposition 'expected cost')."""
import numpy as np


def psd_sqrt(P):
    w, V = np.linalg.eigh((P + P.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def low_rank_factor(M, rtol=1e-12):
    """R with R'R = M (PSD), rows = numerical rank."""
    w, V = np.linalg.eigh((M + M.T) / 2)
    keep = w > rtol * max(1.0, w.max(initial=0.0))
    return (V[:, keep] * np.sqrt(w[keep])).T


class DirectionKernel:
    """Kernels for one direction h: h' Sigma_t h = <P_t, Sigma0> + wconst[t] + sum_{j<t} ||R[t-1-j] xi_j||^2."""

    def __init__(self, model, h, N):
        h = np.asarray(h, float).reshape(-1)
        P = [np.outer(h, h)]
        for _ in range(N):
            P.append(model.Ls(P[-1]))
        self.P = P
        self.Phalf = [psd_sqrt(Ps) for Ps in P]
        self.R = [low_rank_factor(model.Ds(Ps)) for Ps in P]  # index s = t-1-j
        self.wconst = [sum(float(np.sum(P[t - 1 - j] * model.W)) for j in range(t)) for t in range(N + 1)]


class MatrixKernel(DirectionKernel):
    """Kernels for <H, Sigma_t> with a PSD weight H (e.g. tr(S Sigma_t)): same structure as DirectionKernel."""

    def __init__(self, model, H, N):
        H = np.asarray(H, float)
        P = [(H + H.T) / 2]
        for _ in range(N):
            P.append(model.Ls(P[-1]))
        self.P = P
        self.Phalf = [psd_sqrt(Ps) for Ps in P]
        self.R = [low_rank_factor(model.Ds(Ps)) for Ps in P]
        self.wconst = [sum(float(np.sum(P[t - 1 - j] * model.W)) for j in range(t)) for t in range(N + 1)]


class CostKernel:
    """Expected cost J = sum_t ||z_t||_Q^2 + ||v_t||_R^2 + ||z_N||_P^2 + sum_j xi_j' H_j xi_j + <C0, Sigma0> + cW."""

    def __init__(self, model, Q, R, P, N):
        K = model.K
        S = [Q + K.T @ R @ K] * N + [P]
        self.H = []
        for j in range(N):
            Hj = np.zeros((model.n + model.m,) * 2)
            for t in range(j + 1, N + 1):
                Y = S[t]
                for _ in range(t - 1 - j):
                    Y = model.Ls(Y)
                Hj += model.Ds(Y)
            self.H.append((Hj + Hj.T) / 2)
        C0 = np.zeros((model.n, model.n))
        cW = 0.0
        for t in range(N + 1):
            Y = S[t]
            for _ in range(t):
                Y = model.Ls(Y)
            C0 += Y
            for j in range(t):
                Y2 = S[t]
                for _ in range(t - 1 - j):
                    Y2 = model.Ls(Y2)
                cW += float(np.sum(Y2 * model.W))
        self.C0 = (C0 + C0.T) / 2
        self.cW = cW
        self.Hfac = [low_rank_factor(Hj) for Hj in self.H]

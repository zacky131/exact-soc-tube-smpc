"""Closed-loop controllers for Monte-Carlo studies (E1, E13).

TubeController(mode='A')  : Theorem A — reset at k=0, prediction initialisation afterwards (deterministic nominal).
TubeController(mode='DF') : [FS16]-style binary initialisation — reset (x_k, 0) if verified feasible, else predict.
IFController              : indirect feedback ([M24]-style): nominal/covariance always predicted, cost = conditional
                            expected cost from x_k (mean predictor x_bar = z + A_K^t e_k, covariance from 0).
                            Constraints: exact realised covariance (the [M24] setting; Open problem 1).
NominalMPC                : certainty-equivalent MPC (no tightening).
All controllers apply u_k = v*_0 + K (x_k - z_0). Infeasibility in closed loop is logged, never hidden.
"""
import cvxpy as cp
import numpy as np

from .kernels import CostKernel, DirectionKernel, low_rank_factor
from . import ocp as _ocp
from .ocp import _Base, _check_dpp, kappa


class TubeController:
    def __init__(self, problem, mode):
        assert mode in ("A", "DF", "R")  # R: reset-only (ablation A3; no prediction fallback)
        self.pr, self.mode, self.m = problem, mode, problem.model

    def reset(self, x0):
        self.pred = None
        self.last = None  # last applied plan (z, v) for the shifted-candidate fallback
        self.log = dict(init_reset=0, init_predict=0, infeasible=0, unresolved=0, used_shift=0)

    def _propagate(self, res, S0):
        z, v = res["z"], res["v"]
        xi = np.r_[z[:, 0], v[:, 0]]
        return z[:, 1].copy(), self.m.L(S0) + self.m.D(np.outer(xi, xi)) + self.m.W

    def step(self, x, k):
        n = self.m.n
        res = None
        if k == 0 or self.mode in ("DF", "R"):
            r = self.pr.solve(x, np.zeros((n, n)))
            if r["class"] == "verified_feasible":
                res, S0, z0 = r, np.zeros((n, n)), x
                self.log["init_reset"] += 1
        if res is None and self.mode == "R":
            self.log["infeasible"] += 1
            return self.m.K @ x, False
        if res is None:
            if self.pred is None:
                self.log["infeasible"] += 1
                return self.m.K @ x, False
            z0, S0 = self.pred
            res = self.pr.solve(z0, S0)
            self.log["init_predict"] += 1
            if res["class"] != "verified_feasible":
                self.log["unresolved" if res["class"] == "unresolved" else "infeasible"] += 1
                if self.last is None:
                    return self.m.K @ x, False
                # Theorem rf candidate: shifted previous plan (feasible in exact arithmetic; residual = solver eps)
                res = shifted_plan(self.m, *self.last)
                self.log["used_shift"] += 1
        self.pred = self._propagate(res, S0)
        self.last = (res["z"], res["v"])
        return res["v"][:, 0] + self.m.K @ (x - res["z"][:, 0]), True


def shifted_plan(model, z, v):
    """Shifted candidate of Theorem rf: (v_1..v_{N-1}, K z_N), nominal z_1..z_N, A_K z_N."""
    K = model.K
    AK = model.A[0] + model.B[0] @ K
    zN = z[:, -1]
    return dict(z=np.column_stack([z[:, 1:], AK @ zN]), v=np.column_stack([v[:, 1:], K @ zN]))


class IFProblem(_Base):
    """Exact-covariance constraints on the predicted nominal (z0, Sigma0 = predicted), conditional-mean cost
    with data e = x_k - z0 (parameter). Optional reference r for the mean cost (non-regulation objective)."""

    def __init__(self, model, N, Q, R, P, constraints, ref=None):
        super().__init__(model, N, Q, R, P, constraints)
        n, m, K = model.n, model.m, model.K
        AK = model.A[0] + model.B[0] @ K
        self.e = cp.Parameter(n, name="e")
        self._kers, self._sroot = [], []
        cons = []
        for c in constraints:
            k = kappa(c["p"])
            if c["kind"] == "state":
                hv = np.asarray(c["h"], float); times = range(0, N + 1); mean = lambda t, h=hv: h @ self.z[t]
            else:
                l = np.asarray(c["l"], float); hv = K.T @ l; times = range(0, N); mean = lambda t, l=l: l @ self.v[:, t]
            ker = DirectionKernel(model, hv, N)
            for t in times:
                sr = cp.Parameter(nonneg=True)
                self._kers.append((ker, t)); self._sroot.append(sr)
                parts = [cp.reshape(sr, (1,), order="F")]
                for j in range(t):
                    if ker.R[t - 1 - j].shape[0] > 0:
                        parts.append(ker.R[t - 1 - j] @ self.xi[j])
                cons.append(cp.SOC((c["b"] - mean(t)) / k, cp.hstack(parts)))
        # conditional mean predictor: x_bar_t = z_t + A_K^t e, u_bar_t = v_t + K A_K^t e; covariance from 0
        Tp = [np.eye(n)]
        for _ in range(N):
            Tp.append(AK @ Tp[-1])
        r = np.zeros(n) if ref is None else np.asarray(ref, float)
        xb = [self.z[t] + Tp[t] @ self.e for t in range(N + 1)]
        ub = [self.v[:, t] + K @ Tp[t] @ self.e for t in range(N)]
        J = sum(cp.quad_form(xb[t] - r, cp.psd_wrap(Q)) for t in range(1, N))  # t=0 term is data only
        J = J + sum(cp.quad_form(ub[t], cp.psd_wrap(R)) for t in range(N)) + cp.quad_form(xb[N] - r, cp.psd_wrap(P))
        ck = CostKernel(model, Q, R, P, N)
        J = J + sum(cp.sum_squares(F @ cp.hstack([xb[j], ub[j]])) for j, F in enumerate(ck.Hfac) if F.shape[0] > 0)
        self.prob = cp.Problem(cp.Minimize(_ocp.OBJ_SCALE * J), cons + self.dyn)
        _check_dpp(self.prob)

    def _set_extra(self, S0):
        for (ker, t), sr in zip(self._kers, self._sroot):
            sr.value = float(np.sqrt(max(np.sum(ker.P[t] * S0) + ker.wconst[t], 0.0)))


class IFController:
    def __init__(self, problem):
        self.pr, self.m = problem, problem.model

    def reset(self, x0):
        n = self.m.n
        self.z0, self.S0 = np.array(x0, float), np.zeros((n, n))
        self.log = dict(infeasible=0, unresolved=0)
        self.Sigma_hist, self.z_hist = [], []

    def step(self, x, k):
        self.pr.e.value = np.asarray(x, float) - self.z0
        res = self.pr.solve(self.z0, self.S0)
        self.Sigma_hist.append(self.S0.copy()); self.z_hist.append(self.z0.copy())
        if res["class"] != "verified_feasible":
            self.log["unresolved" if res["class"] == "unresolved" else "infeasible"] += 1
            return self.m.K @ x, False
        xi = np.r_[res["z"][:, 0], res["v"][:, 0]]
        u = res["v"][:, 0] + self.m.K @ (x - self.z0)
        self.z0, self.S0 = res["z"][:, 1].copy(), self.m.L(self.S0) + self.m.D(np.outer(xi, xi)) + self.m.W
        return u, True


class NominalMPC:
    """Certainty-equivalent MPC: z0 = x_k, constraints h'z_t <= b (t >= 1), nominal quadratic cost."""

    def __init__(self, model, N, Q, R, P, constraints):
        self.m = model
        n, m = model.n, model.m
        self.x = cp.Parameter(n)
        Z, V = cp.Variable((n, N + 1)), cp.Variable((m, N))
        cons = [Z[:, 0] == self.x] + [Z[:, t + 1] == model.A[0] @ Z[:, t] + model.B[0] @ V[:, t] for t in range(N)]
        for c in constraints:
            if c["kind"] == "state":
                cons += [np.asarray(c["h"]) @ Z[:, t] <= c["b"] for t in range(1, N + 1)]
            else:
                cons += [np.asarray(c["l"]) @ V[:, t] <= c["b"] for t in range(N)]
        J = sum(cp.quad_form(Z[:, t], cp.psd_wrap(Q)) for t in range(1, N)) + sum(cp.quad_form(V[:, t], cp.psd_wrap(R)) for t in range(N))
        J = J + cp.quad_form(Z[:, N], cp.psd_wrap(P))
        self.V, self.prob = V, cp.Problem(cp.Minimize(J), cons)

    def reset(self, x0):
        self.log = dict(infeasible=0)

    def step(self, x, k):
        self.x.value = np.asarray(x, float)
        try:
            self.prob.solve(solver="CLARABEL")
        except cp.error.SolverError:
            pass
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            self.log["infeasible"] += 1
            return self.m.K @ x, False
        return self.V.value[:, 0], True

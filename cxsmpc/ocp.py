"""Finite-horizon optimal control problems.

CXProblem : exact SOC tightening + exact expected cost (Theorem 'Exact SOC', Prop. 'expected cost').
FSProblem : Farina & Scattolini (2016) with fixed gain K: LMI relaxation X_{t+1} >= L(X_t)+D(xi xi')+W
            and epsilon-linearised Cantelli constraint (their eq. 22, rescaled to general b).
Both: path constraints at t = 0..N (state) and t = 0..N-1 (input), initial data (z0, Sigma0) as parameters,
optional terminal set z_N' Ef z_N <= 1 (same for both).
"""
import time

import cvxpy as cp
import numpy as np

from .kernels import CostKernel, DirectionKernel, MatrixKernel, low_rank_factor, psd_sqrt


def _check_dpp(prob):
    if prob.is_dcp(dpp=True):
        return
    bad = [] if prob.objective.is_dcp(dpp=True) else ["objective"]
    bad += [f"constraint {i} ({type(c).__name__})" for i, c in enumerate(prob.constraints) if not c.is_dcp(dpp=True)]
    raise ValueError("problem not DPP: " + ", ".join(bad))


OBJ_SCALE = 1.0  # objective scaling for solver conditioning (same for all methods; reported cost is unscaled)
FALLBACK_MAX_ITERS = 200000  # SCS fallback budget (experiments may lower it; recorded in their config)
VERIFY_TOL = 1e-6  # absolute tolerance on the exact Cantelli constraints (b = O(1) in all benchmarks)


def kappa(p):
    return np.sqrt(p / (1.0 - p))


def kap(c):
    """Tightening factor of a constraint spec: c['kappa'] if given (ablation, e.g. Gaussian quantile), else Cantelli."""
    return c["kappa"] if "kappa" in c else kappa(c["p"])


class _Base:
    def __init__(self, model, N, Q, R, P, constraints, Ef=None):
        self.model, self.N, self.Q, self.R, self.P = model, N, Q, R, P
        self.cons_spec = constraints
        n, m = model.n, model.m
        self.z0 = cp.Parameter(n, name="z0")
        self.Sig0 = cp.Parameter((n, n), PSD=True, name="Sig0")
        self.v = cp.Variable((m, N), name="v")
        A0, B0 = model.A[0], model.B[0]
        # Sparse (non-condensed) formulation: nominal states are variables linked by equality constraints, so each
        # kernel block R xi_j touches only (z_j, v_j). The condensed form makes the KKT system nearly dense.
        self.Z = cp.Variable((n, N + 1), name="Z")
        self.dyn = [self.Z[:, 0] == self.z0] + [self.Z[:, t + 1] == A0 @ self.Z[:, t] + B0 @ self.v[:, t] for t in range(N)]
        z = [self.Z[:, t] for t in range(N + 1)]
        self.z = z
        self.xi = [cp.hstack([z[t], self.v[:, t]]) for t in range(N)]
        self.Ef = Ef
        self._obj_scale = OBJ_SCALE

    def _terminal(self):
        if self.Ef is None:
            return []
        return [cp.quad_form(self.z[self.N], cp.psd_wrap(self.Ef)) <= 1]

    def _mean_cost(self):
        N = self.N
        # ||z_0||_Q^2 depends on parameters only (not DPP); it is added back in solve()
        J = sum(cp.quad_form(self.v[:, t], cp.psd_wrap(self.R)) for t in range(N))
        J = J + sum(cp.quad_form(self.z[t], cp.psd_wrap(self.Q)) for t in range(1, N))
        return J + cp.quad_form(self.z[N], cp.psd_wrap(self.P))

    def solve(self, z0, Sigma0, solver="CLARABEL", **kw):
        self.z0.value = np.asarray(z0, float)
        S0 = (np.asarray(Sigma0, float) + np.asarray(Sigma0, float).T) / 2
        self.Sig0.value = S0
        self._set_extra(S0)
        t0 = time.perf_counter()
        status = self._try(solver, **kw)
        primary = status
        wall = time.perf_counter() - t0
        stats = self.prob.solver_stats
        solve_time = getattr(stats, "solve_time", None) if stats else None
        iters = getattr(stats, "num_iters", None) if stats else None
        used = solver
        # A numerical failure is NOT infeasibility: retry with SCS at high accuracy and record both statuses.
        if status not in ("optimal", "optimal_inaccurate", "infeasible", "unbounded"):
            used = "SCS"
            status = self._try("SCS", eps=1e-7, max_iters=FALLBACK_MAX_ITERS)
        ok = status in ("optimal", "optimal_inaccurate")
        res = dict(status=status, status_primary=primary, solver_used=used, feasible=ok, wall_time=wall,
                   solve_time=solve_time, iters=iters,
                   cost=float(self.prob.value) / self._obj_scale + float(self.z0.value @ self.Q @ self.z0.value) if ok else np.inf)
        if ok:
            res["v"] = self.v.value.copy()
            res["z"] = self.Z.value.copy()
            res["max_residual"] = self.residual(res, S0)
            res["verified"] = bool(res["max_residual"] <= VERIFY_TOL)
        else:
            res["max_residual"], res["verified"] = None, False
        # three-way classification used by all experiments
        res["class"] = ("verified_feasible" if res["verified"] else
                        "certified_infeasible" if status == "infeasible" else "unresolved")
        return res

    def residual(self, res, S0):
        """Max violation of the EXACT Cantelli constraints by the returned plan (method-independent check)."""
        m = self.model
        Sig = m.cov_recursion(S0, res["z"], res["v"])
        worst = -np.inf
        for c in self.cons_spec:
            k = kap(c)
            if c["kind"] == "state":
                h = np.asarray(c["h"], float)
                for t in range(self.N + 1):
                    worst = max(worst, h @ res["z"][:, t] + k * np.sqrt(max(h @ Sig[t] @ h, 0)) - c["b"])
            else:
                l = np.asarray(c["l"], float); hk = m.K.T @ l
                for t in range(self.N):
                    worst = max(worst, l @ res["v"][:, t] + k * np.sqrt(max(hk @ Sig[t] @ hk, 0)) - c["b"])
        return float(worst)

    def _set_extra(self, S0):
        pass

    def _try(self, solver, **kw):
        try:
            self.prob.solve(solver=solver, **kw)
            return self.prob.status
        except cp.error.SolverError as e:
            return f"solver_error: {str(e)[:60]}"


class CXProblem(_Base):
    def __init__(self, model, N, Q, R, P, constraints, Ef=None, trace_terminal=None, cost_mode="exact"):
        """trace_terminal = (S, gamma): adds tr(S Sigma_N) <= gamma (exact SOC, Def. P / Assumption T)."""
        super().__init__(model, N, Q, R, P, constraints, Ef)
        n = model.n
        # sqrt(<P_t, Sigma0> + wconst_t) is computed numerically in _set_extra and passed as ONE scalar parameter
        # per cone (instead of an n^2 block): exact, and much sparser.
        self._kers, self._sroot = [], []
        cons = []
        for c in constraints:
            k = kap(c)
            if c["kind"] == "state":
                ker = DirectionKernel(model, c["h"], N)
                times = range(0, N + 1)
                mean = lambda t, h=np.asarray(c["h"], float): h @ self.z[t]
            else:
                ker = DirectionKernel(model, model.K.T @ np.asarray(c["l"], float), N)
                times = range(0, N)
                mean = lambda t, l=np.asarray(c["l"], float): l @ self.v[:, t]
            for t in times:
                sr = cp.Parameter(nonneg=True)
                self._kers.append((ker, t)); self._sroot.append(sr)
                parts = [cp.reshape(sr, (1,), order="F")]
                for j in range(t):
                    Rj = ker.R[t - 1 - j]
                    if Rj.shape[0] > 0:
                        parts.append(Rj @ self.xi[j])
                cons.append(cp.SOC((c["b"] - mean(t)) / k, cp.hstack(parts)))
        cons += self._terminal()
        self.trace_terminal = trace_terminal
        if trace_terminal is not None:
            S_t, gam = trace_terminal
            kS = MatrixKernel(model, S_t, N)
            sr = cp.Parameter(nonneg=True)
            self._kers.append((kS, N)); self._sroot.append(sr)
            parts = [cp.reshape(sr, (1,), order="F")] + [kS.R[N - 1 - j] @ self.xi[j] for j in range(N) if kS.R[N - 1 - j].shape[0] > 0]
            cons.append(cp.SOC(np.sqrt(gam), cp.hstack(parts)))
        self.ck = CostKernel(model, Q, R, P, N)
        self.cost_mode = cost_mode
        if cost_mode == "exact":
            Jcov = sum(cp.sum_squares(F @ self.xi[j]) for j, F in enumerate(self.ck.Hfac) if F.shape[0] > 0)
            obj = self._mean_cost() + Jcov + cp.trace(self.ck.C0 @ self.Sig0) + self.ck.cW
        else:  # ablation A2: nominal cost only (covariance-inflation not penalised)
            obj = self._mean_cost()
        self.prob = cp.Problem(cp.Minimize(OBJ_SCALE * obj), cons + self.dyn)
        _check_dpp(self.prob)

    def residual(self, res, S0):
        """Exact Cantelli residual plus terminal-set and trace-terminal residuals."""
        r = super().residual(res, S0)
        if self.Ef is not None:
            zN = res["z"][:, self.N]; r = max(r, float(zN @ self.Ef @ zN - 1))
        if self.trace_terminal is not None:
            S_t, gam = self.trace_terminal
            Sig = self.model.cov_recursion(S0, res["z"], res["v"])
            r = max(r, float(np.trace(S_t @ Sig[self.N]) - gam))
        return r

    def _set_extra(self, S0):
        for (ker, t), sr in zip(self._kers, self._sroot):
            sr.value = float(np.sqrt(max(np.sum(ker.P[t] * S0) + ker.wconst[t], 0.0)))


class FSProblem(_Base):
    def __init__(self, model, N, Q, R, P, constraints, eps=0.5, Ef=None, t0_exact=False):
        """t0_exact=True (revision-v7, reviewer R1-W1): impose the exact deterministic constraint at t = 0 instead of
        the linearised (22a) form. Valid only for solves with Sigma0 = 0 (reset initialisation, zero variance at t = 0);
        use FSResetExact to dispatch. Default False = the literal formulation of [FS16]."""
        super().__init__(model, N, Q, R, P, constraints, Ef)
        n, q = model.n, model.q
        self.eps = eps
        self.t0_exact = t0_exact
        self.X = [self.Sig0] + [cp.Variable((n, n), symmetric=True, name=f"X{t}") for t in range(1, N + 1)]
        cons = []
        for t in range(N):
            Gxi = cp.hstack([cp.reshape(G @ self.xi[t], (n, 1), order="F") for G in model.G])
            Lx = sum(Ph @ self.X[t] @ Ph.T for Ph in model.Phi)
            top = self.X[t + 1] - Lx - model.W
            top = (top + top.T) / 2
            blk = cp.bmat([[top, Gxi], [Gxi.T, np.eye(q)]])
            cons.append(blk >> 0)
        for c in constraints:
            k = kap(c)
            if c["kind"] == "state":
                h = np.asarray(c["h"], float)
                for t in range(0, N + 1):
                    if t == 0 and t0_exact:
                        cons.append(h @ self.z[0] <= c["b"])  # zero variance at t = 0: exact constraint
                        continue
                    cons.append(h @ self.z[t] <= c["b"] - eps / 2 - k**2 / (2 * eps) * (h @ self.X[t] @ h))
            else:
                l = np.asarray(c["l"], float)
                hk = model.K.T @ l
                for t in range(0, N):
                    if t == 0 and t0_exact:
                        cons.append(l @ self.v[:, 0] <= c["b"])  # zero variance at t = 0: exact constraint
                        continue
                    cons.append(l @ self.v[:, t] <= c["b"] - eps / 2 - k**2 / (2 * eps) * (hk @ self.X[t] @ hk))
        cons += self._terminal()
        K = model.K
        Sq = Q + K.T @ R @ K
        Jcov = sum(cp.trace(Sq @ self.X[t]) for t in range(N)) + cp.trace(P @ self.X[N])
        self.prob = cp.Problem(cp.Minimize(OBJ_SCALE * (self._mean_cost() + Jcov)), cons + self.dyn)

    def residual(self, res, S0):
        """Exact-Cantelli residual (base) AND FS's own LMI/linear-constraint residuals: the FS solution is only
        accepted if it is a genuine solution of the FS problem (inaccurate solver output is rejected)."""
        exact = super().residual(res, S0)
        m = self.model
        X = [S0] + [Xt.value for Xt in self.X[1:]]
        lmi = 0.0
        for t in range(self.N):
            xi = np.r_[res["z"][:, t], res["v"][:, t]]
            G = X[t + 1] - m.L(X[t]) - m.W - m.D(np.outer(xi, xi))
            lmi = max(lmi, -np.linalg.eigvalsh((G + G.T) / 2).min())
        own = max([c.violation().max() if np.size(c.violation()) else 0.0 for c in self.prob.constraints
                   if not isinstance(c, cp.constraints.PSD)] + [0.0])
        return float(max(exact, lmi, own))
        _check_dpp(self.prob)


class FSResetExact:
    """LMI formulation [FS16] with the exact constraint at t = 0 whenever the initial covariance is zero (reset
    initialisation), and the literal linearised constraint otherwise (predicted initialisation, Sigma0 != 0).
    Sensitivity comparator for reviewer R1-W1 (revision-v7); same gain, weights, horizon, eps and checks."""

    def __init__(self, model, N, Q, R, P, constraints, eps=0.5, Ef=None):
        self.model, self.N, self.eps = model, N, eps
        self.exact = FSProblem(model, N, Q, R, P, constraints, eps=eps, Ef=Ef, t0_exact=True)
        self.literal = FSProblem(model, N, Q, R, P, constraints, eps=eps, Ef=Ef, t0_exact=False)

    def solve(self, z0, Sigma0, **kw):
        S0 = np.asarray(Sigma0, float)
        pr = self.exact if np.all(S0 == 0.0) else self.literal
        return pr.solve(z0, S0, **kw)


def _weight(model, H0, lam):
    """S = sum_k (L*/lam)^k (H0): satisfies L*(S) = lam (S - H0) <= lam S."""
    S = np.array(H0, float)
    for _ in range(200000):
        Sn = H0 + model.Ls(S) / lam
        if np.linalg.norm(Sn - S) < 1e-12 * np.linalg.norm(Sn):
            return (Sn + Sn.T) / 2
        S = Sn
    raise RuntimeError("weight iteration did not converge")


class HybridProblem(_Base):
    """Exact head + trace-chain tail (theory: Sec. 'Exact head with trace-chain tail', Def. P_H, Prop. hybrid).

    (H1) exact SOC Cantelli constraints for t = 0..Ne;
    per constraint direction h (chain='per_direction', default) or one shared chain (chain='shared'):
    (H2) sigma_Ne >= sqrt(tr(S Sigma_Ne)); (H3) sigma_{t+1} >= ||(sqrt(lam) sigma_t, F xi_t, sqrt(tr(S W)))||;
    (H4) tail h'z_t + kappa sqrt(h'S^-1 h) sigma_t <= b, with L*(S) <= lam S.
    Per-direction weight: S_h = sum_k (L*/lam)^k (h h' + delta I) (tight for direction h).
    """

    def __init__(self, model, N, Q, R, P, constraints, Ne, lam=None, chain="per_direction", delta=1e-3, Ef=None):
        super().__init__(model, N, Q, R, P, constraints, Ef)
        rho = model.rho_L()
        assert rho < 1, "hybrid requires rho(L) < 1"
        self.Ne = Ne = int(min(Ne, N))
        self.lam = lam = (1 + rho) / 2 if lam is None else lam
        assert rho < lam < 1
        n = model.n
        self._kers, self._sroot = [], []
        cons = []
        specs = []
        for c in constraints:
            k = kap(c)
            if c["kind"] == "state":
                hv = np.asarray(c["h"], float); head = range(0, Ne + 1); tail = range(Ne + 1, N + 1)
                mean = lambda t, h=hv: h @ self.z[t]
            else:
                l = np.asarray(c["l"], float); hv = model.K.T @ l
                head = range(0, min(Ne, N - 1) + 1); tail = range(Ne + 1, N)
                mean = lambda t, l=l: l @ self.v[:, t]
            ker = DirectionKernel(model, hv, Ne)
            for t in head:
                cons.append(self._exact_cone(ker, t, (c["b"] - mean(t)) / k))
            specs.append((hv, k, c["b"], tail, mean))
        if chain == "shared":
            S = _weight(model, np.eye(n), lam)
            groups = [(S, specs)]
        else:
            groups = [(_weight(model, np.outer(hv, hv) + delta * (hv @ hv) * np.eye(n), lam), [sp]) for sp in specs]
        self.chains = []
        for S, members in groups:
            if not any(len(sp[3]) for sp in members):
                continue  # no tail for these directions
            Sinv = np.linalg.inv(S)
            F = low_rank_factor(model.Ds(S))
            omegaW = float(np.trace(S @ model.W))
            sig = cp.Variable(N - Ne + 1, nonneg=True)
            self.chains.append(sig)
            cons.append(self._exact_cone(MatrixKernel(model, S, Ne), Ne, sig[0]))
            for t in range(Ne, N):
                parts = [cp.reshape(np.sqrt(lam) * sig[t - Ne], (1,), order="F")]
                if F.shape[0] > 0:
                    parts.append(F @ self.xi[t])
                if omegaW > 0:
                    parts.append(np.array([np.sqrt(omegaW)]))
                cons.append(cp.SOC(sig[t + 1 - Ne], cp.hstack(parts)))
            for hv, k, b, tail, mean in members:
                coef = k * np.sqrt(max(hv @ Sinv @ hv, 0.0))
                for t in tail:
                    cons.append(mean(t) + coef * sig[t - Ne] <= b)
        cons += self._terminal()
        self.ck = CostKernel(model, Q, R, P, N)
        Jcov = sum(cp.sum_squares(Fc @ self.xi[j]) for j, Fc in enumerate(self.ck.Hfac) if Fc.shape[0] > 0)
        obj = self._mean_cost() + Jcov + cp.trace(self.ck.C0 @ self.Sig0) + self.ck.cW
        self.prob = cp.Problem(cp.Minimize(OBJ_SCALE * obj), cons + self.dyn)
        _check_dpp(self.prob)

    def _exact_cone(self, ker, t, rhs):
        sr = cp.Parameter(nonneg=True)
        self._kers.append((ker, t)); self._sroot.append(sr)
        parts = [cp.reshape(sr, (1,), order="F")]
        for j in range(t):
            Rj = ker.R[t - 1 - j]
            if Rj.shape[0] > 0:
                parts.append(Rj @ self.xi[j])
        return cp.SOC(rhs, cp.hstack(parts))

    def _set_extra(self, S0):
        for (ker, t), sr in zip(self._kers, self._sroot):
            sr.value = float(np.sqrt(max(np.sum(ker.P[t] * S0) + ker.wconst[t], 0.0)))


class DualModeCXProblem(_Base):
    """Exact CX-SMPC with dual-mode prediction: free inputs v_0..v_{Nc-1}, then v_t = K z_t (t >= Nc).

    Exactness is kept: for t > Nc the tail contributions sum_{j=Nc}^{t-1} xi_j' M_{t,j} xi_j are collapsed into one
    block z_Nc' Mtilde_t z_Nc using xi_j = [I; K] A_K^{j-Nc} z_Nc, so every cone has at most Nc + 1 blocks
    (O(N * Nc) instead of O(N^2) nonzeros per direction). Cost: exact expected cost, collapsed the same way.
    """

    def __init__(self, model, N, Q, R, P, constraints, Nc, Ef=None, trace_terminal=None):
        """trace_terminal = (S, gamma): tr(S Sigma_N) <= gamma as one SOC, tail j >= Nc collapsed on z_Nc (Prop. 4)."""
        super().__init__(model, N, Q, R, P, constraints, Ef)
        self.Nc = Nc = int(min(Nc, N))
        n, m, K = model.n, model.m, model.K
        AK = model.A[0] + model.B[0] @ K
        IK = np.vstack([np.eye(n), K])
        Tpow = [np.eye(n)]
        for _ in range(N):
            Tpow.append(AK @ Tpow[-1])
        cons = [self.v[:, t] == K @ self.z[t] for t in range(Nc, N)]
        self._kers, self._sroot = [], []
        zc = self.z[Nc]

        def tail_factor(Ms, t):  # sum_{j=Nc}^{t-1} (IK A_K^{j-Nc})' M_{t,j} (IK A_K^{j-Nc})
            Mt = np.zeros((n, n))
            for j in range(Nc, t):
                T = IK @ Tpow[j - Nc]
                Mt += T.T @ Ms(t - 1 - j) @ T
            return low_rank_factor(Mt)

        for c in constraints:
            k = kap(c)
            if c["kind"] == "state":
                hv = np.asarray(c["h"], float); times = range(0, N + 1)
                mean = lambda t, h=hv: h @ self.z[t]
            else:
                l = np.asarray(c["l"], float); hv = K.T @ l; times = range(0, N)
                mean = lambda t, l=l: l @ self.v[:, t]
            ker = DirectionKernel(model, hv, N)
            Ms = lambda s, ker=ker: ker.R[s].T @ ker.R[s]
            for t in times:
                sr = cp.Parameter(nonneg=True)
                self._kers.append((ker, t)); self._sroot.append(sr)
                parts = [cp.reshape(sr, (1,), order="F")]
                for j in range(min(t, Nc)):
                    Rj = ker.R[t - 1 - j]
                    if Rj.shape[0] > 0:
                        parts.append(Rj @ self.xi[j])
                if t > Nc:
                    Ft = tail_factor(Ms, t)
                    if Ft.shape[0] > 0:
                        parts.append(Ft @ zc)
                cons.append(cp.SOC((c["b"] - mean(t)) / k, cp.hstack(parts)))
        cons += self._terminal()
        self.trace_terminal = trace_terminal
        if trace_terminal is not None:
            S_t, gam = trace_terminal
            kS = MatrixKernel(model, S_t, N)
            sr = cp.Parameter(nonneg=True)
            self._kers.append((kS, N)); self._sroot.append(sr)
            parts = [cp.reshape(sr, (1,), order="F")]
            parts += [kS.R[N - 1 - j] @ self.xi[j] for j in range(min(N, Nc)) if kS.R[N - 1 - j].shape[0] > 0]
            if N > Nc:
                Ft = tail_factor(lambda s_, kS=kS: kS.R[s_].T @ kS.R[s_], N)
                if Ft.shape[0] > 0:
                    parts.append(Ft @ zc)
            cons.append(cp.SOC(np.sqrt(gam), cp.hstack(parts)))
        self.ck = CostKernel(model, Q, R, P, N)
        # covariance cost: free-move blocks + one collapsed block on z_Nc for j >= Nc
        Jcov = sum(cp.sum_squares(Fc @ self.xi[j]) for j, Fc in enumerate(self.ck.Hfac[:Nc]) if Fc.shape[0] > 0)
        Mt = np.zeros((n, n))
        for j in range(Nc, N):
            T = IK @ Tpow[j - Nc]
            Mt += T.T @ self.ck.H[j] @ T
        Ft = low_rank_factor(Mt)
        if Ft.shape[0] > 0:
            Jcov = Jcov + cp.sum_squares(Ft @ zc)
        obj = self._mean_cost() + Jcov + cp.trace(self.ck.C0 @ self.Sig0) + self.ck.cW
        self.prob = cp.Problem(cp.Minimize(OBJ_SCALE * obj), cons + self.dyn)
        _check_dpp(self.prob)

    def residual(self, res, S0):
        """Exact Cantelli residual plus terminal-set and trace-terminal residuals (as CXProblem)."""
        r = super().residual(res, S0)
        if self.Ef is not None:
            zN = res["z"][:, self.N]; r = max(r, float(zN @ self.Ef @ zN - 1))
        if self.trace_terminal is not None:
            S_t, gam = self.trace_terminal
            Sig = self.model.cov_recursion(S0, res["z"], res["v"])
            r = max(r, float(np.trace(S_t @ Sig[self.N]) - gam))
        return r

    def _set_extra(self, S0):
        for (ker, t), sr in zip(self._kers, self._sroot):
            sr.value = float(np.sqrt(max(np.sum(ker.P[t] * S0) + ker.wconst[t], 0.0)))

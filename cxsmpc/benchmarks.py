"""Benchmark systems.

A : Moussa & Fiacchini (2026) [AP002], eq. (15); parametric noise whitened to the (A_i) form (second moments only).
D : Primbs & Sung (2009) example as used in Farina & Scattolini (2016) and Mark et al. (2024).
C : random systems for scaling studies.
All parameter choices not given in the source papers are marked SOURCE: ours.
"""
import numpy as np

from .model import MultModel, mult_lqr


def bench_A(K=None, Sp_scale=1.0):
    A0 = np.array([[0.4, 0.4, 0.5], [0.1, 0.1, 0.2], [0.2, 0.1, 0.5]])
    B0 = np.array([[1.0], [0.0], [0.0]])
    Sp = Sp_scale * np.array([[0.21, 0.03], [0.03, 0.15]])  # Sp_scale = 1: AP002 eq.(15)
    m22 = Sp[1, 1]  # E[p2^2]
    # r = (p1, p2, p2^2 - 0.15): zero mean, cov = blkdiag(Sp, var(p2^2)) with var(p2^2) = 2*0.15^2 (Gaussian p2)
    Cr = np.zeros((3, 3))
    Cr[:2, :2] = Sp
    Cr[2, 2] = 2 * m22**2
    E1 = np.zeros((3, 3)); E1[0, 0] = 1; E1[0, 2] = 1  # p1 at (1,1),(1,3)
    E2 = np.zeros((3, 3)); E2[2, 2] = 1                # p2 at (3,3)
    E3 = np.zeros((3, 3)); E3[1, 1] = 1                # p2^2 - 0.15 at (2,2)
    U = np.linalg.cholesky(Cr)
    Ai = [sum(U[j, l] * Ej for j, Ej in enumerate([E1, E2, E3])) for l in range(3)]
    if K is None:
        K = np.array([[-0.4, -0.4, -0.5]])  # AP002 Thm 1 gain (alpha = 0.78)
    mdl = MultModel([A0] + Ai, [B0] + [np.zeros((3, 1))] * 3, W=np.eye(3), K=K, name="A_AP002")
    mdl.meta.update(Sp=Sp, Sp_scale=Sp_scale, source="AP002 eq.(15) (Sp scaled by Sp_scale); K from AP002 Sec.5")

    def sample_true(x, u, rng):
        """Exact AP002 noise model (not the whitened surrogate). x: (M, 3), u: (M, 1)."""
        M = x.shape[0]
        p = rng.multivariate_normal(np.zeros(2), Sp, size=M)
        p1, p2 = p[:, 0], p[:, 1]
        Abar = np.zeros((M, 3, 3))
        Abar[:, 0, 0] = p1; Abar[:, 0, 2] = p1; Abar[:, 1, 1] = p2**2 - m22; Abar[:, 2, 2] = p2
        w = rng.standard_normal((M, 3))
        return x @ A0.T + u @ B0.T + np.einsum("mij,mj->mi", Abar, x) + w

    mdl.sample_true = sample_true
    return mdl


def bench_D(K=None, Q=None, R=None):
    A0 = np.array([[1.02, -0.1], [0.1, 0.98]])
    B0 = np.array([[0.1, 0.0], [0.05, 0.01]])
    A1 = np.diag([0.04, 0.04])
    B1 = np.array([[0.04, 0.0], [-0.04, 0.008]])
    Q = np.diag([2.0, 1.0]) if Q is None else Q
    R = np.diag([5.0, 20.0]) if R is None else R
    Kl, P = mult_lqr([A0, A1], [B0, B1], Q, R)
    K = Kl if K is None else K
    mdl = MultModel([A0, A1], [B0, B1], W=np.zeros((2, 2)), K=K, name="D_PrimbsSung")
    mdl.meta.update(Q=Q, R=R, P=P, h=np.array([-2.0, 1.0]) / 2.3, b=1.0, p=0.9,
                    source="Primbs&Sung 2009 / FS16 / M24; K = multiplicative-noise LQR (SOURCE: ours, M24 does not report K)")
    return mdl


def theta_sampler(kind):
    def f(rng, size):
        if kind == "gauss":
            return rng.standard_normal(size)
        if kind == "laplace":
            return rng.laplace(0.0, 1 / np.sqrt(2), size)
        if kind == "rademacher":
            return rng.choice([-1.0, 1.0], size=size)
        if kind.startswith("t"):
            nu = float(kind[1:])
            return rng.standard_t(nu, size=size) / np.sqrt(nu / (nu - 2))
        raise ValueError(kind)
    return f


def bench_C(n, m, q, rng, rho0=1.05, noise=0.08):
    """Random system (SOURCE: ours). Open-loop spectral radius rho0, noise matrices scaled by `noise`."""
    for _ in range(200):
        A0 = rng.normal(size=(n, n))
        A0 *= rho0 / max(abs(np.linalg.eigvals(A0)))
        B0 = rng.normal(size=(n, m))
        Ai = [noise * rng.normal(size=(n, n)) for _ in range(q)]
        Bi = [noise * rng.normal(size=(n, m)) for _ in range(q)]
        try:
            K, P = mult_lqr([A0] + Ai, [B0] + Bi, np.eye(n), np.eye(m))
        except RuntimeError:
            continue
        mdl = MultModel([A0] + Ai, [B0] + Bi, W=0.01 * np.eye(n), K=K, name=f"C_n{n}m{m}q{q}")
        if mdl.rho_L() < 0.98:
            mdl.meta.update(Q=np.eye(n), R=np.eye(m), P=P)
            return mdl
    raise RuntimeError("could not generate MSS random system")


def bench_Q(Ts=0.05, cd=0.01, sig_eta=0.2, sig_xi=0.3, sig_a=0.2, Qw=None, Rw=None):
    """Quadrotor altitude channel (ILLUSTRATIVE). x = [z - z_ref, z_dot], u = thrust deviation from hover [N].

    VERIFIED (NanoBench, arXiv:2603.09908): Crazyflie bare mass 27 g; combined peak thrust ~0.6 N.
    ASSUMED (SOURCE: ours, not verified): linear drag cd [N s/m]; thrust-effectiveness noise sig_eta (multiplies the
    commanded thrust deviation; hover thrust assumed compensated); drag-coefficient noise sig_xi (multiplies the drag
    term); additive acceleration noise sig_a [m/s^2]. Euler discretisation with sample time Ts.
    """
    m, g = 0.027, 9.81
    A0 = np.array([[1.0, Ts], [0.0, 1.0 - Ts * cd / m]])
    B0 = np.array([[0.5 * Ts**2 / m], [Ts / m]])
    A1 = sig_xi * np.array([[0.0, 0.0], [0.0, -Ts * cd / m]]); B1 = np.zeros((2, 1))          # drag noise
    A2 = np.zeros((2, 2)); B2 = sig_eta * np.array([[0.5 * Ts**2 / m], [Ts / m]])               # thrust-effectiveness noise
    W = (sig_a * Ts) ** 2 * np.array([[0.25 * Ts**2, 0.5 * Ts], [0.5 * Ts, 1.0]])               # additive accel noise
    Qw = np.diag([10.0, 1.0]) if Qw is None else Qw
    Rw = np.array([[100.0]]) if Rw is None else Rw
    K, P = mult_lqr([A0, A1, A2], [B0, B1, B2], Qw, Rw)
    mdl = MultModel([A0, A1, A2], [B0, B1, B2], W=W, K=K, name="Q_quadrotor_altitude")
    mdl.meta.update(Q=Qw, R=Rw, P=P, m=m, g=g, Ts=Ts, u_max=0.6 - m * g, u_min=-m * g,
                    source="mass/thrust: NanoBench arXiv:2603.09908; cd, noise levels: ASSUMED")
    return mdl

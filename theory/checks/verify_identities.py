"""Deterministic algebra checks for theory/cx_smpc_theory.tex.

Not an experiment: no closed-loop simulation, no performance metric.
Verifies, on random matrices, the identities/inequalities used in the proofs.
"""
import numpy as np

rng = np.random.default_rng(0)
n, m, q, N = 4, 2, 3, 8

A = [rng.normal(size=(n, n)) * (0.4 if i == 0 else 0.1) for i in range(q + 1)]
B = [rng.normal(size=(n, m)) * (0.4 if i == 0 else 0.1) for i in range(q + 1)]
K = rng.normal(size=(m, n)) * 0.3
W = (lambda M: M @ M.T)(rng.normal(size=(n, n))) * 0.1
Phi = [A[i] + B[i] @ K for i in range(q + 1)]
G = [np.hstack([A[i], B[i]]) for i in range(1, q + 1)]


def L(X):
    return sum(P @ X @ P.T for P in Phi)


def Ls(Y):
    return sum(P.T @ Y @ P for P in Phi)


def D(Y):
    return sum(Gi @ Y @ Gi.T for Gi in G)


def Ds(Y):
    return sum(Gi.T @ Y @ Gi for Gi in G)


def psd_sqrt(P):
    w, V = np.linalg.eigh((P + P.T) / 2)
    return V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T


# --- random plan and initial covariance
xi = [rng.normal(size=n + m) for _ in range(N)]
Sigma0 = (lambda M: M @ M.T)(rng.normal(size=(n, n))) * 0.05

# Lemma (recursion) forward
Sig = [Sigma0]
for t in range(N):
    Sig.append(L(Sig[t]) + D(np.outer(xi[t], xi[t])) + W)

# Lemma (PSD kernel) for random h, all t
worst = 0.0
min_eig = np.inf
for _ in range(20):
    h = rng.normal(size=n)
    for t in range(1, N + 1):
        P = [np.outer(h, h)]
        for s in range(t):
            P.append(Ls(P[-1]))
        c = np.sum(P[t] * Sigma0) + sum(np.sum(P[t - 1 - j] * W) for j in range(t))
        quad = 0.0
        for j in range(t):
            M = Ds(P[t - 1 - j])
            min_eig = min(min_eig, np.linalg.eigvalsh(M).min())
            R = np.vstack([psd_sqrt(P[t - 1 - j]) @ Gi for Gi in G])
            assert np.allclose(R.T @ R, M, atol=1e-9)
            quad += xi[j] @ M @ xi[j]
        lhs = h @ Sig[t] @ h
        worst = max(worst, abs(lhs - (c + quad)) / max(1.0, abs(lhs)))
print(f"[kernel identity] max rel. error = {worst:.2e}; min eig of kernels = {min_eig:.2e}")
assert worst < 1e-9 and min_eig > -1e-10

# Proposition (containment): X_t >= Sigma_t and AM-GM chain, pointwise
kappa = np.sqrt(0.9 / 0.1)
X = [Sigma0]
for t in range(N):
    slack = (lambda M: M @ M.T)(rng.normal(size=(n, n))) * 0.01  # any PSD slack
    X.append(L(X[t]) + D(np.outer(xi[t], xi[t])) + W + slack)
ok = True
for t in range(N + 1):
    ok &= np.linalg.eigvalsh(X[t] - Sig[t]).min() > -1e-9
    for eps in [0.05, 0.1, 0.3, 0.5, 0.9, 2.0]:
        for _ in range(10):
            h = rng.normal(size=n)
            a_true, a_X = h @ Sig[t] @ h, h @ X[t] @ h
            ok &= kappa * np.sqrt(a_true) <= eps / 2 + kappa**2 * a_X / (2 * eps) + 1e-12
print(f"[containment chain] X_t >= Sigma_t and AM-GM bound hold: {ok}")
assert ok

# Lemma (directional bound) h'Σh <= (h'S^{-1}h) tr(SΣ)
ok = True
for _ in range(200):
    S = (lambda M: M @ M.T + 0.1 * np.eye(n))(rng.normal(size=(n, n)))
    Sg = (lambda M: M @ M.T)(rng.normal(size=(n, n)))
    h = rng.normal(size=n)
    ok &= h @ Sg @ h <= (h @ np.linalg.solve(S, h)) * np.trace(S @ Sg) + 1e-9
print(f"[directional bound] holds: {ok}")
assert ok

# Young bound used in the robust-coupling theorem
ok = True
for _ in range(200):
    a, b = rng.normal(size=n), rng.normal(size=n)
    alpha = rng.uniform(0.05, 5)
    gap = (1 + alpha) * np.outer(a, a) + (1 + 1 / alpha) * np.outer(b, b) - np.outer(a + b, a + b)
    ok &= np.linalg.eigvalsh(gap).min() > -1e-9
print(f"[Young matrix bound] holds: {ok}")
assert ok

# Trace-terminal step: tr(S L(Σ)) = <L*(S), Σ>
S = (lambda M: M @ M.T + np.eye(n))(rng.normal(size=(n, n)))
Sg = Sig[N]
print(f"[adjoint identity] |tr(S L(Σ)) - <L*(S),Σ>| = {abs(np.trace(S @ L(Sg)) - np.sum(Ls(S) * Sg)):.2e}")

# Terminal-decrease lemma: E||x+||_P^2 = x' L*(P) x + tr(PW) and <= ... under L*(P)+Q+K'RK <= P
Qc = np.eye(n); Rc = np.eye(m) * 0.5
Phi_s = [P_ * 0.25 / max(1, np.abs(np.linalg.eigvals(P_)).max()) for P_ in Phi]  # make MSS
def Ls2(Y): return sum(P_.T @ Y @ P_ for P_ in Phi_s)
Pm = np.zeros((n, n))
for _ in range(500):  # fixed point of P = L*(P) + Q + K'RK
    Pm = Ls2(Pm) + Qc + K.T @ Rc @ K
x = rng.normal(size=n)
# exact expectation via Rademacher enumeration of theta (q bits) with w ~ zero (W handled separately)
import itertools
Ex = 0.0
for th in itertools.product([-1.0, 1.0], repeat=q):
    xp = Phi_s[0] @ x + sum(Phi_s[i + 1] @ x * th[i] for i in range(q))
    Ex += xp @ Pm @ xp / 2**q
lhs = Ex; rhs = x @ Ls2(Pm) @ x
print(f"[terminal decrease] |E||x+||_P^2 - x'L*(P)x| = {abs(lhs-rhs):.2e}; decrease ok: {lhs <= x@Pm@x - x@Qc@x - (K@x)@Rc@(K@x) + 1e-9}")
assert abs(lhs - rhs) < 1e-9

# Scalar M24 example: exact second moments by Rademacher enumeration (3 steps) vs closed form
a_, b_, c_, M_, L_, x0 = 0.9, 1.0, 0.3, 0.2, 0.5, 1.7
p3 = 0.0; sig3 = 0.0
for th in itertools.product([-1.0, 1.0], repeat=3):
    e, z, Sg = 0.0, x0, 0.0
    for k in range(3):
        v = -M_ * z - L_ * e
        e, z, Sg = a_ * e + c_ * (e + z) * th[k], (a_ - b_ * M_) * z - b_ * L_ * e, (a_**2 + c_**2) * Sg + c_**2 * z**2
    p3 += e**2 / 8; sig3 += Sg / 8
pred = -2 * a_ * b_ * L_ * c_**4 * x0**2
print(f"[M24 scalar bias] p3 - sigma3 = {p3 - sig3:.6e}; closed form = {pred:.6e}")
assert abs((p3 - sig3) - pred) < 1e-12

# Lemma (uniform covariance bound): xi_j in Xi, Sigma0 <= Sbar  =>  Sigma_t <= Sbar
Xib = (lambda M_: M_ @ M_.T + np.eye(n + m))(rng.normal(size=(n + m, n + m)))
Phi_bak = Phi[:]
Phi[:] = Phi_s  # use the MSS operator set
M_ = sum(np.kron(P_, P_) for P_ in Phi)
Sbar = np.linalg.solve(np.eye(n * n) - M_, (W + D(Xib)).reshape(-1, order="F")).reshape(n, n, order="F")
ok = True
for _ in range(50):
    S = 0.5 * Sbar  # Sigma0 <= Sbar
    for t in range(30):
        y = rng.normal(size=n + m); y /= np.sqrt(y @ np.linalg.solve(Xib, y)); y *= rng.uniform(0, 1)  # xi in Xi
        S = L(S) + D(np.outer(y, y)) + W
        ok &= np.linalg.eigvalsh(Sbar - S).min() > -1e-9
Phi[:] = Phi_bak
print(f"[uniform covariance bound] holds: {ok}")
assert ok
print("ALL CHECKS PASSED")

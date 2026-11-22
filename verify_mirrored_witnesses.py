#!/usr/bin/env python3
"""
verify_mirrored_witnesses.py

Deterministic verification script accompanying

  "Exact separable bounds for covariant entanglement witnesses and the mirrored witnesses in 4 (x) 4"

which audits J. Seong, A. Bera, B. C. Hiesmayr, D. Chruscinski and J. Bae,
"Mirrored entanglement witnesses for multipartite and high-dimensional quantum systems",
Quantum Sci. Technol. 11, 015037 (2026) [arXiv:2510.06863v2].  "SBHCB (n)" = equation (n) of arXiv v2.

Conventions (SBHCB (68), (70), (72)):
  W[alpha] = sum_{i,k} alpha_k |i,i+k><i,i+k|  -  sum_{i != j} |ii><jj|      on C^n (x) C^n, indices mod n,
  alpha = (a,b,c) for n = 3 and (a,b,c,d) for n = 4;  basis index of |i,j> is i*n + j.
  Mirrored operator with constant mu:  M = mu*1 - W.
  u_W  = max over product states of <e f|W|e f>   (the separable upper bound of SBHCB (11)).

Requires numpy and sympy.  Run:  python verify_mirrored_witnesses.py
Exit status 0 iff every check passes.
"""
import itertools
import math
import sys
import time
from fractions import Fraction as Fr

import numpy as np
import sympy as sp

TOL = 1e-9
RESULTS = []
T0 = time.time()


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def header(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


# ----------------------------------------------------------------------------------------------
# operators
# ----------------------------------------------------------------------------------------------
def W_op(alpha, exact=False):
    n = len(alpha)
    zero = Fr(0) if exact else 0.0
    W = [[zero] * (n * n) for _ in range(n * n)]
    for i in range(n):
        for k in range(n):
            W[i * n + (i + k) % n][i * n + (i + k) % n] += alpha[k]
        for j in range(n):
            if i != j:
                W[i * n + i][j * n + j] -= 1
    return W if exact else np.array(W, dtype=float)


def PT_A(rho, dA, dB):
    return rho.reshape(dA, dB, dA, dB).transpose(2, 1, 0, 3).reshape(dA * dB, dA * dB)


def product_extremum(W, dims, sign=+1, batch=200, iters=200, seed=0):
    """max (sign=+1) / min (sign=-1) of <psi|W|psi> over product vectors, batched alternating eigen-updates."""
    rng = np.random.default_rng(seed)
    k = len(dims)
    T = (sign * W).reshape(tuple(dims) + tuple(dims))
    vecs = []
    for d in dims:
        v = rng.normal(size=(batch, d)) + 1j * rng.normal(size=(batch, d))
        vecs.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    letters = "abcdefgh"
    val = None
    for _ in range(iters):
        for t in range(k):
            ins, ops = [], []
            for s in range(k):
                if s != t:
                    ins.append(f"z{letters[s]}"); ops.append(vecs[s].conj())
                    ins.append(f"z{letters[s].upper()}"); ops.append(vecs[s])
            sub = ",".join(ins) + "," + "".join(letters[:k]) + "".join(letters[:k]).upper()
            red = np.einsum(sub + f"->z{letters[t]}{letters[t].upper()}", *ops, T, optimize=True)
            w, v = np.linalg.eigh(red)
            vecs[t] = v[:, :, -1]
            val = w[:, -1]
    return float(sign * val.max())


def L4(al):
    """Closed form of u_W for n = 4 (Theorem 2)."""
    a, b, c, d = al
    cands = [a, b, c, d, (a + c + 1) / 2]
    if a + 1 > max(b, d):
        cands.append(((a + 1) ** 2 - b * d) / (2 * (a + 1) - b - d))
    return max(cands)


def L3(al):
    """Closed form of u_W for n = 3."""
    a, b, c = al
    cands = [a, b, c]
    if a + 1 > max(b, c):
        cands.append(((a + 1) ** 2 - b * c) / (2 * (a + 1) - b - c))
    return max(cands)


def certificate4(al, v):
    """Explicit P >= 0, Q >= 0 with v*1 - W[al] = P + Q^Gamma (Theorem 1). Returns (P, Q, lambda)."""
    a, b, c, d = al
    lam = min(1.0, v - a)
    n = 4
    P = np.zeros((16, 16)); Q = np.zeros((16, 16))
    m = [v - a, v - b, v - c, v - d]
    for i in range(n):
        P[i * n + i, i * n + i] = m[0] - lam
        for j in range(n):
            P[i * n + i, j * n + j] += lam
        for k in (1, 2, 3):
            Q[i * n + (i + k) % n, i * n + (i + k) % n] = m[k]
        for j in range(n):
            if i != j:
                Q[i * n + j, j * n + i] = 1 - lam
    return P, Q, lam


def certificate3(al, v):
    a, b, c = al
    lam = min(1.0, v - a)
    n = 3
    P = np.zeros((9, 9)); Q = np.zeros((9, 9))
    m = [v - a, v - b, v - c]
    for i in range(n):
        P[i * n + i, i * n + i] = m[0] - lam
        for j in range(n):
            P[i * n + i, j * n + j] += lam
        for k in (1, 2):
            Q[i * n + (i + k) % n, i * n + (i + k) % n] = m[k]
        for j in range(n):
            if i != j:
                Q[i * n + j, j * n + i] = 1 - lam
    return P, Q, lam


def classI(th):
    a = (2 - math.sin(th)) / 2; b = (1 + math.cos(th)) / 2
    return [a, b, 2 - a, 1 - b]


def classII(th):
    a = (1 + math.cos(th)) / 2; b = (2 - math.sin(th)) / 2
    return [a, b, 1 - a, 2 - b]


def fam3(phi):
    return [2 / 3 * (1 + math.cos(phi)), (2 - math.cos(phi) - math.sqrt(3) * math.sin(phi)) / 3,
            (2 - math.cos(phi) + math.sqrt(3) * math.sin(phi)) / 3]


rng = np.random.default_rng(20261117)

# ==============================================================================================
header("A. The 4 (x) 4 example of SBHCB Sec. IV.B.1:  M[1,1,1,0] = (4/3) 1 - W[1,1,1,0]")
# ==============================================================================================
al0 = [Fr(1), Fr(1), Fr(1), Fr(0)]
Wx = W_op(al0, exact=True)
e = [Fr(1), Fr(0), Fr(1), Fr(0)]           # |0> + |2>   (unnormalised, norm^2 = 2)
f = [Fr(1), Fr(0), Fr(-1), Fr(0)]          # |0> - |2>
psi = [e[i] * f[j] for i in range(4) for j in range(4)]
expW = sum(psi[r] * Wx[r][c] * psi[c] for r in range(16) for c in range(16)) / 4
print(f"   <psi*|W[1,1,1,0]|psi*> = {expW},   <psi*|(4/3)1 - W|psi*> = {Fr(4, 3) - expW}")
check("exact: <W[1,1,1,0]> = 3/2 on the product state (|0>+|2>)(|0>-|2>)/2", expW == Fr(3, 2))
check("exact: M[1,1,1,0] = (4/3)1 - W[1,1,1,0] has expectation -1/6 < 0 on a product state", Fr(4, 3) - expW == Fr(-1, 6))
# explicit decomposition (3/2)1 - W = 2|phi+><phi+| + Q^Gamma, in exact arithmetic
P = [[Fr(0)] * 16 for _ in range(16)]; Q = [[Fr(0)] * 16 for _ in range(16)]
for i in range(4):
    for j in range(4):
        P[i * 4 + i][j * 4 + j] = Fr(1, 2)
        if i != j:
            Q[i * 4 + j][j * 4 + i] = Fr(1, 2)
    Q[i * 4 + (i + 1) % 4][i * 4 + (i + 1) % 4] = Fr(1, 2)
    Q[i * 4 + (i + 2) % 4][i * 4 + (i + 2) % 4] = Fr(1, 2)
    Q[i * 4 + (i + 3) % 4][i * 4 + (i + 3) % 4] = Fr(3, 2)
QG = [[Fr(0)] * 16 for _ in range(16)]
for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                QG[k * 4 + j][i * 4 + l] = Q[i * 4 + j][k * 4 + l]
ident = all((Fr(3, 2) if r == c else Fr(0)) - Wx[r][c] == P[r][c] + QG[r][c] for r in range(16) for c in range(16))
evP = sp.Matrix(P).eigenvals(); evQ = sp.Matrix(Q).eigenvals()
print("   spectrum of P:", {str(k): v for k, v in evP.items()}, "  spectrum of Q:", {str(k): v for k, v in evQ.items()})
check("exact: (3/2)1 - W[1,1,1,0] = P + Q^Gamma with P = 2|phi+><phi+|", ident)
check("exact: P >= 0 and Q >= 0 (all eigenvalues non-negative)", min(evP) >= 0 and min(evQ) >= 0)
Wn = W_op([1, 1, 1, 0])
umax = product_extremum(Wn, (4, 4), +1, batch=400, iters=300)
check(f"numerical maximum of <ef|W[1,1,1,0]|ef> over product states = {umax:.9f} (= 3/2)", abs(umax - 1.5) < 1e-8)
check(f"numerical minimum over product states = {product_extremum(Wn, (4, 4), -1):+.2e} (W is block positive)",
      product_extremum(Wn, (4, 4), -1) > -1e-8)
ev = np.linalg.eigvalsh(1.5 * np.eye(16) - Wn)
check(f"(3/2)1 - W[1,1,1,0] is not PSD (lambda_min = {ev[0]:.3f}): a decomposable witness", ev[0] < -0.4)
# share of Haar-random product states on which the printed mirror is negative
Nmc = 400000
ee = rng.normal(size=(Nmc, 4)) + 1j * rng.normal(size=(Nmc, 4)); ee /= np.linalg.norm(ee, axis=1, keepdims=True)
ff = rng.normal(size=(Nmc, 4)) + 1j * rng.normal(size=(Nmc, 4)); ff /= np.linalg.norm(ff, axis=1, keepdims=True)
pp, qq = np.abs(ee) ** 2, np.abs(ff) ** 2
vals = sum(([1, 1, 1, 0][k] + (1 if k == 0 else 0)) * np.sum(pp * np.roll(qq, -k, axis=1), axis=1) for k in range(4)) \
    - np.abs(np.sum(ee * ff, axis=1)) ** 2
frac = float(np.mean(vals > 4 / 3))
print(f"   Monte Carlo ({Nmc} Haar-random product states, fixed seed): fraction with <(4/3)1 - W> < 0 = {frac:.5f};"
      f" largest sampled <W> = {vals.max():.4f}")
check("a set of product states of positive measure violates the printed mirror", frac > 0)
# SBHCB (75): PPT state rho_x and the trace formulas
x = sp.symbols("x", positive=True)
rho = sp.zeros(16, 16)
for i in range(4):
    for k, val in enumerate([3, x, 1, 1 / x]):
        rho[i * 4 + (i + k) % 4, i * 4 + (i + k) % 4] = val
    for j in range(4):
        if i != j:
            rho[i * 4 + i, j * 4 + j] = -1
Ws = sp.Matrix(W_op([sp.Integer(1), sp.Integer(1), sp.Integer(1), sp.Integer(0)], exact=True))
t43 = sp.simplify((sp.Rational(4, 3) * sp.eye(16) - Ws).multiply(rho).trace())
t32 = sp.simplify((sp.Rational(3, 2) * sp.eye(16) - Ws).multiply(rho).trace())
print(f"   Tr[((4/3)1-W) rho_x] = {sp.factor(t43)};   Tr[((3/2)1-W) rho_x] = {sp.factor(t32)}")
check("with 4/3 the (unnormalised) value is 4(x^2-5x+4)/(3x), as printed in SBHCB",
      sp.simplify(t43 - 4 * (x ** 2 - 5 * x + 4) / (3 * x)) == 0)
check("with the correct constant 3/2 it is 2(x^2-2x+3)/x > 0: rho_x is not detected",
      sp.simplify(t32 - 2 * (x ** 2 - 2 * x + 3) / x) == 0)
okppt = True
for xv in (0.3, 1.0, 1.7, 2.5, 4.0, 9.0):
    r = np.array(rho.subs(x, xv), dtype=float)
    okppt &= np.linalg.eigvalsh(r)[0] > -TOL and np.linalg.eigvalsh(PT_A(r, 4, 4))[0] > -TOL
check("rho_x of SBHCB (75) is positive and PPT for the sampled x", okppt)

# ==============================================================================================
header("B1. Theorem 1 (certificate) and Theorem 2 (closed form of u_W) for arbitrary real (a,b,c,d)")
# ==============================================================================================
worst_cert, worst_gap, worst_att = 0.0, 0.0, 0.0
NR = 300
for t in range(NR):
    al = list(rng.uniform(-0.5, 2.5, size=4)) if t % 3 else list(np.round(rng.uniform(0, 2, size=4) * 4) / 4)
    v = L4(al)
    W = W_op(al)
    Pm, Qm, lam = certificate4(al, v)
    resid = np.abs(v * np.eye(16) - W - Pm - PT_A(Qm, 4, 4)).max()
    worst_cert = max(worst_cert, resid, -np.linalg.eigvalsh(Pm)[0], -np.linalg.eigvalsh(Qm)[0])
    if t < 120:
        worst_gap = max(worst_gap, abs(product_extremum(W, (4, 4), +1, batch=200, iters=400, seed=t) - v))
    # attainment by the three explicit product families
    a, b, c, d = al
    best = max(a, b, c, d, (a + c + 1) / 2)
    if a + 1 > max(b, d):
        ts = (a + 1 - d) / (2 * (a + 1) - b - d)
        ev_ = np.array([math.sqrt(ts), math.sqrt(1 - ts), 0, 0]); fv = np.array([math.sqrt(1 - ts), -math.sqrt(ts), 0, 0])
        psi_ = np.kron(ev_, fv)
        best = max(best, psi_ @ W @ psi_)
    worst_att = max(worst_att, abs(best - v))
check(f"{NR} random (a,b,c,d): L*1 - W = P + Q^Gamma with P, Q >= 0 (worst violation {worst_cert:.1e})", worst_cert < 1e-9)
check(f"{NR} random (a,b,c,d): L is attained by an explicit product state (worst gap {worst_att:.1e})", worst_att < 1e-9)
check(f"120 random (a,b,c,d): numerical product-state maximum equals the closed form L (worst gap {worst_gap:.1e}, tolerance 1e-5)", worst_gap < 1e-5)
# explicit product families: exact values
a_, b_, c_, d_, t_ = sp.symbols("a b c d t", real=True)
Wsym = sp.Matrix(W_op([a_, b_, c_, d_], exact=True))
evA = sp.Matrix([1, 0, 1, 0]) / sp.sqrt(2); fvA = sp.Matrix([1, 0, -1, 0]) / sp.sqrt(2)
psiA = sp.kronecker_product(evA, fvA)
valA = sp.simplify((psiA.T * Wsym * psiA)[0])
evB = sp.Matrix([sp.sqrt(t_), sp.sqrt(1 - t_), 0, 0]); fvB = sp.Matrix([sp.sqrt(1 - t_), -sp.sqrt(t_), 0, 0])
psiB = sp.kronecker_product(evB, fvB)
valB = sp.expand((psiB.T * Wsym * psiB)[0])
check("symbolic: antipodal family gives (a+c+1)/2", sp.simplify(valA - (a_ + c_ + 1) / 2) == 0)
check("symbolic: adjacent family gives 2(a+1)t(1-t) + b t^2 + d (1-t)^2",
      sp.simplify(valB - (2 * (a_ + 1) * t_ * (1 - t_) + b_ * t_ ** 2 + d_ * (1 - t_) ** 2)) == 0)

# ==============================================================================================
header("B2. The two classes of SBHCB (73): exact u_W and the nature of the mirrored operator")
# ==============================================================================================
th = sp.symbols("theta", real=True)
aI = (2 - sp.sin(th)) / 2; bI = (1 + sp.cos(th)) / 2; cI = 2 - aI; dI = 1 - bI
check("class I satisfies SBHCB (73): a+b+c+d = a^2+b^2+c^2+d^2 = 3, ac+bd = 1, (a+c)(b+d) = 2",
      all(sp.simplify(z) == 0 for z in (aI + bI + cI + dI - 3, aI ** 2 + bI ** 2 + cI ** 2 + dI ** 2 - 3, aI * cI + bI * dI - 1,
                                        (aI + cI) * (bI + dI) - 2)))
check("class I: (a+c+1)/2 = 3/2 identically", sp.simplify((aI + cI + 1) / 2 - sp.Rational(3, 2)) == 0)
okI, okI_cert, minlam = True, True, 9.0
grid = np.linspace(0, 2 * np.pi, 721)
for tv in grid:
    al = classI(tv)
    okI &= abs(L4(al) - 1.5) < 1e-12
    Pm, Qm, lam = certificate4(al, 1.5)
    okI_cert &= np.abs(1.5 * np.eye(16) - W_op(al) - Pm - PT_A(Qm, 4, 4)).max() < 1e-12 and \
        np.linalg.eigvalsh(Pm)[0] > -1e-12 and np.linalg.eigvalsh(Qm)[0] > -1e-12
check("class I, 721 values of theta in [0, 2pi]: closed form gives u_W = 3/2", okI)
check("class I, same grid: (3/2)1 - W = 2(1+sin theta)|phi+><phi+| + Q^Gamma with Q >= 0", okI_cert)
num_I = max(abs(product_extremum(W_op(classI(tv)), (4, 4), +1, batch=150, iters=150, seed=7) - 1.5) for tv in np.linspace(0, 2 * np.pi, 25))
check(f"class I, 25 values of theta: numerical product-state maximum = 3/2 (worst deviation {num_I:.1e})", num_I < 1e-7)
psd_at = [tv for tv in grid if max(classI(tv)[0] + 1, max(classI(tv)[1:])) <= 1.5 + 1e-12]
check("class I: (3/2)1 - W is PSD only at theta = pi/2; otherwise it is a decomposable witness",
      len(psd_at) == 1 and abs(psd_at[0] - np.pi / 2) < 1e-9)
aII = (1 + sp.cos(th)) / 2; bII = (2 - sp.sin(th)) / 2; cII = 1 - aII; dII = 2 - bII
check("class II satisfies SBHCB (73)",
      all(sp.simplify(z) == 0 for z in (aII + bII + cII + dII - 3, aII ** 2 + bII ** 2 + cII ** 2 + dII ** 2 - 3,
                                        aII * cII + bII * dII - 1, (aII + cII) * (bII + dII) - 2)))
Bstar = sp.simplify(((aII + 1) ** 2 - bII * dII) / (2 * (aII + 1) - bII - dII))
check("class II: the adjacent-family value ((a+1)^2 - bd)/(2(a+1)-b-d) equals 3/2 identically", sp.simplify(Bstar - sp.Rational(3, 2)) == 0)
okII = True
for tv in np.linspace(0, np.pi, 361):
    al = classII(tv)
    expect = 1.5 if tv <= np.pi / 2 + 1e-12 else al[3]
    okII &= abs(L4(al) - expect) < 1e-10
    lam_max = max(al[0] + 1, max(al[1:]))
    if tv > np.pi / 2 + 1e-9:
        okII &= abs(L4(al) - lam_max) < 1e-10            # mirror at mu = u_W is PSD
    elif tv < np.pi / 2 - 1e-9:
        okII &= lam_max > 1.5 + 1e-12                     # mirror at mu = u_W is a (decomposable) witness
check("class II: u_W = 3/2 for theta in [0, pi/2] and u_W = d = lambda_max(W) for theta in [pi/2, pi]", okII)
num_II = max(abs(product_extremum(W_op(classII(tv)), (4, 4), +1, batch=150, iters=150, seed=5) - L4(classII(tv)))
             for tv in np.linspace(0, np.pi, 13))
check(f"class II, 13 values of theta: numerical product-state maximum = closed form (worst deviation {num_II:.1e}, tolerance 1e-5)", num_II < 1e-5)


def ppt_min_state(al, X):
    """Covariant PPT state minimising Tr(W rho) for given diagonal weights X (sum 1)."""
    g = min(X[0], X[2]); tt = min(math.sqrt(X[1] * X[3]), (X[0] + g) / 2)
    r = np.zeros((16, 16))
    for i in range(4):
        for k in range(4):
            r[i * 4 + (i + k) % 4, i * 4 + (i + k) % 4] = X[k] / 4
        for j in range(4):
            if i != j:
                r[i * 4 + i, j * 4 + j] = (tt if (j - i) % 4 in (1, 3) else g) / 4
    return r


def ppt_min(al, tries=4000, seed=1):
    r_ = np.random.default_rng(seed)
    best = (1e9, None)
    for _ in range(tries):
        X = r_.dirichlet(np.ones(4) * r_.choice([0.3, 1.0, 3.0]))
        rho_ = ppt_min_state(al, X)
        val = float(np.trace(W_op(al) @ rho_))
        if val < best[0]:
            best = (val, rho_)
    return best


okdec = True
for tv in (0.0, 0.6, 1.2, 2.0, 2.6, np.pi, 4.0, 5.5):
    val, rho_ = ppt_min(classI(tv))
    okdec &= val < -1e-3 and np.linalg.eigvalsh(rho_)[0] > -1e-12 and np.linalg.eigvalsh(PT_A(rho_, 4, 4))[0] > -1e-12
check("class I: W(theta) detects a PPT state (is non-decomposable) for the sampled theta != pi/2", okdec)
# W(pi/2) = W[1/2,1/2,3/2,1/2] = P + Q^Gamma with P = 2|chi><chi|, chi = (1/2) sum_i (-1)^i |ii>
Wh = W_op(classI(np.pi / 2))
Ph = np.zeros((16, 16)); Qh = np.zeros((16, 16))
for i in range(4):
    for j in range(4):
        Ph[i * 4 + i, j * 4 + j] = 0.5 * (-1) ** (i + j)
        if i != j:
            Qh[i * 4 + j, j * 4 + i] = -0.5 if (j - i) % 4 in (1, 3) else -1.5
    Qh[i * 4 + (i + 1) % 4, i * 4 + (i + 1) % 4] = 0.5
    Qh[i * 4 + (i + 2) % 4, i * 4 + (i + 2) % 4] = 1.5
    Qh[i * 4 + (i + 3) % 4, i * 4 + (i + 3) % 4] = 0.5
check("class I: W(pi/2) = W[1/2,1/2,3/2,1/2] = P + Q^Gamma with P, Q >= 0 (decomposable, as stated in SBHCB)",
      np.allclose(Wh, Ph + PT_A(Qh, 4, 4)) and np.linalg.eigvalsh(Ph)[0] > -1e-12 and np.linalg.eigvalsh(Qh)[0] > -1e-12)

# ==============================================================================================
header("B3. The 3 (x) 3 family SBHCB (70)-(71): closed form, the constant 4/3 and Proposition 6")
# ==============================================================================================
check("Choi witness W[1,1,0]: u_W = 4/3 (closed form and numerics)",
      abs(L3([1, 1, 0]) - 4 / 3) < 1e-12 and abs(product_extremum(W_op([1, 1, 0]), (3, 3), +1) - 4 / 3) < 1e-8)
check("4 (x) 4, W[1,1,1,0]: product vectors supported on two adjacent levels reach at most 4/3",
      abs(((1 + 1) ** 2 - 1 * 0) / (2 * 2 - 1 - 0) - 4 / 3) < 1e-12)
ok3, ok3c, okP6 = True, True, True
for i, ph in enumerate(np.linspace(0, 2 * np.pi, 49)):
    al = fam3(ph); v = L3(al); W = W_op(al)
    Pm, Qm, lam = certificate3(al, v)
    ok3c &= np.abs(v * np.eye(9) - W - Pm - PT_A(Qm, 3, 3)).max() < 1e-10 and np.linalg.eigvalsh(Pm)[0] > -1e-10 and \
        np.linalg.eigvalsh(Qm)[0] > -1e-10
    if i % 4 == 0:
        ok3 &= abs(product_extremum(W, (3, 3), +1, batch=150, iters=400, seed=i) - v) < 1e-5
    lam_max = max(al[0] + 1, al[1], al[2])
    psd = abs(v - lam_max) < 1e-10
    if abs(al[0] - 1 / 3) > 1e-6:
        okP6 &= psd == (al[0] < 1 / 3)
check("3 (x) 3 family: numerical product-state maximum equals the closed form", ok3)
check("3 (x) 3 family: u_W*1 - W = P + Q^Gamma with P, Q >= 0 (mirror never non-decomposable)", ok3c)
check("SBHCB Proposition 6: the mirror at mu = u_W is PSD iff a <= 1/3, a decomposable witness otherwise", okP6)

# ==============================================================================================
header("B4. Arbitrary dimension n: closed form, certificate and attaining product vectors (Theorems 2 and 3)")
# ==============================================================================================


def B_term(al, k):
    n = len(al)
    a0, x, y = al[0], al[k], al[(n - k) % n]
    if a0 + 1 > max(x, y):
        return ((a0 + 1) ** 2 - x * y) / (2 * (a0 + 1) - x - y)
    return -math.inf


def Ln(al):
    n = len(al)
    return max(list(al) + [B_term(al, k) for k in range(1, n // 2 + 1)])


def certificate_n(al, v):
    n = len(al)
    lam = min(1.0, v - al[0])
    P = np.zeros((n * n, n * n)); Q = np.zeros((n * n, n * n))
    for i in range(n):
        P[i * n + i, i * n + i] = v - al[0] - lam
        for j in range(n):
            P[i * n + i, j * n + j] += lam
            if i != j:
                Q[i * n + j, j * n + i] = 1 - lam
        for k in range(1, n):
            Q[i * n + (i + k) % n, i * n + (i + k) % n] = v - al[k]
    return P, Q, lam


def best_two_level_value(al):
    """Largest value of <W> on basis vectors and on the two-level product vectors of Theorem 3."""
    n = len(al)
    W = W_op(al)
    best = max(al)
    for k in range(1, n // 2 + 1):
        x, y = al[k], al[(n - k) % n]
        if al[0] + 1 > max(x, y):
            ts = (al[0] + 1 - y) / (2 * (al[0] + 1) - x - y)
            ev_ = np.zeros(n); fv = np.zeros(n)
            ev_[0], ev_[k] = math.sqrt(ts), math.sqrt(1 - ts)
            fv[0], fv[k] = math.sqrt(1 - ts), -math.sqrt(ts)
            psi_ = np.kron(ev_, fv)
            best = max(best, float(psi_ @ W @ psi_))
    return best


check("n = 4 and n = 3: the general formula reduces to L4 and L3",
      all(abs(Ln(al) - L4(al)) < 1e-12 for al in [list(rng.uniform(-0.5, 2.5, 4)) for _ in range(200)]) and
      all(abs(Ln(al) - L3(al)) < 1e-12 for al in [list(rng.uniform(-0.5, 2.5, 3)) for _ in range(200)]))
for n in (2, 3, 4, 5, 6, 7, 8):
    wc, wa, wl = 0.0, 0.0, 0.0
    for t in range(150):
        al = list(rng.uniform(-0.5, 2.5, size=n)) if t % 3 else list(np.round(rng.uniform(0, 2, size=n) * 4) / 4)
        v = Ln(al); W = W_op(al)
        Pm, Qm, lam = certificate_n(al, v)
        wc = max(wc, np.abs(v * np.eye(n * n) - W - Pm - PT_A(Qm, n, n)).max(), -np.linalg.eigvalsh(Pm)[0], -np.linalg.eigvalsh(Qm)[0])
        wa = max(wa, abs(best_two_level_value(al) - v))
        wl = max(wl, abs(np.linalg.eigvalsh(W)[-1] - max(al[0] + 1, max(al[1:]))))
    check(f"n={n}: 150 random alpha: L*1 - W = P + Q^Gamma with P, Q >= 0 (worst violation {wc:.1e}); "
          f"L attained on a two-level product vector (worst gap {wa:.1e}); lambda_max formula (gap {wl:.1e})",
          wc < 1e-9 and wa < 1e-9 and wl < 1e-9)
for n, nsamp in ((2, 12), (5, 12), (6, 10), (7, 6)):
    wg = 0.0
    for t in range(nsamp):
        al = list(rng.uniform(0, 2, size=n))
        wg = max(wg, abs(product_extremum(W_op(al), (n, n), +1, batch=200, iters=300, seed=100 + t) - Ln(al)))
    check(f"n={n}: numerical product-state maximum equals the closed form for {nsamp} random alpha "
          f"(worst gap {wg:.1e}, tolerance 1e-5)", wg < 1e-5)
vals_choi = {n: Ln([1.0] * (n - 1) + [0.0]) for n in range(3, 10)}
print("   u_W of the Choi-type witness W[1,...,1,0] in dimension n:", {n: round(v, 6) for n, v in vals_choi.items()})
check("Choi-type witness W[1,...,1,0]: u_W = 4/3 for n = 3 and u_W = 3/2 for every n >= 4",
      abs(vals_choi[3] - 4 / 3) < 1e-12 and all(abs(vals_choi[n] - 1.5) < 1e-12 for n in range(4, 10)))

# ==============================================================================================
header("C1. The 3 (x) 3 pair of SBHCB (77)-(78)")
# ==============================================================================================
W77 = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 1], [0, 3, 0, 0, 0, -2, -2, 0, 0], [0, 0, 3, -2, 0, 0, 0, -2, 0],
                [0, 0, -2, 3, 0, 0, 0, -2, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1], [0, -2, 0, 0, 0, 3, -2, 0, 0],
                [0, -2, 0, 0, 0, -2, 3, 0, 0], [0, 0, -2, -2, 0, 0, 0, 3, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=float)
M77 = 4 * np.eye(9) - W77
M77_printed = np.array([[4, 0, 0, 0, -1, 0, 0, 0, -1], [0, 1, 0, 0, 0, 2, 2, 0, 0], [0, 0, 1, 2, 0, 0, 0, 2, 0],
                        [0, 0, 2, 1, 0, 0, 0, 2, 0], [-1, 0, 0, 0, 4, 0, 0, 0, -1], [0, 2, 0, 0, 0, 1, 2, 0, 0],
                        [0, 2, 0, 0, 0, 2, 1, 0, 0], [0, 0, 2, 2, 0, 0, 0, 1, 0], [-1, 0, 0, 0, -1, 0, 0, 0, 4]], dtype=float)
rhoW = np.array([[3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 0],
                 [0, 0, 1, 1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 3, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 1, 0, 0],
                 [0, 1, 0, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 3]], dtype=float) / 15
rhoM = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 1], [0, 2, 0, 0, 0, -1, -1, 0, 0], [0, 0, 2, -1, 0, 0, 0, -1, 0],
                 [0, 0, -1, 2, 0, 0, 0, -1, 0], [1, 0, 0, 0, 1, 0, 0, 0, 1], [0, -1, 0, 0, 0, 2, -1, 0, 0],
                 [0, -1, 0, 0, 0, -1, 2, 0, 0], [0, 0, -1, -1, 0, 0, 0, 2, 0], [1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=float) / 15
check("SBHCB (77): W + M = 4*1 for the printed matrices", np.allclose(M77, M77_printed))
mW = product_extremum(W77, (3, 3), -1, batch=300, iters=300); mM = product_extremum(M77, (3, 3), -1, batch=300, iters=300)
check(f"(77): both operators are block positive numerically (minima {mW:+.1e}, {mM:+.1e})", mW > -1e-7 and mM > -1e-7)
okst = all(np.linalg.eigvalsh(r)[0] > -TOL and np.linalg.eigvalsh(PT_A(r, 3, 3))[0] > -TOL and abs(np.trace(r) - 1) < TOL
           for r in (rhoW, rhoM))
check("(77): rho_W and rho_M are PPT states with Tr[W rho_W] = Tr[M rho_M] = -2/5 (non-decomposable pair)",
      okst and abs(np.trace(W77 @ rhoW) + 0.4) < TOL and abs(np.trace(M77 @ rhoM) + 0.4) < TOL)
om = np.exp(2j * np.pi / 3)
U78 = np.array([[1, 1, om], [1, om, 1], [om.conjugate(), om, om]]) / np.sqrt(3)
UU = np.kron(U78, U78.conj())
check("(78): U is unitary and M = (U (x) U*) W (U (x) U*)^dagger",
      np.allclose(U78 @ U78.conj().T, np.eye(3)) and np.allclose(UU @ W77 @ UU.conj().T, M77, atol=1e-10))

# ==============================================================================================
header("C2. SBHCB Sec. V: generalised mirroring with a Weyl operator (class II)")
# ==============================================================================================
xs = sp.symbols("x", positive=True)


def cov_state(diag, off):
    r = sp.zeros(16, 16)
    for i in range(4):
        for k in range(4):
            r[i * 4 + (i + k) % 4, i * 4 + (i + k) % 4] = diag[k]
        for j in range(4):
            if i != j:
                r[i * 4 + i, j * 4 + j] = off
    return r


WII = sp.Matrix(W_op([a_, b_, c_, d_], exact=True))
tr75 = sp.simplify(WII.multiply(cov_state([3, xs, 1, 1 / xs], -1)).trace() / 4)
trst = sp.simplify(WII.multiply(cov_state([1, xs, 1, 1 / xs], 1)).trace() / 4)
print(f"   per-site trace with rho_x of (75): {tr75};   with diag (1,x,1,1/x), off-diagonal +1: {trst}")
check("the printed formula a + bx + c + d/x - 3 holds for the state with diagonal (1,x,1,1/x) and off-diagonal +1,"
      " not for rho_x of (75)", sp.simplify(trst - (a_ + b_ * xs + c_ + d_ / xs - 3)) == 0 and
      sp.simplify(tr75 - (3 * a_ + b_ * xs + c_ + d_ / xs + 3)) == 0)
r_ = np.array(cov_state([1, 1.7, 1, 1 / 1.7], 1), dtype=float)
check("that state is positive and PPT", np.linalg.eigvalsh(r_)[0] > -TOL and np.linalg.eigvalsh(PT_A(r_, 4, 4))[0] > -TOL)

# ==============================================================================================
header("C3. Multipartite part of SBHCB (Sec. III): GHZ witnesses and the three-qubit pair")
# ==============================================================================================
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], dtype=complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1.0, -1.0]).astype(complex)


def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def ghz_generators(n):
    gens = [kron(*[X] * n)]
    for i in range(1, n):
        gens.append(kron(*[Z if q in (i - 1, i) else I2 for q in range(n)]))
    return gens


for n in (3, 4):
    D = 2 ** n
    gens = ghz_generators(n)
    S = sum(gens)
    ghz = np.zeros(D, dtype=complex); ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    Pg = np.outer(ghz, ghz.conj())
    Wc = (0.5 * np.eye(D) - Pg) / (2 ** (n - 1) - 1)
    resid = Wc + Pg - np.eye(D) * np.trace(Wc + Pg).real / D
    check(f"n={n}: canonical pair as printed, W_c + |GHZ><GHZ|, is not a multiple of the identity "
          f"(it is for mu_c*1 - W_c = |GHZ><GHZ|/(2^(n-1)-1))",
          np.abs(resid).max() > 1e-3 and np.allclose(np.eye(D) / (2 ** n - 2) - Wc, Pg / (2 ** (n - 1) - 1)))
    Wa = (np.eye(D) - S / (n - 1)) / D; Ma = (np.eye(D) + S / (n - 1)) / D
    check(f"n={n}: alternative pair W_a + M_a = 2^(1-n)*1", np.allclose(Wa + Ma, np.eye(D) / 2 ** (n - 1)))
    mx = product_extremum(S.real if np.allclose(S.imag, 0) else S, (2,) * n, +1, batch=200, iters=100)
    mn = product_extremum(S, (2,) * n, -1, batch=200, iters=100)
    check(f"n={n}: sum of generators over product states lies in [-(n-1), n-1] (numerics: [{mn:.6f}, {mx:.6f}])",
          abs(mx - (n - 1)) < 1e-7 and abs(mn + (n - 1)) < 1e-7)
    nrm = np.linalg.norm(Wa, 2)
    check(f"n={n}: ||W_a||_op = (2n-1)/((n-1)2^n) = {(2 * n - 1) / ((n - 1) * D):.5f}, not n/((n-1)2^n) = "
          f"{n / ((n - 1) * D):.5f} as in SBHCB (50)", abs(nrm - (2 * n - 1) / ((n - 1) * D)) < 1e-12)
    check(f"n={n}: Tr[W_a GHZ] = -1/((n-1)2^n); hence |Tr|/(2||W||) = 1/(2(2n-1)), not 1/(2n) [SBHCB (49)]",
          abs(np.trace(Wa @ Pg).real + 1 / ((n - 1) * D)) < 1e-12)
    P1 = (np.eye(D) + gens[0]) / 2
    P2 = np.eye(D)
    for g in gens[1:]:
        P2 = P2 @ (np.eye(D) + g) / 2
    W2m = (1.5 * np.eye(D) - P1 - P2) / (D - 2); M2m = (0.5 * np.eye(D) + P1 + P2) / (D + 2)
    s2 = W2m + M2m
    check(f"n={n}: W_2m + M_2m as printed is not a multiple of the identity; (2^n-2)W_2m + (2^n+2)M_2m = 2*1",
          np.abs(s2 - np.eye(D) * np.trace(s2).real / D).max() > 1e-4 and np.allclose((D - 2) * W2m + (D + 2) * M2m, 2 * np.eye(D)))
    # local unitary taking W_a to M_a and the state of SBHCB (40)
    U_paper = kron(*[Y if (q % 2 == 1) else I2 for q in range(n)])        # 1 (x) Y (x) 1 (x) Y ...
    flips = np.allclose(U_paper @ Wa @ U_paper.conj().T, Ma)
    bits_a = [q % 2 for q in range(n)]
    ia = int("".join(map(str, bits_a)), 2); ib = D - 1 - ia
    ghza = np.zeros(D, dtype=complex); ghza[ia] = 1 / np.sqrt(2); ghza[ib] = -1 / np.sqrt(2)
    det = np.trace(Ma @ np.outer(ghza, ghza.conj())).real
    check(f"n={n}: M_a detects the state of SBHCB (40) with Tr = -1/((n-1)2^n)", abs(det + 1 / ((n - 1) * D)) < 1e-12)
    check(f"n={n}: the printed local unitary 1(x)Y(x)1(x)Y... maps W_a to M_a: {flips} "
          f"(it does so iff the number floor(n/2) of Y factors is odd)", flips == ((n // 2) % 2 == 1))
    if not flips:
        U_fix = kron(*([Z] + [Y if (q % 2 == 1) else I2 for q in range(1, n)]))
        check(f"n={n}: Z(x)Y(x)1(x)Y... does map W_a to M_a", np.allclose(U_fix @ Wa @ U_fix.conj().T, Ma))
for n in (5, 6):                                              # local-unitary statement only
    D = 2 ** n
    S = sum(ghz_generators(n))
    U_paper = kron(*[Y if (q % 2 == 1) else I2 for q in range(n)])
    flips = np.allclose(U_paper @ S @ U_paper.conj().T, -S)
    check(f"n={n}: the printed local unitary flips the sign of all generators: {flips} (iff floor(n/2) is odd)",
          flips == ((n // 2) % 2 == 1))

# three-qubit family SBHCB (60)-(61)
okW3, okmin = True, True
for i1, i2, i3 in itertools.product((0, 1), repeat=3):
    T = kron(Z, Z, Z) + (-1) ** i1 * kron(X, X, X) + (-1) ** i2 * kron(X, Y, Y) + (-1) ** i3 * kron(Y, X, Y) \
        + (-1) ** (i1 + i2 + i3 + 1) * kron(Y, Y, X)
    W3 = np.eye(8) - T; M3 = np.eye(8) + T
    YYY = kron(Y, Y, Y)
    okW3 &= np.allclose(YYY @ W3 @ YYY.conj().T, M3) and abs(np.linalg.norm(W3, 2) - (1 + math.sqrt(17))) < 1e-10
    okmin &= product_extremum(W3, (2, 2, 2), -1, batch=200, iters=150) > -1e-7
check("three-qubit W[i1,i2,i3]: W + M = 2*1, M = YYY W YYY, ||W||_op = 1 + sqrt(17) for all eight sign choices", okW3)
check("three-qubit W[i1,i2,i3]: non-negative on product states (numerical minimum 0) for all eight sign choices", okmin)
def W3q(i1, i2, i3):                                         # SBHCB (60) as printed
    return np.eye(8) - (kron(Z, Z, Z) + (-1) ** i1 * kron(X, X, X) + (-1) ** i2 * kron(X, Y, Y)
                        + (-1) ** i3 * kron(Y, X, Y) + (-1) ** (i1 + i2 + i3 + 1) * kron(Y, Y, X))


def ppt_all_cuts(r):
    ok = np.linalg.eigvalsh(r)[0] > -1e-12
    for cut in range(3):
        perm = [cut] + [q for q in range(3) if q != cut]
        rp = r.reshape([2] * 6).transpose(perm + [3 + p for p in perm]).reshape(8, 8)
        ok &= np.linalg.eigvalsh(PT_A(rp, 2, 4))[0] > -1e-12
    return ok


labels = list(itertools.product((0, 1), repeat=3))
ok_ppt, ok_011, ok_010, only_011 = True, True, True, True
for (xv, yv, zv) in ((0.5, 0.5, 2.0), (0.3, 0.8, 1.5), (2.0, 3.0, 0.7), (0.5, 0.4, 2.0)):
    nu = 2 + xv + yv + zv + 1 / xv + 1 / yv + 1 / zv
    r = np.diag([1, xv, yv, zv, 1 / zv, 1 / yv, 1 / xv, 1]).astype(complex); r[0, 7] = r[7, 0] = 1; r /= nu
    ok_ppt &= ppt_all_cuts(r)
    target = 2 * (xv + yv + 1 / zv - 3) / nu
    match = [lab for lab in labels if abs(np.trace(W3q(*lab) @ r).real - target) < 1e-12]
    only_011 &= match == [(0, 1, 1)]
    ok_010 &= abs(np.trace(W3q(0, 1, 0) @ r).real - 2 * (xv + yv + 1 / zv + 1) / nu) < 1e-12
check("rho(x,y,z) of SBHCB is positive and PPT across all three cuts", ok_ppt)
check("the printed value 2(x+y+1/z-3)/nu is Tr[W[0,1,1] rho(x,y,z)]; no other label of (60) reproduces it", only_011)
check("for the label W[0,1,0] named in SBHCB, (60) as printed gives 2(x+y+1/z+1)/nu > 0", ok_010)
ok_pptb, ok_110, ok_min = True, True, True
for (bv, cv) in ((1.0, 1.0), (2.0, 0.5), (0.5, 2.0), (4.0, 2.5), (3.0, 0.4)):
    r = np.diag([1, 1, 1, bv, cv, 1, 1, 1]).astype(complex)
    r[0, 7] = r[7, 0] = -1; r[1, 6] = r[6, 1] = -1; r[2, 5] = r[5, 2] = 1; r[3, 4] = r[4, 3] = -1
    r /= 6 + bv + cv
    ok_pptb &= ppt_all_cuts(r)
    vals3 = {lab: np.trace(W3q(*lab) @ r).real for lab in labels}
    ok_110 &= abs(vals3[(1, 1, 0)] - 2 * (cv + 7) / (bv + cv + 6)) < 1e-12
    ok_min &= abs(min(vals3.values()) - 2 * (cv - 1) / (bv + cv + 6)) < 1e-12 and \
        all(abs(v - 2 * (cv - 3) / (bv + cv + 6)) > 1e-6 for v in vals3.values())
check("rho(b,c) of SBHCB is positive and PPT across all three cuts for the sampled bc >= 1", ok_pptb)
check("for (60) and rho(b,c) as printed, Tr[W[1,1,0] rho(b,c)] = 2(c+7)/(b+c+6) > 0", ok_110)
check("no label of (60) gives the printed 2(c-3)/(b+c+6); the smallest value over the eight labels is 2(c-1)/(b+c+6)", ok_min)

# ==============================================================================================
header("D. A non-decomposable mirrored pair in 4 (x) 4 by embedding the 3 (x) 3 pair (Proposition 4)")
# ==============================================================================================
emb = np.zeros((16, 9))
for i in range(3):
    for j in range(3):
        emb[i * 4 + j, i * 3 + j] = 1
Pi = emb @ emb.T
W4 = emb @ W77 @ emb.T + 2 * (np.eye(16) - Pi)
M4 = emb @ M77 @ emb.T + 2 * (np.eye(16) - Pi)
check("W4 + M4 = 4*1 on C^4 (x) C^4", np.allclose(W4 + M4, 4 * np.eye(16)))
m4w = product_extremum(W4, (4, 4), -1, batch=400, iters=300); m4m = product_extremum(M4, (4, 4), -1, batch=400, iters=300)
check(f"W4 and M4 are block positive (numerical minima over product states {m4w:+.1e}, {m4m:+.1e})", m4w > -1e-7 and m4m > -1e-7)
r4w = emb @ rhoW @ emb.T; r4m = emb @ rhoM @ emb.T
ok4 = all(np.linalg.eigvalsh(r)[0] > -TOL and np.linalg.eigvalsh(PT_A(r, 4, 4))[0] > -TOL for r in (r4w, r4m))
check("the embedded states are PPT and Tr[W4 rho_W] = Tr[M4 rho_M] = -2/5: both witnesses are non-decomposable",
      ok4 and abs(np.trace(W4 @ r4w) + 0.4) < TOL and abs(np.trace(M4 @ r4m) + 0.4) < TOL)
check("neither W4 nor M4 is PSD", np.linalg.eigvalsh(W4)[0] < -1e-6 and np.linalg.eigvalsh(M4)[0] < -1e-6)

# ----------------------------------------------------------------------------------------------
header("SUMMARY")
n_fail = sum(1 for _, ok in RESULTS if not ok)
print(f"  {len(RESULTS) - n_fail} / {len(RESULTS)} checks passed;  runtime {time.time() - T0:.1f} s")
if n_fail:
    print("  FAILED CHECKS:")
    for name, ok in RESULTS:
        if not ok:
            print("    -", name)
sys.exit(1 if n_fail else 0)

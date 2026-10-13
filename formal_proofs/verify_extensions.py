"""
Checks for the extension theorems (Theorems 11-17 in extensions.md).
Joint laws are enumerated exactly where they are finite; logs are base 2.
"""
import random
from itertools import product, combinations
from math import log2, log, pi, e

import numpy as np

res = []
def ck(name, ok, detail=""):
    res.append(bool(ok))
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")

def MI(rows, X, Y):
    """I(X;Y) from rows [(dict, prob)]."""
    pxy, px, py = {}, {}, {}
    for a, p in rows:
        kx, ky = tuple(a[v] for v in X), tuple(a[v] for v in Y)
        pxy[(kx, ky)] = pxy.get((kx, ky), 0) + p
        px[kx] = px.get(kx, 0) + p
        py[ky] = py.get(ky, 0) + p
    return sum(p * log2(p / (px[kx] * py[ky])) for (kx, ky), p in pxy.items() if p > 0)

def H(ps):
    return -sum(p * log2(p) for p in ps if p > 0)

def h2(x):
    return H([x, 1 - x])

def capacity(K, iters=3000):
    """Blahut-Arimoto capacity of a channel matrix K[x][y] (rows sum to 1)."""
    K = np.asarray(K, float); n = K.shape[0]; p = np.full(n, 1 / n)
    for _ in range(iters):
        q = p @ K
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.where(K > 0, K * np.log2(K / q), 0).sum(1)
        p = p * np.exp2(d); p /= p.sum()
    q = p @ K
    with np.errstate(divide='ignore', invalid='ignore'):
        return float((p[:, None] * np.where(K > 0, K * np.log2(K / q), 0)).sum())

# ---------------------------------------------------------------------------
print("Theorem 11  compute-gated channels: a password-locked tool")
# Theta uniform on 4 values (2 bits); a password W uniform on 2^k values.
# The agent tries S distinct passwords; if one is right, the tool reveals Theta.
k = 3
for S in range(0, 2**k + 1):
    rows = []
    for th, w in product(range(4), range(2**k)):
        unlocked = w < S                       # tries 0..S-1 (any order is equivalent)
        tries = tuple(int(w == g) for g in range(S))
        rows.append(({'Th': th, 'R': (tries, th if unlocked else None)}, 1 / (4 * 2**k)))
    got, want = MI(rows, ['Th'], ['R']), min(1, S / 2**k) * 2
    ck(f"S={S} tries: I(Theta;record) = min(1, S/2^k)*H(Theta)", abs(got - want) < 1e-12, f"{got:.4f}")

# ---------------------------------------------------------------------------
print("Theorem 12  worst-case (minimax) Coffee Law on random channels")
rng = np.random.default_rng(0); ok = True; worst_slack = 9.0
for _ in range(300):
    M, nO = int(rng.integers(3, 7)), int(rng.integers(2, 5))
    K = rng.dirichlet(np.ones(nO) * 0.7, size=M)          # one tool call, M hypotheses
    C = capacity(K)
    # Bayes error under the uniform prior (the minimum average error), and the best
    # possible worst-case error over the hypotheses (>= Bayes error)
    bayes_err = 1 - K.max(0).sum() / M
    fano = (log2(M) - min(log2(M), C) - 1) / log2(M - 1)
    ok &= bayes_err >= fano - 1e-9; worst_slack = min(worst_slack, bayes_err - fano)
ck("300 random channels: worst-case error >= Bayes error >= minimax Fano floor", ok,
   f"min slack {worst_slack:.4f}")

# ---------------------------------------------------------------------------
print("Theorem 13  acting is bounded by knowing: a 4-armed bandit")
# World W in {0..3} names the best arm (reward prob 0.9; others 0.5). Each round the
# agent pulls one arm and sees its reward: that pull is the round's tool call.
M, T, hi, lo = 4, 4, 0.9, 0.5
def kernel(arm):          # rows: W, cols: reward 0/1
    return [[1 - (hi if w == arm else lo), hi if w == arm else lo] for w in range(M)]
C = max(capacity(kernel(a)) for a in range(M))
# greedy-posterior policy; exact probability of pulling a wrong arm at round t
p_wrong = [0.0] * T
def recurse(w, t, post, prob):
    if t == T: return
    arm = int(np.argmax(post))
    if arm != w: p_wrong[t] += prob
    for r in (0, 1):
        pr = (hi if w == arm else lo); pr = pr if r else 1 - pr
        lik = np.array([((hi if v == arm else lo) if r else 1 - (hi if v == arm else lo)) for v in range(M)])
        newp = post * lik; newp /= newp.sum()
        recurse(w, t + 1, newp, prob * pr)
for w in range(M):
    recurse(w, 0, np.full(M, 1 / M), 1 / M)
ok = True
for t in range(T):
    budget = min(log2(M), t * C)
    floor = (log2(M) - budget - 1) / log2(M - 1)
    gap = hi - lo
    ok &= gap * p_wrong[t] >= gap * floor - 1e-12
    print(f"      round {t+1}: E[regret] = {gap*p_wrong[t]:.4f} >= gap * Fano floor = {gap*max(0,floor):.4f}")
ck("expected regret per round >= gap x Fano floor from the Coffee Law budget", ok, f"C = {C:.4f} bits/pull")

# ---------------------------------------------------------------------------
print("Theorem 14  communication ceiling: inner product split between two agents")
# Lindsey's lemma (the discrepancy bound behind Theorem 14b), exhaustively for n = 3:
n = 3; N = 2**n
sign = np.array([[(-1) ** bin(x & y).count('1') for y in range(N)] for x in range(N)])
masks = np.array([[(s >> i) & 1 for i in range(N)] for s in range(2**N)])
S = masks @ sign @ masks.T                                  # sum over A x B of (-1)^<x,y>
sizes = masks.sum(1)
bound = np.sqrt(np.outer(sizes, sizes) * N)
ck(f"Lindsey's lemma, all {2**N}x{2**N} rectangles (n=3)", bool((np.abs(S) <= bound + 1e-9).all()))
# One-way protocols for n = 6 with an L = 1 bit message: random and locally optimized
# partitions of Bob's inputs never beat the bound log2(1 + 2^(L + 1 - n/2)).
n, L = 6, 1; N = 2**n
ip = np.array([[bin(x & y).count('1') % 2 for y in range(N)] for x in range(N)])
def info(part):           # I(Theta; x, f(y)) for the message f(y) = part[y]
    p1 = ip.mean()
    cond = 0.0                                   # H(Theta | x, m)
    for x in range(N):
        for m in range(2**L):
            ys = np.where(part == m)[0]
            if len(ys):
                cond += (1 / N) * (len(ys) / N) * h2(ip[x, ys].mean())
    return h2(p1) - cond
rngp = random.Random(3); best = 0.0
for trial in range(40):
    part = np.array([rngp.randrange(2**L) for _ in range(N)])
    cur = info(part)
    for _ in range(300):                       # hill-climb: flip one y's message
        y = rngp.randrange(N); old = part[y]; part[y] = rngp.randrange(2**L)
        new = info(part)
        if new >= cur: cur = new
        else: part[y] = old
    best = max(best, cur)
bnd = log2(1 + 2 ** (L + 1 - n / 2))     # Theorem 14(b): eps = 2^(L - n/2), info <= log2(1 + 2 eps)
ck(f"n={n}, L={L}: best found one-way protocol info {best:.4f} <= bound {bnd:.4f}", best <= bnd + 1e-12)

# ---------------------------------------------------------------------------
print("Theorem 15  changing world: knowledge of a drifting bit is capped")
rho, p = 0.1, 0.1                  # drift flip prob, tool flip prob
eta = (1 - 2 * rho) ** 2           # SDPI constant of the drift (binary symmetric)
C = 1 - h2(p)
Tmax = 9; ok = True; bound_x = 0.0
for t in range(1, Tmax + 1):
    # exact I(Theta_t ; O_0..O_{t-1}) with Theta_0 uniform, observe then drift
    rows = []
    for path in product([0, 1], repeat=t + 1):            # Theta_0..Theta_t
        pp = 0.5
        for a, b in zip(path, path[1:]): pp *= rho if a != b else 1 - rho
        for obs in product([0, 1], repeat=t):              # O_s observes Theta_s
            po = pp
            for s in range(t): po *= p if obs[s] != path[s] else 1 - p
            rows.append(({'Th': path[t], 'O': obs}, po))
    x = MI(rows, ['Th'], ['O'])
    bound_x = eta * (bound_x + C)                           # x_{t} <= eta (x_{t-1} + C)
    ok &= x <= bound_x + 1e-12
    if t in (1, 3, 9): print(f"      t={t}: I = {x:.4f} <= recursion bound {bound_x:.4f}")
cap = eta * C / (1 - eta)
ck(f"drift cap eta*C/(1-eta) = {cap:.4f} holds at every t <= {Tmax}", ok)

# ---------------------------------------------------------------------------
print("Theorem 16  adversaries and provenance")
# Honest agent sees Theta through BSC(0.2) and posts it; an adversary posts anything.
ok_mono = True
for strat in product([0, 1], repeat=4):     # adversary's post: any function of (Theta, honest)
    rows = []
    for th, flip in product([0, 1], [0, 1]):
        hp = th ^ flip; pr = 0.5 * (0.2 if flip else 0.8)
        adv = strat[2 * th + hp]
        rows.append(({'Th': th, 'H': hp, 'A': adv}, pr))
    ok_mono &= MI(rows, ['Th'], ['H', 'A']) >= MI(rows, ['Th'], ['H']) - 1e-12
ck("with provenance, no adversarial post lowers available information (16 strategies)", ok_mono)
rows_p, rows_np = [], []
for th in (0, 1):
    honest, adv = th, 1 - th                 # adversary posts the opposite of the honest post
    rows_p.append(({'Th': th, 'R': (honest, adv)}, 0.5))
    rows_np.append(({'Th': th, 'R': tuple(sorted((honest, adv)))}, 0.5))
ck("with provenance the record carries 1 bit", abs(MI(rows_p, ['Th'], ['R']) - 1) < 1e-12)
ck("without provenance one adversary erases it to 0 bits", abs(MI(rows_np, ['Th'], ['R'])) < 1e-12)

# ---------------------------------------------------------------------------
print("Theorem 17  continuous truth: Gaussian tool")
sigma2 = 1.0; ok_b = ok_eq = True
C = 0.5 * log2(1 + 1 / sigma2)                     # capacity at input power 1
for T in range(1, 11):
    I = 0.5 * log2(1 + T / sigma2)                 # exact I(Theta; O_1..O_T)
    mmse = 1 / (1 + T / sigma2)
    floor = (1 / (2 * pi * e)) * 2 ** (2 * (0.5 * log2(2 * pi * e) - I))
    ok_b &= I <= T * C + 1e-12 and mmse >= (1 / (2 * pi * e)) * 2 ** (2 * (0.5 * log2(2 * pi * e) - T * C)) - 1e-12
    ok_eq &= abs(mmse - floor) < 1e-12
ck("I <= T*C and MSE >= 2^(2(h - budget))/(2 pi e) for T = 1..10", ok_b)
ck("estimation bound is met with equality for the Gaussian", ok_eq)

print()
print("ALL PASSED" if all(res) else "SOME FAILED")
raise SystemExit(0 if all(res) else 1)

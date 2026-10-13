"""
Verification of the LLM-specialization witnesses: miscalibrated updating,
channel compression loss, and the persuasion channel. Beliefs use exact
rational mass; information quantities computed from the definition.
"""
from fractions import Fraction as Fr
from math import log2
from itertools import product

def H(b):  # entropy of a belief dict {val: prob}
    return -sum(float(p)*log2(float(p)) for p in b.values() if p > 0)
def KL(p, q):
    return sum(float(p[x])*log2(float(p[x])/float(q[x])) for x in p if p[x] > 0)
def cross_entropy(p, q):     # E_{x~p}[-log q(x)]
    return -sum(float(p[x])*log2(float(q[x])) for x in p if p[x] > 0)
def logloss(b, truth):       # -log b(truth)
    return -log2(float(b[truth]))
def MAP(b):
    return max(b, key=lambda k: b[k])

def marginal(table, vars):
    out = {}
    for a, p in table:
        k = tuple(a[v] for v in vars); out[k] = out.get(k, Fr(0)) + p
    return out
def MI(table, X, Y):
    pxy, px, py = marginal(table, X+Y), marginal(table, X), marginal(table, Y)
    nx = len(X); s = 0.0
    for k, v in pxy.items():
        s += float(v)*log2(float(v)/(float(px[k[:nx]])*float(py[k[nx:]])))
    return s

res = []
def ck(name, got, want, tol=1e-12):
    ok = abs(got-want) < tol; res.append(ok)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: got {got:.12g}, want {want}")

print("B1  Miscalibration gap = KL(posterior || reported)")
# Bayesian posterior b*, conservative report b_hat = (1-g)*prior + g*b*
bstar = {0: Fr(9,10), 1: Fr(1,10)}
prior = {0: Fr(1,2),  1: Fr(1,2)}
g = Fr(1,2)
bhat = {x: (1-g)*prior[x] + g*bstar[x] for x in bstar}      # = {0:0.7, 1:0.3}
excess = cross_entropy(bstar, bhat) - cross_entropy(bstar, bstar)   # extra log-loss
ck("excess log-loss", excess, KL(bstar, bhat))
ck("gap >= 0", float(KL(bstar,bhat) >= -1e-15), 1.0)
print(f"      (b_hat = {{0:{float(bhat[0])}, 1:{float(bhat[1])}}}, gap = {KL(bstar,bhat):.4f} bits)")

print("B2  Channel compression loss = I(Th;D) - I(Th;M), DPI floor")
# Th=(B1,B2), full data D=(B1,B2), compressed message M=B1 only
t = [({'B1':b1,'B2':b2}, Fr(1,4)) for b1,b2 in product([0,1],[0,1])]
I_D = MI(t, ['B1','B2'], ['B1','B2'])      # receiver with full data
I_M = MI(t, ['B1','B2'], ['B1'])            # receiver sees only B1
ck("I(Th;D)", I_D, 2)
ck("I(Th;M)", I_M, 1)
ck("channel loss", I_D - I_M, 1)
# Theorem B2: E[ KL(b^D || b^M) ] = I(Th;D) - I(Th;M)
def posterior(table, th_vars, cond_vars):
    joint, pc = marginal(table, th_vars + cond_vars), marginal(table, cond_vars)
    n = len(th_vars); post = {}
    for k, v in joint.items():
        post.setdefault(k[n:], {})[k[:n]] = v / pc[k[n:]]
    return post, pc
def expected_KL(table, th_vars, fine, coarse):
    # E over (fine, coarse) of KL(p(Th|fine) || p(Th|coarse)); coarse is a function of fine
    pf, pfm = posterior(table, th_vars, fine)
    pc, _ = posterior(table, th_vars, coarse)
    joint = marginal(table, fine + coarse); nf = len(fine); s = 0.0
    for k, w in joint.items():
        s += float(w) * KL(pf[k[:nf]], pc[k[nf:]])
    return s
tB2 = [({'B1':b1,'B2':b2,'D1':b1,'D2':b2,'M':b1}, Fr(1,4)) for b1,b2 in product([0,1],[0,1])]
ck("E KL(b^D || b^M) = I(Th;D) - I(Th;M)", expected_KL(tB2, ['B1','B2'], ['D1','D2'], ['M']), I_D - I_M)

print("B3  Persuasion: belief flips with ZERO truth-information")
# Theta fair bit, true value 0; persuasive message M is constant 'claim=1' (indep of Theta)
tP = [({'Th':th, 'M':'claim1'}, Fr(1,2)) for th in [0,1]]
ck("I(Th;M)", MI(tP, ['Th'], ['M']), 0)     # message carries no information about truth
prior_r = {0: Fr(6,10), 1: Fr(4,10)}        # receiver starts favoring the TRUTH (0)
truth = 0
def anchor(b, alpha):                        # b_hat = (1-a)*b + a*(point mass on claim=1)
    return {0: (1-alpha)*b[0], 1: (1-alpha)*b[1] + alpha}
print(f"      prior MAP = {MAP(prior_r)} (correct), prior log-loss = {logloss(prior_r,truth):.3f}")
for a in [Fr(1,10), Fr(2,10), Fr(5,10)]:
    bh = anchor(prior_r, a); flipped = (MAP(bh) != truth)
    print(f"      alpha={float(a)}: b_hat={{0:{float(bh[0]):.2f},1:{float(bh[1]):.2f}}} "
          f"MAP={MAP(bh)} {'WRONG' if flipped else 'ok'} logloss={logloss(bh,truth):.3f}")
# threshold: flip when (1-a)*0.6 < (1-a)*0.4 + a  <=>  a > 1/6
ck("flip at alpha=0.2 (>1/6)", float(MAP(anchor(prior_r,Fr(2,10))) != truth), 1.0)
ck("no flip at alpha=0.1 (<1/6)", float(MAP(anchor(prior_r,Fr(1,10))) != truth), 0.0)

import random
c = 0.5 * 0.6931471805599453   # ln 2 / 2
def TV(p, q):
    return 0.5 * sum(abs(float(p.get(x, 0)) - float(q.get(x, 0))) for x in set(p) | set(q))

print("B0  Plug-in regret: r <= 2U TV(b*, b_hat) + delta <= 2U sqrt(c KL) + delta")
rng = random.Random(0); worst = -1.0; ok0 = True
for _ in range(2000):
    K, A = rng.randint(2, 5), rng.randint(2, 4)
    def rand_belief():
        w = [rng.random() + 1e-3 for _ in range(K)]; z = sum(w)
        return {k: w[k] / z for k in range(K)}
    bs, bh = rand_belief(), rand_belief()
    u = [[rng.random() for _ in range(K)] for _ in range(A)]     # U = 1
    Eb = lambda b, a: sum(b[k] * u[a][k] for k in range(K))
    best_bh = max(Eb(bh, a) for a in range(A))
    a_hat = rng.randrange(A)                                       # any action...
    delta = best_bh - Eb(bh, a_hat)                                # ...with its own delta
    r = max(Eb(bs, a) for a in range(A)) - Eb(bs, a_hat)
    tv, kl = TV(bs, bh), KL(bs, bh)
    ok0 &= r <= 2*tv + delta + 1e-12 and 2*tv <= 2*(c*kl)**0.5 + 1e-12
    worst = max(worst, r - (2*tv + delta))
res.append(ok0)
print(f"  [{'OK ' if ok0 else 'FAIL'}] 2000 random cases; max of r - (2TV + delta) = {worst:.4f} (must be <= 0)")

print("B4  Per-round decomposition (2-bit truth, noisy data, 1-bit message, persuaded receiver)")
# Theta = (B1,B2) uniform; data D = each bit flipped w.p. 1/10; message M = D1;
# receiver anchors b^M toward the constant claim (1,1) with alpha = 1/4, then plays MAP.
pf, alpha, claim = Fr(1,10), Fr(1,4), (1,1)
t4 = []
for b1, b2, n1, n2 in product([0,1],[0,1],[0,1],[0,1]):
    pr = Fr(1,4) * (pf if n1 else 1-pf) * (pf if n2 else 1-pf)
    t4.append(({'B1':b1,'B2':b2,'D1':b1^n1,'D2':b2^n2,'M':b1^n1}, pr))
postD, _ = posterior(t4, ['B1','B2'], ['D1','D2'])
postM, _ = posterior(t4, ['B1','B2'], ['M'])
states = list(product([0,1],[0,1]))
def full(b): return {s_: b.get(s_, Fr(0)) for s_ in states}
def bhat_of(m):
    bm = full(postM[(m,)])
    return {s_: (1-alpha)*bm[s_] + (alpha if s_ == claim else 0) for s_ in states}
regret = term1 = term3 = 0.0
for a, pr in t4:
    th, d, m = (a['B1'], a['B2']), (a['D1'], a['D2']), a['M']
    bh = bhat_of(m); a_hat = MAP(bh)
    regret += float(pr) * (1.0 - (1.0 if a_hat == th else 0.0))   # u = 1[a = theta], U = 1, delta = 0
    term1 += float(pr) * (1.0 - float(full(postD[d])[th]))
    term3 += float(pr) * (c * KL(full(postM[(m,)]), bh))**0.5
I_D4 = MI(t4, ['B1','B2'], ['D1','D2']); I_M4 = MI(t4, ['B1','B2'], ['M'])
term2 = (c * (I_D4 - I_M4))**0.5
ck("B2 identity on noisy data", expected_KL(t4, ['B1','B2'], ['D1','D2'], ['M']), I_D4 - I_M4)
bound = 2 * (term1 + term2 + term3)
print(f"      E[regret] = {regret:.4f}; terms: lack-of-info {term1:.4f}, channel {term2:.4f}, "
      f"miscalibration {term3:.4f}; bound = {bound:.4f}")
ok4 = regret <= bound + 1e-12; res.append(ok4)
print(f"  [{'OK ' if ok4 else 'FAIL'}] E[regret] <= bound")

print()
print("ALL PASSED" if all(res) else "SOME FAILED")
raise SystemExit(0 if all(res) else 1)

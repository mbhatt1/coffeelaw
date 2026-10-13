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

print()
print("ALL PASSED" if all(res) else "SOME FAILED")

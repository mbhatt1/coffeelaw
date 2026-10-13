"""
Numerical checks of the geometry lemmas in proof.md (G1-G5).
Logs are base 2 throughout, matching the document.
"""
import random
from math import log, log2

LN2 = log(2)

def H(b):
    return -sum(p * log2(p) for p in b if p > 0)
def KL(p, q):
    return sum(pi * log2(pi / qi) for pi, qi in zip(p, q) if pi > 0)
def F(b):
    return -H(b)
def gradF(b):
    return [log2(p) + 1 / LN2 for p in b]
def bregman(p, q):
    return F(p) - F(q) - sum(g * (pi - qi) for g, pi, qi in zip(gradF(q), p, q))

res = []
def ck(name, ok, detail=""):
    res.append(ok)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")

rng = random.Random(0)
def rand_belief(K):
    w = [rng.random() + 1e-3 for _ in range(K)]
    z = sum(w)
    return [x / z for x in w]

print("G1  Bregman divergence of F = -H equals KL (bits)")
worst = max(abs(bregman(p, q) - KL(p, q))
            for p, q in ((rand_belief(k), rand_belief(k)) for k in [2, 3, 5, 8] * 250))
ck("1000 random pairs", worst < 1e-12, f"max |D_F - KL| = {worst:.2e}")

print("G2/G3  I(Th;M) = E KL(b^M||b) = H(b) - E H(b^M) = VoI under log loss")
# Theta in {0,1,2}, M in {a,b}: the joint from verify.py Scenario 6
joint = {(0, 'a'): 2/12, (0, 'b'): 1/12, (1, 'a'): 1/12, (1, 'b'): 3/12,
         (2, 'a'): 4/12, (2, 'b'): 1/12}
pM = {m: sum(joint[(t, m)] for t in range(3)) for m in 'ab'}
b = [sum(joint[(t, m)] for m in 'ab') for t in range(3)]
post = {m: [joint[(t, m)] / pM[m] for t in range(3)] for m in 'ab'}
I = sum(v * log2(v / (b[t] * pM[m])) for (t, m), v in joint.items())
EKL = sum(pM[m] * KL(post[m], b) for m in 'ab')
gap = H(b) - sum(pM[m] * H(post[m]) for m in 'ab')
V = lambda bb: max(sum(bt * log2(qt) for bt, qt in zip(bb, q))
                   for q in [bb] + [rand_belief(3) for _ in range(200)])  # max at q = b
VoI = sum(pM[m] * V(post[m]) for m in 'ab') - V(b)
ck("I = E KL(b^M||b)", abs(I - EKL) < 1e-12, f"{I:.6f} vs {EKL:.6f}")
ck("I = H(b) - E H(b^M)", abs(I - gap) < 1e-12, f"{gap:.6f}")
ck("I = VoI under log loss", abs(I - VoI) < 1e-12, f"{VoI:.6f}")

print("G3  Fisher constant: KL(b+t*d || b) / t^2 -> (1/(2 ln 2)) sum d^2/b")
b0 = [0.5, 0.3, 0.2]
d = [0.1, -0.04, -0.06]
target = sum(x * x / p for x, p in zip(d, b0)) / (2 * LN2)
t = 1e-4
ratio = KL([p + t * x for p, x in zip(b0, d)], b0) / t**2
ck("bits constant 1/(2 ln 2)", abs(ratio - target) / target < 1e-3,
   f"ratio {ratio:.6f}, target {target:.6f}")

print("G4  Evidence drift: E_o[L(b) - L(b+)] = KL(p(.|th*) || predictive) >= 0")
ok = True; worst_gap = 0.0
for _ in range(500):
    K, O = rng.randint(2, 5), rng.randint(2, 4)
    bb = rand_belief(K)
    lik = [rand_belief(O) for _ in range(K)]        # p(o | theta_k)
    ts = rng.randrange(K)                            # true theta*
    pred = [sum(bb[k] * lik[k][o] for k in range(K)) for o in range(O)]
    drift = 0.0
    for o in range(O):
        bplus_ts = bb[ts] * lik[ts][o] / pred[o]
        drift += lik[ts][o] * (-log2(bb[ts]) + log2(bplus_ts))
    want = KL(lik[ts], pred)
    worst_gap = max(worst_gap, abs(drift - want))
    ok &= drift >= -1e-12
ck("500 random models: identity", worst_gap < 1e-12, f"max error {worst_gap:.2e}")
ck("500 random models: drift >= 0", ok)

print("G5  Linear pooling")
ok1 = ok2 = True
for _ in range(500):
    K, n = rng.randint(2, 5), rng.randint(2, 5)
    bs = [rand_belief(K) for _ in range(n)]
    w = rand_belief(n)
    pooled = [sum(w[j] * bs[j][k] for j in range(n)) for k in range(K)]
    ts = rng.randrange(K)
    ok1 &= -log2(pooled[ts]) <= sum(w[j] * -log2(bs[j][ts]) for j in range(n)) + 1e-12
    ok2 &= H(pooled) >= sum(w[j] * H(bs[j]) for j in range(n)) - 1e-12
ck("(1) L(pooled) <= sum_j W_ij L(b_j)", ok1)
ck("(2) H(pooled) >= sum_j W_ij H(b_j)", ok2)
b1, b2 = [0.9, 0.1], [0.1, 0.9]
pooled = [0.5 * x + 0.5 * y for x, y in zip(b1, b2)]
L_before, L_after = -log2(b1[0]), -log2(pooled[0])
ck("(3) pooling can hurt the better agent", L_after > L_before,
   f"L rises from {L_before:.3f} to {L_after:.3f} bits")

print()
print("ALL PASSED" if all(res) else "SOME FAILED")
raise SystemExit(0 if all(res) else 1)

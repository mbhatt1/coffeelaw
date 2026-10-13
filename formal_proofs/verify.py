"""
Machine verification of every finite witness in the formal proofs.
Each scenario is specified as a FULL joint distribution with exact rational
probabilities (fractions.Fraction); all mutual informations are computed from
the DEFINITION, then asserted. No value is hard-coded into the estimator.
"""
from fractions import Fraction as Fr
from math import log2
from itertools import product

Row = dict  # assignment: var-name -> value

def marginal(table, vars):
    """P over the given ordered list of variables, as {value-tuple: Fraction}."""
    out = {}
    for assign, p in table:
        key = tuple(assign[v] for v in vars)
        out[key] = out.get(key, Fr(0)) + p
    return out

def entropy(table, vars):
    m = marginal(table, vars)
    return -sum(float(p) * log2(float(p)) for p in m.values() if p > 0)

def MI(table, X, Y):
    pxy, px, py = marginal(table, X + Y), marginal(table, X), marginal(table, Y)
    s = 0.0
    nx = len(X)
    for k, v in pxy.items():
        xk, yk = k[:nx], k[nx:]
        s += float(v) * log2(float(v) / (float(px[xk]) * float(py[yk])))
    return s

def CMI(table, X, Y, Z):
    pxyz = marginal(table, X + Y + Z)
    pxz, pyz, pz = marginal(table, X + Z), marginal(table, Y + Z), marginal(table, Z)
    nx, ny = len(X), len(Y)
    s = 0.0
    for k, v in pxyz.items():
        xk, yk, zk = k[:nx], k[nx:nx+ny], k[nx+ny:]
        num = float(v) * float(pz[zk])
        den = float(pxz[xk+zk]) * float(pyz[yk+zk])
        s += float(v) * log2(num / den)
    return s

def close(a, b, tol=1e-12):
    return abs(a - b) < tol

def build(assign_iter):
    return list(assign_iter)

results = []
def check(name, got, want):
    ok = close(got, want)
    results.append(ok)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: got {got:.12g}, want {want}")

print("Scenario 1  XOR synergy:  Theta = B1 xor B2,  Y1=B1, Y2=B2")
t = []
for b1, b2 in product([0,1],[0,1]):
    t.append(({'B1':b1,'B2':b2,'Th':b1^b2,'Y1':b1,'Y2':b2}, Fr(1,4)))
check("I(Th;Y1)", MI(t,['Th'],['Y1']), 0)
check("I(Th;Y2)", MI(t,['Th'],['Y2']), 0)
check("I(Th;Y1,Y2)", MI(t,['Th'],['Y1','Y2']), 1)
check("net synergy sigma", MI(t,['Th'],['Y1','Y2'])-MI(t,['Th'],['Y1'])-MI(t,['Th'],['Y2']), 1)

print("Scenario 2  additive two bits:  Theta=(B1,B2),  Y1=B1, Y2=B2")
t = []
for b1, b2 in product([0,1],[0,1]):
    t.append(({'B1':b1,'B2':b2,'Y1':b1,'Y2':b2}, Fr(1,4)))
# Theta is the pair (B1,B2); use both as the X block
check("I(Th;Y1)", MI(t,['B1','B2'],['Y1']), 1)
check("I(Th;Y2)", MI(t,['B1','B2'],['Y2']), 1)
check("I(Th;Y1,Y2)", MI(t,['B1','B2'],['Y1','Y2']), 2)
check("net synergy sigma", MI(t,['B1','B2'],['Y1','Y2'])-MI(t,['B1','B2'],['Y1'])-MI(t,['B1','B2'],['Y2']), 0)

print("Scenario 3  routing:  K in {1,2}, V in {0,1} indep; tool O = V iff q==K else bot")
# isolated: q uniform indep of K,V
t = []
for k,v,q in product([1,2],[0,1],[1,2]):
    o = v if q==k else 'bot'
    t.append(({'K':k,'V':v,'Q':q,'O':o}, Fr(1,2)*Fr(1,2)*Fr(1,2)))
check("isolated I(V;O|Q)", CMI(t,['V'],['O'],['Q']), 0.5)
# pooled: q = K deterministically
t = []
for k,v in product([1,2],[0,1]):
    o = v  # q==k always
    t.append(({'K':k,'V':v,'Q':k,'O':o}, Fr(1,2)*Fr(1,2)))
check("pooled I(V;O)", MI(t,['V'],['O']), 1)
print("  routing gain = 1 - 0.5 = 0.5 bit")
# Theorem 5: the gain is the VoI of K in the decision problem with
# action q, state k, utility u(q,k) = I(V;O | Q=q, K=k).
def u(q, k):
    tt = [({'V':v,'O':(v if q==k else 'bot')}, Fr(1,2)) for v in [0,1]]
    return MI(tt, ['V'], ['O'])
V_prior = max(sum(0.5*u(q,k) for k in [1,2]) for q in [1,2])
V_post = sum(0.5*max(u(q,k) for q in [1,2]) for k in [1,2])
check("V_u(prior) = best blind yield", V_prior, 0.5)
check("E_K max_q u(q,K) = routed yield", V_post, 1)
check("VoI(K) = routing gain", V_post - V_prior, 0.5)

print("Scenario 4  copy coupling vs independence")
t = [({'A1':r,'A2':r}, Fr(1,2)) for r in [0,1]]                         # A2 copies A1=R
check("copy I(A1;A2)", MI(t,['A1'],['A2']), 1)
t = [({'A1':a,'A2':b}, Fr(1,4)) for a,b in product([0,1],[0,1])]        # independent
check("indep I(A1;A2)", MI(t,['A1'],['A2']), 0)

print("Scenario 5  consensus is not knowledge:  Theta, R indep fair bits; all copy R")
t = [({'Th':th,'R':r}, Fr(1,4)) for th,r in product([0,1],[0,1])]
check("I(Th;R)", MI(t,['Th'],['R']), 0)
acc = sum(p for a,p in t if a['Th']==a['R'])
check("P(R=Theta)", float(acc), 0.5)

print("Scenario 6  posterior-mean martingale:  E_M[ p(theta|M) ] = p(theta)")
# arbitrary coupling: Theta in {0,1,2}, signal M in {a,b}
joint = {(0,'a'):Fr(2,12),(0,'b'):Fr(1,12),
         (1,'a'):Fr(1,12),(1,'b'):Fr(3,12),
         (2,'a'):Fr(4,12),(2,'b'):Fr(1,12)}
pM = {m: sum(joint[(th,m)] for th in [0,1,2]) for m in ['a','b']}
pTh = {th: sum(joint[(th,m)] for m in ['a','b']) for th in [0,1,2]}
for th in [0,1,2]:
    em = sum(pM[m]*(joint[(th,m)]/pM[m]) for m in ['a','b'])  # E_M[p(th|M)]
    ok = (em == pTh[th]); results.append(ok)
    print(f"  [{'OK ' if ok else 'FAIL'}] E_M[p({th}|M)] = {em} = p({th}) = {pTh[th]}")

print("Scenario 7  capacity budget: adaptive agents + noisy tool (Theorem 7)")
# Theta uniform on {0,1,2,3}; D0 empty, so I(Th;D0)=0. Each round the collective
# issues one call q in {0,1}; the tool returns bit q of Theta through a BSC(p).
# Round-2 query adapts to round-1 output and to fresh noise; a 'debate' variable Z
# (function of the transcript plus independent noise) is appended at the end.
def h2(x):
    return 0.0 if x in (0, 1) else -(x*log2(x) + (1-x)*log2(1-x))
def scenario7(p, adaptive=True):
    p = Fr(p); t = []
    for th, x1, x2, n1, n2, z in product(range(4), [0,1], [0,1], [0,1], [0,1], [0,1]):
        pr = Fr(1,4) * Fr(1,8)                        # Theta, xi1, xi2, zeta: uniform
        pr *= (p if n1 else 1-p) * (p if n2 else 1-p)  # tool noise
        q1 = x1                                        # round 1: random query
        o1 = ((th >> q1) & 1) ^ n1
        q2 = ((1 - q1) if o1 == 0 else x2) if adaptive else 1 - q1  # round 2 query
        o2 = ((th >> q2) & 1) ^ n2
        Z = (o1 + o2 + z) % 2                          # 'debate' post: computation + noise
        if pr == 0: continue
        t.append(({'Th':th,'X1':x1,'Q1':q1,'O1':o1,'X2':x2,'Q2':q2,'O2':o2,'Z':Z}, pr))
    return t
for p, adaptive, label in [(Fr(1,10), True, "noisy adaptive"), (Fr(1,10), False, "noisy other-bit"),
                           (Fr(0), False, "noiseless other-bit")]:
    t = scenario7(p, adaptive); C = 1 - h2(float(p))
    D1 = ['X1','Q1','O1']; D2 = D1 + ['X2','Q2','O2']; DT = D2 + ['Z']
    I1, I2, IT = MI(t,['Th'],D1), MI(t,['Th'],D2), MI(t,['Th'],DT)
    print(f"  [{label}] C = {C:.6f}, I(Th;D_1) = {I1:.6f}, I(Th;D_2) = {I2:.6f}, I(Th;D_T) = {IT:.6f}")
    check(f"{label}: round-1 gain = I(Th;O1|X1,Q1)", I1, CMI(t,['Th'],['O1'],['X1','Q1']))
    check(f"{label}: round-2 gain = I(Th;O2|D1,X2,Q2)", I2 - I1, CMI(t,['Th'],['O2'],D1+['X2','Q2']))
    check(f"{label}: debate adds nothing", IT, I2)
    for name, gain in [("round-1", I1), ("round-2", I2 - I1)]:
        ok = gain <= C + 1e-12; results.append(ok)
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}: {name} gain {gain:.6f} <= C {C:.6f}")
    ok = IT <= min(2.0, 2*C) + 1e-12; results.append(ok)
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}: budget I(Th;D_T) {IT:.6f} <= min(H, 2C) = {min(2.0, 2*C):.6f}")
    # Fano: MAP error from the full transcript is at least (H(Th) - I - 1)/log2(|Th|-1)
    post = marginal(t, ['Th'] + DT)
    best = {}
    for k, v in post.items():
        best[k[1:]] = max(best.get(k[1:], Fr(0)), v)
    pe = 1 - float(sum(best.values()))
    fano = (2 - IT - 1) / log2(3)
    ok = pe >= fano - 1e-12; results.append(ok)
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}: MAP error {pe:.6f} >= Fano floor {fano:.6f}")
check("noiseless other-bit attains the budget: I(Th;D_T) = H(Th)", MI(scenario7(0, False),['Th'],['X1','Q1','O1','X2','Q2','O2','Z']), 2)

print()
print("ALL PASSED" if all(results) else "SOME FAILED")

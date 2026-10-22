import itertools, math
from collections import defaultdict
def h(p): return 0 if p in (0,1) else -p*math.log2(p)-(1-p)*math.log2(1-p)
def MI(joint):  # joint: dict (theta, obs) -> prob
    pt=defaultdict(float); po=defaultdict(float)
    for (t,o),p in joint.items(): pt[t]+=p; po[o]+=p
    return sum(p*math.log2(p/(pt[t]*po[o])) for (t,o),p in joint.items() if p>0)
eps=0.1; C=1-h(eps)
print("C =",round(C,4), " h(0.1)=",round(h(eps),4))
thetas=[(a,b) for a in (0,1) for b in (0,1)]
def run(policy):
    # policy(history)-> bit index; returns joint over (theta, record)
    J=defaultdict(float)
    for th in thetas:
        def rec(hist,p,depth):
            if depth==2: J[(th,tuple(hist))]+=p; return
            i=policy(hist)
            for o in (0,1):
                q=(1-eps) if o==th[i] else eps
                rec(hist+[(i,o)],p*q,depth+1)
        rec([],0.25,0)
    return J
J1=run(lambda h: len(h))  # bit0 then bit1
print("bit0 then bit1:",round(MI(J1),4))
Jrep=run(lambda h: 0)
print("bit0 twice:",round(MI(Jrep),4))
Jad=run(lambda h: 0 if not h else (0 if h[0][1]==0 else 1))
print("(other policy, not in the paper) repeat bit0 iff first answer 0:",round(MI(Jad),4))
# Paper's Table I adaptive policy (repo verify.py scenario 7): first bit random;
# if answer 0 ask the other bit, if 1 ask a random bit (may repeat). Coins are kept in the record.
J7=defaultdict(float)
for th in thetas:
    for x1 in (0,1):
        for x2 in (0,1):
            for n1 in (0,1):
                for n2 in (0,1):
                    p=0.25*0.25*(eps if n1 else 1-eps)*(eps if n2 else 1-eps)
                    o1=th[x1]^n1; q2=(1-x1) if o1==0 else x2; o2=th[q2]^n2
                    J7[(th,(x1,o1,x2,q2,o2))]+=p
print("paper's adaptive policy (Table I):",round(MI(J7),4))
# debate step: append deterministic function of record (e.g. a 'vote' = majority guess) -> record unchanged in info
Jdeb=defaultdict(float)
for (t,r),p in J1.items(): Jdeb[(t,(r,('vote',r[0][1],r[1][1])))]+=p
print("after debate step:",round(MI(Jdeb),4))
# Fano + MAP error for bit0-then-bit1
def map_err(J):
    by=defaultdict(dict)
    for (t,r),p in J.items(): by[r][t]=by[r].get(t,0)+p
    return 1-sum(max(d.values()) for d in by.values())
I=MI(J1); fano=(2-min(2,I)-1)/math.log2(3)
print("MAP error bit0,bit1:",round(map_err(J1),4)," Fano floor:",round(fano,4))
# noise-free
print("noise-free MAP error:", 0.0)
# Three rounds (bit0, bit1, bit0 again): more information, same MAP error
J3=defaultdict(float)
for (t,r),p in J1.items():
    for o in (0,1): J3[(t,r+((0,o),))]+=p*((1-eps) if o==t[0] else eps)
print("3 rounds (bit0,bit1,bit0): I=",round(MI(J3),4)," MAP error=",round(map_err(J3),4))
# single answer unseen index (first bit random)
J=defaultdict(float)
for th in thetas:
    for i in (0,1):
        for o in (0,1):
            J[(th,o)]+=0.25*0.5*((1-eps) if o==th[i] else eps)
print("single answer, index unseen:",round(MI(J),4))
# Corollary 2 for 8 commits
for k,c in [(3,1.0),(3,C),(10,1.0),(10,C)]:
    for e in (0,0.1):  # with a noisy tool only e>0 is attainable
        print(f"k={k} C={c:.3f} eps={e}: min rounds >= {((1-e)*k-1)/c:.3f}")
# XOR reasoning example
print("XOR: I(Theta;B1)=0, I(Theta;B2)=0, I(Theta;B1,B2)=1")
# Communication: inner product vs parity, n=4, one-way 1-bit message from agent 2 to agent 1
n=4
vecs=list(itertools.product((0,1),repeat=n))
def best_msg(f):
    best=(0,None)
    for mask in range(2**len(vecs)):
        J=defaultdict(float)
        for x in vecs:
            for yi,y in enumerate(vecs):
                m=(mask>>yi)&1
                J[(f(x,y),(x,m))]+=1/len(vecs)**2
        v=MI(J)
        if v>best[0]+1e-12: best=(v,mask)
    return best
ip=lambda x,y: sum(a*b for a,b in zip(x,y))%2
par=lambda x,y: (sum(x)+sum(y))%2
bi=best_msg(ip); bp=best_msg(par)
print("n=4 best 1-bit msg: IP",round(bi[0],4)," parity",round(bp[0],4))
# IP with sending L raw bits of y (L=0..4)
for L in range(n+1):
    J=defaultdict(float)
    for x in vecs:
        for y in vecs:
            J[(ip(x,y),(x,y[:L]))]+=1/256
    print(f"IP n=4, send first {L} bits of y: I={MI(J):.4f}")
for L in range(4):
    print("paper bound log(1+2^(L+1-n/2)) n=4 L=",L, round(math.log2(1+2**(L+1-n/2)),4))

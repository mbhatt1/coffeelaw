"""
Checks for Theorem 20 in extensions.md: tool capacity from the output format.

A tool's capacity is at most log2 of the number of results it can return,
counting everything that enters the record (payload, status or error code,
truncation, and latency if the agents can read it). This script

  1. counts result spaces exactly (Python integers) for byte strings, token
     strings and JSON-style schemas, and checks the closed-form bit bounds;
  2. checks the entropy bounds on random channels with Blahut-Arimoto:
     Cap <= log2|O|, Cap <= log2|O| - min_theta H(K(.|theta)), joint calls are
     subadditive, truncation can only lower the bound;
  3. checks that the bound is attained (bisection with pass/fail tests);
  4. prints the bound for a few concrete tool formats.

Counts are exact; logarithms of big integers are taken with int.bit_length so
nothing overflows.
"""
import random
from fractions import Fraction as Fr
from itertools import product
from math import comb, log2, perm

res = []
def ck(name, ok, extra=""):
    res.append(bool(ok))
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f": {extra}" if extra else ""))

def lg(n):
    """log2 of a positive integer of any size."""
    n = int(n)
    if n < 2 ** 1000:
        return log2(n)
    s = n.bit_length() - 64
    return s + log2(n >> s)

# ---------------------------------------------------------------------------
# Counting result spaces. A schema is a nested tuple:
#   ("const",)              one value
#   ("bool",)               2 values
#   ("enum", k)             k values
#   ("int", lo, hi)         hi - lo + 1 values
#   ("float64",)            2^64 bit patterns
#   ("bytes", L)            byte strings of length <= L
#   ("bytes_exact", L)      byte strings of length exactly L
#   ("tokens", V, L)        token strings of length <= L over a vocabulary of V
#   ("nullable", T)         T or null
#   ("object", [T1, ...])   all fields present: product
#   ("union", [T1, ...])    tagged alternatives: sum
#   ("array", T, n)         lists of 0..n items of T
#   ("set", N, k)           a set of exactly k distinct items out of N
#   ("ranked", N, k)        an ordered list of k distinct items out of N
def strings_up_to(alpha, L):
    """Number of strings of length 0..L over an alphabet of size alpha."""
    return sum(alpha ** l for l in range(L + 1))

def count(s):
    tag = s[0]
    if tag == "const":       return 1
    if tag == "bool":        return 2
    if tag == "enum":        return s[1]
    if tag == "int":         return s[2] - s[1] + 1
    if tag == "float64":     return 2 ** 64
    if tag == "bytes":       return strings_up_to(256, s[1])
    if tag == "bytes_exact": return 256 ** s[1]
    if tag == "tokens":      return strings_up_to(s[1], s[2])
    if tag == "nullable":    return count(s[1]) + 1
    if tag == "object":
        n = 1
        for f in s[1]:
            n *= count(f)
        return n
    if tag == "union":       return sum(count(f) for f in s[1])
    if tag == "array":       return strings_up_to(count(s[1]), s[2])
    if tag == "set":         return comb(s[1], s[2])
    if tag == "ranked":      return perm(s[1], s[2])
    raise ValueError(tag)

def bits(s):
    return lg(count(s))

print("Part 1  counting result spaces")
# Geometric-sum formula against brute force on small alphabets.
ok = all(strings_up_to(a, L) == sum(1 for l in range(L + 1) for _ in product(range(a), repeat=l))
         for a in (1, 2, 3, 4) for L in range(5))
ck("strings of length <= L: count matches brute-force enumeration", ok)
ok = all(strings_up_to(256, L) * 255 == 256 ** (L + 1) - 1 for L in range(200))
ck("<= L bytes: |O| = (256^(L+1) - 1) / 255, L = 0..199", ok)
# Exact integer form of  log2|O| < 8L + log2(256/255):  255 |O| < 256^(L+1).
ok = all(255 * count(("bytes", L)) < 256 ** (L + 1) for L in range(200))
ck("<= L bytes: log2|O| < 8L + log2(256/255) = 8L + 0.00565", ok)
# Tokens: (V - 1) |O| < V^(L+1)  <=>  log2|O| < L log2 V + log2(V / (V - 1)).
ok = all((V - 1) * count(("tokens", V, L)) < V ** (L + 1)
         for V in (2, 50, 1000, 50257, 100277, 200019) for L in (0, 1, 7, 64, 512))
ck("<= L tokens over V: log2|O| < L log2 V + log2(V/(V-1))", ok)
# Products add, sums are at most the max plus log2 of the number of alternatives.
fields = [("enum", 5), ("int", -90, 60), ("bool",), ("bytes", 3)]
ck("object: bits add over fields",
   abs(bits(("object", fields)) - sum(bits(f) for f in fields)) < 1e-9)
alts = [("enum", 7), ("int", 0, 999), ("const",)]
ck("union: bits <= max + log2(#alternatives)",
   bits(("union", alts)) <= max(bits(a) for a in alts) + log2(len(alts)) + 1e-12)
ck("nullable adds less than one bit to anything with >= 1 value",
   all(bits(("nullable", ("enum", k))) - bits(("enum", k)) <= 1 for k in range(1, 50)))

# ---------------------------------------------------------------------------
# Channels. Pure-Python Blahut-Arimoto (capacity in bits) and entropies.
def H(p):
    return -sum(x * log2(x) for x in p if x > 0)

def mi(px, K):
    """I(Theta; O) for input law px and channel rows K[theta][o]."""
    m = len(K[0])
    po = [sum(px[t] * K[t][o] for t in range(len(px))) for o in range(m)]
    return H(po) - sum(px[t] * H(K[t]) for t in range(len(px)))

def capacity(K, iters=3000):
    n, m = len(K), len(K[0])
    p = [1.0 / n] * n
    for _ in range(iters):
        q = [sum(p[t] * K[t][o] for t in range(n)) for o in range(m)]
        c = []
        for t in range(n):
            d = sum(K[t][o] * log2(K[t][o] / q[o]) for o in range(m) if K[t][o] > 0)
            c.append(p[t] * 2 ** d)
        z = sum(c)
        p = [x / z for x in c]
    return mi(p, K), p

def rand_channel(n, m, rng, sparse=False):
    K = []
    for _ in range(n):
        row = [rng.random() ** (4 if sparse else 1) for _ in range(m)]
        z = sum(row)
        K.append([x / z for x in row])
    return K

print("Part 2  entropy bounds on random channels")
rng = random.Random(20)
ok1 = ok2 = ok3 = ok4 = True
worst = 0.0
for trial in range(150):
    n, m = rng.randint(2, 6), rng.randint(2, 6)
    K = rand_channel(n, m, rng, sparse=trial % 2 == 0)
    C, p = capacity(K, iters=400)
    ok1 &= C <= log2(m) + 1e-9
    ok2 &= C <= log2(m) - min(H(row) for row in K) + 1e-9
    ok3 &= C <= log2(n) + 1e-9
    # Any input law, in particular the capacity-achieving one, stays under log2 m.
    ok4 &= all(mi(px, K) <= log2(m) + 1e-12
               for px in ([1.0 / n] * n, p, [1.0] + [0.0] * (n - 1)))
    worst = max(worst, C - log2(m))
ck("Cap(K) <= log2|O|  (150 random channels)", ok1, f"max Cap - log2|O| = {worst:.4f}")
ck("Cap(K) <= log2|O| - min_theta H(K(.|theta))  (noise floor)", ok2)
ck("Cap(K) <= log2|supp Theta|", ok3)
ck("I(Theta; O) <= log2|O| for every input law tried", ok4)

# Two calls in one round with CORRELATED results (one joint channel on O1 x O2):
# the bound log2|O1| + log2|O2| needs no independence.
ok = True
for _ in range(60):
    n, m1, m2 = rng.randint(2, 5), rng.randint(2, 4), rng.randint(2, 4)
    K = rand_channel(n, m1 * m2, rng, sparse=True)
    C, _ = capacity(K, iters=400)
    ok &= C <= log2(m1) + log2(m2) + 1e-9
ck("two correlated calls: Cap <= log2|O1| + log2|O2|", ok)

# Harness truncation: the record holds f(O), not O. Then the bound is log2|f(O)|
# and the information can only drop (data processing).
ok = True
for _ in range(60):
    n, m, k = rng.randint(2, 5), rng.randint(3, 8), 2
    K = rand_channel(n, m, rng)
    f = [rng.randrange(k) for _ in range(m)]          # keep k of m values
    Kf = [[sum(row[o] for o in range(m) if f[o] == j) for j in range(k)] for row in K]
    px = [rng.random() for _ in range(n)]
    z = sum(px); px = [x / z for x in px]
    ok &= mi(px, Kf) <= min(log2(k), mi(px, K)) + 1e-12
ck("truncation f(O): I(Theta; f(O)) <= min(log2|f(O)|, I(Theta; O))", ok)

# A side channel: the payload is constant but latency depends on Theta. Leaving
# latency out of O gives bound 0 and is wrong; counting it restores the bound.
K_payload = [[1.0], [1.0]]
K_full = [[0.9, 0.1], [0.1, 0.9]]                     # (payload, latency bucket)
C_full, _ = capacity(K_full)
ck("latency side channel: payload alone has capacity 0 but the call leaks bits",
   capacity(K_payload)[0] < 1e-12 and C_full > 0.5, f"Cap with latency = {C_full:.4f}")
ck("counting latency in O restores the bound: Cap <= log2(1 * 2)", C_full <= 1 + 1e-12)

# ---------------------------------------------------------------------------
print("Part 3  the format bound is attained")
# Theta uniform on 2^k values; each call is a pass/fail test "is bit j of Theta 1?".
# k tests reveal Theta exactly: I = k = sum of log2|O| over the calls.
for k in (1, 3, 5):
    rows = []
    for th in range(2 ** k):
        rows.append((th, tuple((th >> j) & 1 for j in range(k)), Fr(1, 2 ** k)))
    pt = {}
    for _, o, p in rows:
        pt[o] = pt.get(o, Fr(0)) + p
    I = H([float(p) for p in pt.values()])          # Theta determines O, so I = H(O)
    ck(f"bisection with {k} pass/fail tests: I = sum log2|O| = {k}",
       abs(I - k) < 1e-12 and abs(I - k * bits(("bool",))) < 1e-12)

# ---------------------------------------------------------------------------
print("Part 4  bounds for concrete tool formats")
FORMATS = [
    ("unit test (pass/fail)", ("bool",)),
    ("test runner: pass/fail + exit code", ("object", [("bool",), ("int", 0, 255)])),
    ("HTTP status only", ("int", 100, 599)),
    ("sensor reading, 12-bit ADC", ("int", 0, 4095)),
    ("weather API {status, temp_c, conditions, alerts[<=3]}",
     ("object", [("enum", 3), ("int", -90, 60), ("enum", 12), ("array", ("enum", 20), 3)])),
    ("classifier: one of 1000 labels or an error code (of 8)",
     ("union", [("enum", 1000), ("enum", 8)])),
    ("retrieval: ranked top-10 of 1,000,000 docs", ("ranked", 10 ** 6, 10)),
    ("retrieval: unordered top-10 of 1,000,000 docs", ("set", 10 ** 6, 10)),
    ("tweet-length text, <= 280 bytes", ("bytes", 280)),
    ("LLM tool reply, <= 2000 tokens, V = 100,277", ("tokens", 100277, 2000)),
]
for name, s in FORMATS:
    print(f"      {bits(s):>12.2f} bits   {name}")
ck("weather API bound = log2 3 + log2 151 + log2 12 + log2(1+20+400+8000)",
   abs(bits(FORMATS[4][1]) - (log2(3) + log2(151) + log2(12) + log2(8421))) < 1e-9)
ck("test runner bound = 1 + 8 bits", abs(bits(FORMATS[1][1]) - 9) < 1e-12)
ck("ranked retrieval = log2(10^6!/(10^6-10)!) < 10 log2 10^6",
   bits(FORMATS[6][1]) < 10 * log2(10 ** 6))

# Corollary 7.1 with format bounds: identifying one of n = 2^20 suspects with
# error eps needs T >= ((1 - eps) k - 1) / C calls of a 1-bit tool.
k, eps = 20, Fr(1, 20)
ck("1-bit tool, 2^20 suspects, eps = 0.05: at least 18 calls",
   ((1 - eps) * k - 1) / Fr(bits(("bool",))) == 18)

print()
print("ALL PASSED" if all(res) else "SOME FAILED")
raise SystemExit(0 if all(res) else 1)

#!/usr/bin/env python3
"""
Experiment 4 - Prefix-reconstruction query complexity (CPU, no GPU, no timing noise).

This is the information-theoretic core of the "how bad is the timing channel" question,
separated from the noisy GPU timing measurement. Given the hash-chain property proven in
exp1 (a cache HIT on block k requires the ENTIRE preceding chain + these 16 tokens to
match), an attacker who can distinguish HIT from MISS (via TTFT timing) can recover an
unknown prefix BLOCK-BY-BLOCK, left to right, 16 tokens at a time. Each recovered block
prunes the search for the next one.

We model the attacker as an oracle O(candidate_prefix) -> {HIT, MISS} that returns HIT iff
candidate_prefix is a cached prefix of the victim (this is exactly what block-aligned TTFT
probing gives you on a shared-cache instance). We count ORACLE QUERIES to full recovery
under different amounts of attacker prior knowledge, because queries -> wall-clock and ->
detectability, and this is what a defense must inflate.

NO GPU and NO fabricated timings: this measures the SEARCH cost that the timing channel,
once it works, would incur. The GPU harness (exp3) measures whether the oracle itself is
reliable (KS/KDE separation); this measures what you can do with a reliable oracle.

Key result to report: recovery is linear in prefix length and linear in per-block search
space -> a low-entropy secret (a templated system prompt, a PIN, a short id) embedded in a
shared prefix is cheaply recoverable; a high-entropy secret is not. This is the precise
statement of the threat and the precise thing cache_salt / no-cross-user-sharing defeats.
"""
import random, statistics, math

BLOCK = 16
line = "="*72
def hdr(t): print("\n"+line+"\n"+t+"\n"+line)

class VictimCache:
    """Ground-truth oracle: HIT iff `cand` is a block-aligned prefix of the secret."""
    def __init__(self, secret_tokens):
        self.secret = list(secret_tokens)
        self.queries = 0
    def probe(self, cand):
        self.queries += 1
        n = (len(cand) // BLOCK) * BLOCK          # only full blocks are cached
        return n > 0 and self.secret[:n] == list(cand[:n])

def recover_blockwise(oracle, nblocks, candidates_per_pos):
    """
    candidates_per_pos(block_index) -> iterable of the 16-token candidate blocks the
    attacker is willing to try at that position (this encodes prior knowledge).
    Returns (recovered_tokens or None, success).
    """
    recovered = []
    for b in range(nblocks):
        found = None
        for blk in candidates_per_pos(b):
            if oracle.probe(recovered + list(blk)):
                found = list(blk); break
        if found is None:
            return recovered, False
        recovered += found
    return recovered, True

# ---------------------------------------------------------------- scenario A
hdr("A. Low-entropy secret in a shared prefix (templated prompt w/ a k-of-V slot)")
print("Model: a system prompt whose blocks are known EXCEPT one token per block drawn")
print("from a small set of V candidates (e.g. a field the attacker knows the format of).")
for V in (10, 100, 1000):
    qs = []
    for trial in range(30):
        r = random.Random(trial)
        nb = 8
        secret = []
        unknown_pos = []                        # (block, index-in-block, true value)
        for b in range(nb):
            blk = [r.randrange(50000) for _ in range(BLOCK)]
            j = r.randrange(BLOCK)              # one unknown slot in this block
            true_v = r.randrange(V)
            blk[j] = 900000 + true_v           # encode the secret value
            unknown_pos.append((b, j, true_v))
            secret += blk
        oracle = VictimCache(secret)
        known_blocks = {b: (list(secret[b*BLOCK:(b+1)*BLOCK]), j) for (b, j, _) in unknown_pos}
        def cands(b, known=known_blocks, V=V):
            base, j = known[b]
            for v in range(V):
                t = list(base); t[j] = 900000 + v
                yield t
        rec, ok = recover_blockwise(oracle, nb, cands)
        assert ok and rec == secret
        qs.append(oracle.queries)
    print(f"  V={V:5d} candidates/slot, 8 blocks: mean {statistics.mean(qs):6.1f} queries "
          f"(<= {V*8} worst-case) -> LINEAR in V and in #blocks, NOT exponential.")

# ---------------------------------------------------------------- scenario B
hdr("B. Why block-chaining makes it linear, not V^blocks (the crucial asymmetry)")
print("A naive attacker guessing the whole prefix at once faces V^(#unknowns) space.")
print("The hash CHAIN lets the oracle confirm each block independently once the prefix")
print("up to it is fixed, collapsing V^n -> V*n. Demonstration at V=50:")
V, nb = 50, 6
naive = V**nb
r = random.Random(0)
secret = []
known = {}
for b in range(nb):
    blk = [r.randrange(50000) for _ in range(BLOCK)]; j = r.randrange(BLOCK)
    blk[j] = 900000 + r.randrange(V); known[b] = (list(blk), j); secret += blk
oracle = VictimCache(secret)
def cands(b):
    base, j = known[b]
    for v in range(V):
        t = list(base); t[j] = 900000 + v; yield t
rec, ok = recover_blockwise(oracle, nb, cands)
print(f"  naive joint search space  : V^n = {V}^{nb} = {naive:,}")
print(f"  block-chained oracle cost : {oracle.queries} queries (success={ok})")
print(f"  speedup                   : ~{naive/max(oracle.queries,1):,.0f}x")

# ---------------------------------------------------------------- scenario C
hdr("C. High-entropy secret is safe (the defense the channel already has by accident)")
print("If the unknown slot is a full high-entropy token (V ~ vocab = 32000) AND there is")
print("more than one unknown per block, per-block cost ~ V^(unknowns/block); recovery")
print("becomes infeasible. The channel only threatens LOW-entropy / templated secrets.")
for unk in (1, 2, 3):
    per_block = 32000**unk
    print(f"  {unk} unknown token(s)/block @ vocab 32000 -> ~{per_block:,} queries/block "
          f"({'cheap' if per_block<1e5 else 'infeasible'})")

# ---------------------------------------------------------------- scenario D
hdr("D. What a defense must do (mapped to vLLM knobs)")
print("cache_salt (per tenant/session): the victim's blocks hash differently from the")
print("  attacker's identical candidates -> EVERY probe MISSES -> oracle destroyed,")
print("  recovery cost -> infinite. This is the clean fix and exp3 prices its TTFT cost.")
print("disable cross-request sharing / per-tenant KV pools: same effect, coarser.")
print("rate-limiting probes: raises query cost linearly; does NOT change the V*n scaling,")
print("  so it only slows a low-entropy-secret attack, does not stop it.")

hdr("done")
print("Reconstruction-complexity analysis complete (CPU, exact oracle). Real HIT/MISS")
print("oracle RELIABILITY under timing noise: modal_exp3_ttft.py (KS / KDE separation).")

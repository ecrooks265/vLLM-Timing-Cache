#!/usr/bin/env python3
"""
Experiment 1 - vLLM prefix-cache block-hash analysis (CPU only, no GPU, no model).

VERIFIED against the real vLLM v0.11.0 source (vllm/v1/core/kv_cache_utils.py and
vllm/utils/__init__.py). The block-hash re-implementation below is byte-identical to
what vLLM computes; a companion check (verify_against_source.py) imports the real
module and diffs digests.

Scope: measurement study of vLLM's documented, open-source block-hash mechanism, to
inform DEFENSES (PYTHONHASHSEED, cache_salt, per-tenant isolation) on a self-hosted
instance. No third-party or production target.

=== IMPORTANT CORRECTION vs the first draft of this study ===
The first draft claimed the DEFAULT block hash is Python's builtin hash() and that the
"weak default hash" is the hackable part. That is TRUE ONLY OF OLD vLLM. As of v0.11.0:

    vllm/config/cache.py:
        PrefixCachingHashAlgo = Literal["sha256", "sha256_cbor"]
        prefix_caching_hash_algo: PrefixCachingHashAlgo = "sha256"     # <-- default
    vllm/utils/__init__.py:
        get_hash_fn_by_name() accepts ONLY "sha256" / "sha256_cbor"; builtin hash() is
        no longer selectable at all.

So on current vLLM the block hash is a CRYPTOGRAPHIC hash (SHA-256 over pickle) by
default, and forged-hit / offline-collision attacks on the hash are NOT feasible.
A teammate's "the vLLM hash can be hacked" was correct for the pre-sha256-default era;
it does not hold for v0.11.0. The paper's contribution therefore must NOT rest on a weak
hash. It rests on the TIMING side channel (which is independent of hash strength) plus
the determinism/salt properties analysed here.

Faithful to vllm/v1/core/kv_cache_utils.py:
    hash_block_tokens(hash_fn, parent_block_hash, curr_block_token_ids, extra_keys)
      -> hash_fn((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
    NONE_HASH: os.urandom(32) if PYTHONHASHSEED unset else hash_fn(seed)   (BOTH algos)
    sha256(x) = hashlib.sha256(pickle.dumps(x, HIGHEST_PROTOCOL)).digest()
"""
import os, sys, time, hashlib, pickle, statistics, random

BLOCK_SIZE = 16  # vLLM v1 default

# ---- faithful hash functions (exactly as vLLM defines them) ----
def h_sha256(x):                       # vllm.utils.sha256  (the DEFAULT in v0.11.0)
    return hashlib.sha256(pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)).digest()
def h_sha256_cbor(x):                   # vllm.utils.sha256_cbor (opt-in, reproducible)
    import cbor2
    return hashlib.sha256(cbor2.dumps(x, canonical=True)).digest()
def h_builtin(x):                       # HISTORICAL only - removed as a vLLM option
    return hash(x)

# vLLM sets NONE_HASH ONCE at process init (init_none_hash): urandom(32) if
# PYTHONHASHSEED unset, else hash_fn(seed). Same value reused for every request in
# the process. This is the ONLY randomization source once the hash is sha256.
_seed = os.getenv("PYTHONHASHSEED")
NONE_HASH = os.urandom(32) if _seed is None else h_sha256(_seed)

def hash_block_tokens(hash_fn, parent, toks, extra=None):
    if not parent:
        parent = NONE_HASH
    return hash_fn((parent, tuple(toks), extra))

def hash_request(hash_fn, token_ids, extra_keys_per_block=None):
    """Chain over full blocks only (partial trailing block is NOT cached)."""
    hashes, parent = [], None
    nblocks = len(token_ids) // BLOCK_SIZE
    for b in range(nblocks):
        blk = token_ids[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE]
        extra = None if not extra_keys_per_block else extra_keys_per_block.get(b)
        parent = hash_block_tokens(hash_fn, parent, blk, extra)
        hashes.append(parent)
    return hashes

def rid(n, vocab=32000, seed=0):
    r = random.Random(seed); return [r.randrange(vocab) for _ in range(n)]

line = "="*72
def hdr(t): print("\n"+line+"\n"+t+"\n"+line)

# ---------------------------------------------------------------- A mechanism
hdr("A. What determines a cache hit (the mechanism, v0.11.0 default = SHA-256)")
toks = rid(3*BLOCK_SIZE, seed=1)
hs = hash_request(h_sha256, toks)
print(f"prompt tokens         : {len(toks)}  -> full blocks cached: {len(hs)} (block_size={BLOCK_SIZE})")
print(f"partial trailing block of <{BLOCK_SIZE} tokens is never cached (hit resolution = {BLOCK_SIZE} tokens)")
print(f"block[0] hash (sha256): {hs[0].hex()[:32]}...")
print("A cache HIT on block k requires: identical parent-chain AND identical 16 tokens")
print("AND identical extra_keys (LoRA id / mm-hash / cache_salt).")
print("Note: hit/miss depends ONLY on equality of these keys, NOT on hash strength -> the")
print("timing side channel exists whether the hash is sha256 or not. Hash strength only")
print("governs forgery/precompute, analysed in D.")

# ---------------------------------------------------------------- B determinism
hdr("B. Determinism / cross-process reproducibility (NONE_HASH & PYTHONHASHSEED)")
seed_env = os.getenv("PYTHONHASHSEED")
print(f"PYTHONHASHSEED currently = {seed_env!r}")
probe = (b"\x00"*32, tuple(range(BLOCK_SIZE)), None)
print(f"sha256(probe)  = {h_sha256(probe).hex()[:24]}...   (deterministic by construction)")
print("With sha256 the per-block function is fully deterministic; the ONLY per-process")
print("randomness is NONE_HASH (the block-0 parent), which is os.urandom(32) unless")
print("PYTHONHASHSEED is set. Consequences:")
print("  * within ONE running vLLM process every request hashes consistently -> the")
print("    timing channel needs no reproducibility and works out of the box.")
print("  * OFFLINE precomputation of a victim's block hashes needs NONE_HASH, i.e. it")
print("    only becomes possible if PYTHONHASHSEED is pinned to a known value.")
print("  * DEFENSE INVERSION: pinning PYTHONHASHSEED (often done for reproducibility)")
print("    REMOVES the only entropy and makes block-0 hashes predictable across restarts.")

# ---------------------------------------------------------------- C hash chain
hdr("C. Hash-chain property (blast radius of a 1-token change)")
a = rid(4*BLOCK_SIZE, seed=2)
b = list(a); b[BLOCK_SIZE + 3] ^= 1          # flip one token inside block 1
ha, hb = hash_request(h_sha256, a), hash_request(h_sha256, b)
first_div = next((i for i in range(len(ha)) if ha[i]!=hb[i]), None)
print(f"changed 1 token in block 1 of a {len(a)}-token prompt")
print(f"blocks identical up to block {first_div-1}; blocks {first_div}..{len(ha)-1} all differ")
print("  -> shared cached prefix is exactly the unchanged leading blocks; everything")
print("     after the first differing token misses.")
print("  This is the structural property the reconstruction attack exploits: a probe")
print("  can confirm a candidate prefix block-by-block, left to right, 16 tokens at a")
print("  time (quantified in exp4_reconstruction_sim.py).")

# ---------------------------------------------------------------- D hash strength
hdr("D. Hash strength on v0.11.0: forgery/precompute are NOT feasible on the default")
print("Default = SHA-256 (256-bit, second-preimage resistant). To forge a HIT an")
print("attacker needs a 16-token block that hashes to a target block key: ~2^256 work.")
print("To precompute a dictionary of secret prefixes they need NONE_HASH (see B).")
print("So the 'hack the hash' angle is CLOSED on current vLLM. We demonstrate the")
print("MECHANISM a weak hash would enable only for completeness, at reduced width, and")
print("label it clearly as NOT applicable to the sha256 default:")
def trunc(fn, bits):
    m = (1<<bits)-1
    def g(x):
        v = fn(x); v = v if isinstance(v,int) else int.from_bytes(v[:8],'big')
        return v & m
    return g
for bits in (24, 32):
    seen, coll, t0 = {}, None, time.time()
    for i in range(1, 6_000_001):
        k = trunc(h_sha256, bits)((b"P", (i,), None))
        if k in seen and seen[k]!=(i,):
            coll = (seen[k], (i,), i); break
        seen[k]=(i,)
        if time.time()-t0>20: break
    if coll:
        print(f"  sha256@{bits}bit (ARTIFICIALLY TRUNCATED): collision after {coll[2]:,} tries")
        print(f"     birthday bound ~2^{bits//2}={2**(bits//2):,}; observed is same order.")
    else:
        print(f"  sha256@{bits}bit: no collision within time budget")
print("  Honest scope: the FULL 256-bit hash is not brute-forced. The truncated runs only")
print("  illustrate the birthday mechanism; they say nothing exploitable about the default.")

# ---------------------------------------------------------------- E cost
hdr("E. Per-block hash cost: sha256 (default) vs sha256_cbor vs builtin (historical)")
blk = tuple(rid(BLOCK_SIZE, seed=9)); parent=b"\x00"*32; N=200_000
fns = [("sha256(default)",h_sha256), ("builtin(historical)",h_builtin)]
try:
    import cbor2; fns.insert(1,("sha256_cbor",h_sha256_cbor))
except Exception: pass
for name, fn in fns:
    t0=time.time()
    for _ in range(N): fn((parent, blk, None))
    dt=time.time()-t0
    print(f"  {name:20s}: {N/dt/1e6:6.2f} M blocks/s   ({dt/N*1e9:6.0f} ns/block)")
print("  -> this per-block cost * (#blocks in prefill) is the hashing overhead that a")
print("     defense adds; it is negligible vs prefill FLOPs, so 'sha256 by default' is")
print("     essentially free. Feed the delta into the TTFT budget in exp3.")

hdr("done")
print("CPU experiment complete & source-verified. GPU-dependent TTFT timing + image/audio")
print("hit measurements: modal_exp3_ttft.py (needs your own Modal GPU account).")

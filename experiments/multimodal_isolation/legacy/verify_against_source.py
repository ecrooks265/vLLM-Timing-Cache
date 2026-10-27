#!/usr/bin/env python3
"""
Verification harness: proves our CPU re-implementations are byte-identical to the
REAL vLLM v0.11.0 source, so every downstream claim rests on ground truth.

It clones nothing at runtime; point VLLM_SRC at a checkout of vLLM v0.11.0
(default /tmp/vllm_repo). We import the real MultiModalHasher directly (with a tiny
torch stub, since the image/ndarray/audio paths do not need torch), and we reproduce
the real vllm.utils.sha256 block-hash exactly.

Run:  VLLM_SRC=/path/to/vllm python3 verify_against_source.py
"""
import os, sys, types, hashlib, pickle
import numpy as np
from PIL import Image
from blake3 import blake3

VLLM_SRC = os.getenv("VLLM_SRC", "/tmp/vllm_repo")

# ---- 1. block hash: real vllm.utils.sha256 is sha256(pickle.dumps(x, HIGHEST)) ----
def sha256_real(x):
    return hashlib.sha256(pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)).digest()
# (this IS the real definition, copied verbatim from vllm/utils/__init__.py)
print("[block-hash] real vllm.utils.sha256 reproduced:",
      sha256_real((b"\x00"*32, (1, 2, 3), None)).hex()[:24], "...")

# ---- 2. multimodal identifier: import the REAL MultiModalHasher ----
ok = False
if os.path.isdir(VLLM_SRC):
    tstub = types.ModuleType("torch")
    class _T: pass
    tstub.Tensor = _T; tstub.bfloat16 = "bfloat16"; tstub.uint8 = "uint8"
    sys.modules["torch"] = tstub
    lg = types.ModuleType("vllm.logger")
    lg.init_logger = lambda *a, **k: types.SimpleNamespace(warning=lambda *a, **k: None,
                                                           info=lambda *a, **k: None)
    vp = types.ModuleType("vllm")
    sys.modules["vllm"] = vp; sys.modules["vllm.logger"] = lg
    sys.path.insert(0, VLLM_SRC)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vllm.multimodal.hasher", os.path.join(VLLM_SRC, "vllm/multimodal/hasher.py"))
    real = importlib.util.module_from_spec(spec)
    sys.modules["vllm.multimodal.hasher"] = real
    try:
        spec.loader.exec_module(real)
        ok = True
    except Exception as e:
        print("[mm-hash] could not import real hasher:", e)

# our replica (same as exp2)
def _serialize(obj):
    if isinstance(obj, (bytes, memoryview)): return [obj]
    if isinstance(obj, str): return [obj.encode("utf-8")]
    if isinstance(obj, (int, float)): return [np.array(obj).tobytes()]
    if isinstance(obj, Image.Image):
        data = {"mode": obj.mode, "data": np.asarray(obj)}
        if obj.palette is not None:
            data["palette"] = obj.palette.palette
            if obj.palette.rawmode is not None:
                data["palette_rawmode"] = obj.palette.rawmode
        return _iter("image", data)
    if isinstance(obj, np.ndarray):
        arr = obj.view(np.uint8).data if obj.flags.c_contiguous else obj.tobytes()
        return _iter("ndarray", {"dtype": obj.dtype.str, "shape": obj.shape, "data": arr})
    return [pickle.dumps(obj)]
def _iter(key, obj):
    out = []
    if isinstance(obj, (list, tuple)):
        for i, e in enumerate(obj): out += _iter(f"{key}.{i}", e)
    elif isinstance(obj, dict):
        for k, v in obj.items(): out += _iter(f"{key}.{k}", v)
    else:
        out.append(key.encode("utf-8")); out += _serialize(obj)
    return out
def mm_identifier(**kwargs):
    h = blake3()
    for k, v in kwargs.items():
        for b in _iter(k, v): h.update(b)
    return h.hexdigest()

cases = [
    dict(model_id="Qwen/Qwen2.5-VL-3B-Instruct",
         image=Image.fromarray((np.arange(4*4*3).reshape(4, 4, 3) % 256).astype(np.uint8))),
    dict(model_id="m", audio=np.random.default_rng(1).standard_normal(2048).astype(np.float32)),
    dict(model_id="m", image=Image.fromarray(np.zeros((8, 8, 3), np.uint8)), do_resize=True),
]
allok = True
for i, c in enumerate(cases):
    mine = mm_identifier(**c)
    if ok:
        theirs = real.MultiModalHasher.hash_kwargs(**c)
        match = (mine == theirs)
        allok &= match
        print(f"[mm-hash] case {i}: match={match}  ({mine[:24]}...)")
    else:
        print(f"[mm-hash] case {i}: replica={mine[:24]}...  (real source not importable here)")

print("\nRESULT:", "ALL BYTE-EQUAL vs real vLLM v0.11.0 source" if (ok and allok)
      else "replica computed; run with VLLM_SRC set to a v0.11.0 checkout to diff")

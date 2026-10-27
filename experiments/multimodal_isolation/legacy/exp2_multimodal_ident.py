#!/usr/bin/env python3
"""
Experiment 2 - Multimodal prefix-cache identifiers: image & audio (CPU, no GPU).

VERIFIED byte-for-byte against the real vLLM v0.11.0 MultiModalHasher
(vllm/multimodal/hasher.py) - see verify_against_source.py, which imports the real
class and confirms identical digests.

=== CORRECTION vs the first draft ===
The first draft assumed the multimodal identifier = sha256(processed_feature_tensor).
The real v0.11.0 behaviour (vllm/multimodal/hasher.py + processing.py::_hash_mm_items):

    identifier = blake3( model_id, <media>, **hf_processor_mm_kwargs, **tokenization_kwargs )

where <media> for an image is the DECODED PIL image serialized as
    {"mode": img.mode, "data": np.asarray(img), "palette": ...}
i.e. the RAW decoded pixel array (plus model_id and processor kwargs), NOT the
post-processor feature tensor. Two extra real facts that matter:
  * if the PIL image carries an EXIF ImageID UUID tag, that UUID is used INSTEAD of
    pixel content (an app-supplied identity can override content).
  * callers may pass explicit mm_uuids to bypass content hashing entirely.

Consequences for the side channel (all confirmed on CPU below):
  1. image: identical decoded pixels + same model_id + same kwargs -> identical id (hit);
     any 1-pixel change -> different id (miss).  => EXACT-content membership oracle.
  2. cross-model isolation is automatic: different model_id -> different id, so the
     oracle only confirms media submitted to the SAME model deployment.
  3. audio: same structure over the audio array; the real question is whether "the same
     clip" produces byte-identical arrays after decode+resample - the determinism caveat
     that bounds how usable the audio channel is.
  4. capability = MEMBERSHIP / CONFIRMATION oracle on a KNOWN image/audio
     ("did someone else submit exactly this?"), NOT reconstruction of unknown media.
"""
import hashlib, pickle, numpy as np
from blake3 import blake3
from PIL import Image

# ---- faithful re-implementation of MultiModalHasher (verified byte-equal) ----
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

def mm_identifier(**kwargs) -> str:
    h = blake3()
    for k, v in kwargs.items():
        for b in _iter(k, v): h.update(b)
    return h.hexdigest()

def block_extra_key(mm_id, offset): return (mm_id, offset)

line = "="*72
def hdr(t): print("\n"+line+"\n"+t+"\n"+line)

rng = np.random.default_rng(0)
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# ---------------------------------------------------------------- 1 image
hdr("1. IMAGE: identical decoded pixels -> hit; 1-pixel change -> miss")
px = rng.integers(0, 256, size=(336, 336, 3), dtype=np.uint8)
img = Image.fromarray(px)
id0 = mm_identifier(model_id=MODEL, image=img)
id_same = mm_identifier(model_id=MODEL, image=Image.fromarray(px.copy()))
px2 = px.copy(); px2[0, 0, 0] ^= 1
id_flip = mm_identifier(model_id=MODEL, image=Image.fromarray(px2))
print(f"id(image)            = {id0[:32]}...")
print(f"id(identical decode) = {id_same[:32]}...   HIT={id0==id_same}")
print(f"id(1-bit pixel flip) = {id_flip[:32]}...   HIT={id0==id_flip}")
diff = sum(a != b for a, b in zip(id0, id_flip))
print(f"avalanche: {diff}/{len(id0)} hex chars differ from a single-bit change (~50% expected)")
print("Reading: the key is exact decoded pixels -> a HIT confirms byte-identical media;")
print("it does NOT leak a similar/nearby image. Reconstruction via this channel is")
print("infeasible; CONFIRMING a specific known image was submitted by someone IS feasible.")

# ---------------------------------------------------------------- 1b model isolation & UUID override
hdr("1b. Cross-model isolation + EXIF-UUID / explicit-uuid override (real code paths)")
idA = mm_identifier(model_id="org/model-A", image=img)
idB = mm_identifier(model_id="org/model-B", image=img)
print(f"same image, different model_id -> different id: {idA!=idB}")
print("  => the membership oracle is automatically scoped to ONE model deployment.")
# processor-kwargs dependence (this is the real audio-determinism lever too)
idk0 = mm_identifier(model_id=MODEL, image=img, do_resize=True)
idk1 = mm_identifier(model_id=MODEL, image=img, do_resize=False)
print(f"same image, different processor kwarg -> different id: {idk0!=idk1}")
print("  => any change in processor/tokenization kwargs also changes the id.")
print("EXIF ImageID UUID path: if img.getexif()[ImageID] is a UUID, vLLM uses that UUID")
print("INSTEAD of pixels (hasher.py) -> an app can pin identity independent of content;")
print("and callers may pass explicit mm_uuids to skip content hashing entirely.")

# ---------------------------------------------------------------- 2 audio
hdr("2. AUDIO: same hasher over the audio array + the decode-determinism caveat")
audio = rng.standard_normal(16000).astype(np.float32)     # 1s @16kHz stand-in
a0 = mm_identifier(model_id=MODEL, audio=audio)
a_same = mm_identifier(model_id=MODEL, audio=audio.copy())
a_eps = audio.copy(); a_eps[0] += np.float32(1e-6)         # sub-perceptual change
a_eps_id = mm_identifier(model_id=MODEL, audio=a_eps)
a_f64 = mm_identifier(model_id=MODEL, audio=audio.astype(np.float64))  # dtype change
print(f"id(audio)            = {a0[:32]}...")
print(f"id(identical)        HIT={a0==a_same}")
print(f"id(+1e-6 one sample) HIT={a0==a_eps_id}   <- sub-perceptual change already misses")
print(f"id(float32->float64) HIT={a0==a_f64}   <- dtype alone changes the id")
print("Caveat that decides REAL audio hits: two recordings of 'the same' clip collide only")
print("if the full decode path (container->PCM, resample rate, channel mix, trim, dtype)")
print("is byte-identical. exp3 measures whether a re-encode of the same wav still hits;")
print("this bounds the audio channel's practicality. A bit-exact re-send of the same file")
print("WILL hit; a re-encode / resample generally will NOT.")

# ---------------------------------------------------------------- 3 into block hash
hdr("3. How the media id enters the block hash (and why cache_salt closes the channel)")
print("generate_block_hash_extra_keys() appends (mm_identifier, offset) for each block")
print("covering a media placeholder:")
print(f"  example extra_key = {block_extra_key(id0[:16], 0)}")
print("block_hash = sha256((parent, tokens, extra_keys)) -> the media id is IN the key, so")
print("cross-user media sharing is the multimodal analogue of prefix sharing.")
print("Defense (cache_salt): appended to the FIRST block's extra_keys when start_token_idx==0")
print("(generate_block_hash_extra_keys / kv_cache_utils.py). A per-tenant salt makes even")
print("identical media hash differently across tenants -> the membership oracle closes.")
print("exp3 measures the TTFA cost of turning it on.")

hdr("done")
print("CPU multimodal analysis complete & source-verified. Real TTFA hit/miss timing for")
print("image & audio: modal_exp3_ttft.py (your own Modal GPU account).")

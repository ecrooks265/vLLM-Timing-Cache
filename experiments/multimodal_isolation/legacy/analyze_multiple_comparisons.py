#!/usr/bin/env python3
"""THREATS_TO_VALIDITY.md item: we computed dozens of KS tests and never corrected for
multiple comparisons. The base paper does (Bonferroni /3 over v in {1,5,25}, /2 over two
timing sources, alpha = 1e-8).

This script collects EVERY KS p-value in the project, groups them into the two families we
actually make claims about, and applies both Bonferroni and Benjamini-Hochberg FDR.

Families are grouped by the CLAIM under test, not by which condition the request used:
  * SIGNAL family -- "there IS a detectable channel here"      (we claim YES)
  * NULL family   -- "there is NO detectable channel here"     (we claim NO / bounded)
Lumping an image-under-salt test (where we claim a signal) together with a text-under-salt
test (where we claim no signal) would put opposite hypotheses in one family -- a bug in the
first version of this script.
"""
import json, glob, os
import numpy as np

TESTS = []


def add(family, name, p):
    if p is None or not np.isfinite(p):
        return
    TESTS.append({"family": family, "name": name, "p": float(p)})


def get(d, *path):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def sep_p(sep):
    """Prefer the base paper's one-sided p when present, else the two-sided p."""
    if not isinstance(sep, dict):
        return None, None
    if "ks_p_one_sided_greater" in sep:
        return sep["ks_p_one_sided_greater"], "one-sided"
    return sep.get("ks_p"), "two-sided"


def scan_text2():
    d = json.load(open("exp3_text2.json"))
    for L, r in d["by_prefix_tokens"].items():
        add("signal", "text %s tok, defense OFF" % L, sep_p(r["off_detector"])[0])
        add("null", "text %s tok, defense ON (claim: closed)" % L, sep_p(r["on_detector"])[0])


def scan_simple(fn, pairs):
    if not os.path.exists(fn):
        return
    d = json.load(open(fn))
    for fam, label, path in pairs:
        p = sep_p(get(d, *path))[0]
        add(fam, label, p)


def scan_repeat(fn, label_prefix, keys):
    if not os.path.exists(fn):
        return
    d = json.load(open(fn))
    for r in d.get("reps", []):
        for k in keys:
            if k in r and isinstance(r[k], dict):
                add("null" if k == "salt_on_attack" and label_prefix == "text" else "signal",
            "%s rep%d %s" % (label_prefix, r["rep"], k), sep_p(r[k])[0])


if __name__ == "__main__":
    scan_text2()
    scan_simple("exp3_mm.json", [("signal", "image: repeated vs fresh image",
                                  ("off", "attacker_detector_hit_vs_miss")),
                                 ("signal", "image: attacker vs unseen under salt (claim: OPEN)",
                                  ("on", "attacker_detector_cached_vs_unseen"))])
    scan_simple("exp3_audio.json", [("signal", "audio: identical vs fresh clip",
                                     ("off", "attacker_detector_hit_vs_miss")),
                                    ("signal", "audio: attacker vs unseen under salt (claim: OPEN)",
                                     ("on", "attacker_detector_cached_vs_unseen"))])
    scan_simple("exp3_mm_extra.json", [("signal", "image: same image / different prompt",
                                        ("same_image_different_prompt", "detector")),
                                       ("null", "image: one pixel different (claim: closed)",
                                        ("one_pixel_different", "detector"))])
    scan_repeat("exp3_repeat_text.json", "text", ("idle_open_channel", "load8_open_channel",
                                                  "salt_on_attack"))
    scan_repeat("exp3_repeat_mm.json", "image", ("salt_on_attack", "same_image_different_prompt"))
    scan_repeat("exp3_repeat_audio.json", "audio", ("salt_on_attack",))

    # cross-check: the 8-thread measurements that disagreed across runs
    if os.path.exists("exp3_text_load.json"):
        d = json.load(open("exp3_text_load.json"))
        for k, v in d["by_load"].items():
            add("signal", "text_load %s threads" % k, sep_p(v["detector"])[0])

    out = {}
    for family in ("signal", "null"):
        fam = [t for t in TESTS if t["family"] == family]
        if not fam:
            continue
        m = len(fam)
        ps = np.array([t["p"] for t in fam])
        order = np.argsort(ps)
        # Benjamini-Hochberg
        bh = np.empty(m)
        prev = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            val = ps[i] * m / (rank + 1)
            prev = min(prev, val)
            bh[i] = prev
        print("=" * 92)
        print("FAMILY: %-9s  (%d tests)" % (family, m))
        print("=" * 92)
        print("  %-52s %11s %11s %11s %s" % ("test", "raw p", "Bonferroni", "BH-FDR", "verdict@0.05"))
        for i, t in enumerate(sorted(fam, key=lambda x: x["p"])):
            idx = fam.index(t)
            bonf = min(1.0, t["p"] * m)
            bhq = bh[idx]
            # with alpha=1e-8, the base paper's threshold, for reference
            verdict = "significant" if bhq < 0.05 else "NOT significant"
            print("  %-52s %11.3g %11.3g %11.3g %s" % (t["name"], t["p"], bonf, bhq, verdict))
        print()
        print("  -> at BH-FDR q=0.05: %d/%d significant" %
              (int((bh < 0.05).sum()), m))
        print("  -> at the base paper's alpha=1e-8 (Bonferroni-corrected): %d/%d significant" %
              (int((np.minimum(1.0, ps * m) < 1e-8).sum()), m))
        print()
        out[family] = {"m": m,
                       "bonferroni_significant_0.05": int((np.minimum(1.0, ps * m) < 0.05).sum()),
                       "bh_significant_0.05": int((bh < 0.05).sum()),
                       "bonferroni_significant_1e-8": int((np.minimum(1.0, ps * m) < 1e-8).sum()),
                       "tests": [{"name": t["name"], "p": t["p"],
                                  "p_bonferroni": float(min(1.0, t["p"] * m)),
                                  "p_bh": float(bh[fam.index(t)])} for t in fam]}

    allp = np.array([t["p"] for t in TESTS])
    m = len(allp)
    print("=" * 92)
    print("POOLED (both families mixed) -- reported only for completeness: %d tests, "
          "Bonferroni alpha=0.05/%d=%.2g" % (m, m, 0.05 / m))
    print("=" * 92)
    print("  significant after Bonferroni at 0.05: %d/%d" %
          (int((allp * m < 0.05).sum()), m))
    out["pooled"] = {"m": m, "bonferroni_significant_0.05": int((allp * m < 0.05).sum())}
    json.dump(out, open("multiple_comparisons.json", "w"), indent=2)
    print("\nwrote multiple_comparisons.json")

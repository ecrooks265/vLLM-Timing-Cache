#!/usr/bin/env python3
"""Rigour pass over the exp3 JSONs -- no GPU needed.

The first pass said "cache_salt closes the text channel" on the strength of KS p-values
that merely failed to reach significance. That is absence of evidence, and a reviewer
would (correctly) attack it. This script turns it into a MEASURED BOUND, using the raw
per-trial latencies that the second-pass functions persist (gt() -> "raw_ttft_ms").

For every salt-ON comparison it reports:
  * bootstrap CI on the p50 latency delta between the two arms (ms);
  * bootstrap CI on AUC (0.5 = indistinguishable);
  * whether ANY threshold is error-free (the gap between the arms' extremes);
  * the residual leak as a percentage of that SAME channel's open (salt-off) effect,
    so "<= 1.5 ms" becomes "<= 2% of the leak we are trying to stop".

Usage: python3 analyze_rigor.py
"""
import json, os
import numpy as np

B = 12000
_rng = np.random.default_rng(12345)


def ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def boot_delta(a, b, B=B):
    """CI of median(b) - median(a). Positive => the 'b' arm is slower."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = np.empty(B)
    for i in range(B):
        d[i] = (np.median(_rng.choice(b, b.size, True)) -
                np.median(_rng.choice(a, a.size, True)))
    return float(np.median(b) - np.median(a)), ci(d)


def auc(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    gt = float((a[:, None] > b[None, :]).sum())
    eq = float((a[:, None] == b[None, :]).sum())
    return (gt + 0.5 * eq) / (a.size * b.size)


def boot_auc(a, b, B=B):
    a = np.asarray(a, float); b = np.asarray(b, float)
    v = np.empty(B)
    for i in range(B):
        v[i] = auc(_rng.choice(a, a.size, True), _rng.choice(b, b.size, True))
    return auc(a, b), ci(v)


def perfect_margin(a, b):
    """min(b) - max(a) > 0  <=>  some threshold separates the arms with zero errors."""
    return float(np.min(b) - np.max(a))


def report(tag, a, b, open_ms=None):
    d, dci = boot_delta(a, b)
    au, auci = boot_auc(a, b)
    marg = perfect_margin(a, b)
    print("  %-46s n=%d/%d" % (tag, len(a), len(b)))
    print("      p50 delta %+8.2f ms   95%% CI [%+.2f, %+.2f]" % (d, dci[0], dci[1]))
    print("      AUC           %.3f      95%% CI [%.3f, %.3f]" % (au, auci[0], auci[1]))
    try:
        from scipy.stats import ks_2samp as _k
        p1 = float(_k(a, b, alternative="greater").pvalue)
        print("      one-sided KS p (the paper's test, alternative='greater') = %.3g" % p1)
    except Exception:
        pass
    print("      perfect-detector margin (min_b - max_a) = %+.2f ms  -> %s" %
          (marg, "an error-free threshold EXISTS" if marg > 0
           else "no threshold is error-free"))
    if open_ms is not None:
        bound = max(abs(dci[0]), abs(dci[1]))
        print("      ==> residual leak bounded by %.2f ms = %.1f%% of this channel's "
              "open effect (%.1f ms)" % (bound, bound / open_ms * 100.0, open_ms))
    return {"delta_p50_ms": d, "delta_ci": dci, "auc": au, "auc_ci": auci,
            "perfect_margin_ms": marg, "n": [len(a), len(b)]}


def analyse_text2(path):
    d = json.load(open(path))
    print("=" * 88)
    print("TEXT (raw samples) -- is 'cache_salt closes the channel' a BOUND or a failed test?")
    print("=" * 88)
    summary = {}
    for L, r in d["by_prefix_tokens"].items():
        off = r["off_detector"]["delta_p50_ms"]
        A = r["on_attacker"]; Bb = r["on_attacker_control"]
        s = report("prefix %5s tokens | SALT ON" % L, A["raw_ttft_ms"], Bb["raw_ttft_ms"],
                   open_ms=off)
        gta, gtb = A.get("cached_tokens_p50"), Bb.get("cached_tokens_p50")
        print("      ground truth: cached KV tokens %s vs %s -> %s" %
              (gta, gtb, "IDENTICAL (KV cache properly isolated)" if gta == gtb
               else "DIFFERENT (investigate!)"))
        print("      open-channel effect at this length: %+.2f ms" % off)
        print()
        s["open_effect_ms"] = off
        s["cached_tokens_arms"] = [gta, gtb]
        summary[L] = s
    return summary


def analyse_positive(path, title, arms):
    if not os.path.exists(path):
        print("(skip %s -- not present yet)" % path)
        return None
    d = json.load(open(path))
    print("=" * 88)
    print(title)
    print("=" * 88)
    out = {}
    for label, sec, ka, kb in arms:
        A = d.get(sec, {}).get(ka); Bb = d.get(sec, {}).get(kb)
        if not A or not Bb or not A.get("raw_ttft_ms") or not Bb.get("raw_ttft_ms"):
            print("  %-44s (no raw samples)" % label); continue
        out[label] = report(label, A["raw_ttft_ms"], Bb["raw_ttft_ms"])
        print("      ground truth: cached KV tokens %s vs %s" %
              (A.get("cached_tokens_p50"), Bb.get("cached_tokens_p50")))
        print()
    return out


if __name__ == "__main__":
    res = {}

    if os.path.exists("exp3_text2.json"):
        res["text_salt_on_bounds"] = analyse_text2("exp3_text2.json")
        d = json.load(open("exp3_text2.json"))

        rec = d.get("recovery")
        if rec:
            print("=" * 88)
            print("END-TO-END LIVE SECRET RECOVERY (TTFT only; no oracle access)")
            print("=" * 88)
            print("  design: %s" % rec["design"])
            print("  V=%d candidates/field x %d fields = %d queries/trial; naive joint "
                  "search space = %d (%.0fx more)" %
                  (rec["v_candidates_per_field"], rec["fields"], rec["queries_per_trial"],
                   rec["naive_joint_space"], rec["naive_joint_space"] / rec["queries_per_trial"]))
            for arm in ("salt_off", "salt_on"):
                s = rec[arm]
                print("  %-8s fields recovered %d/%d   trials fully correct %d/%d   "
                      "median margin %.1f ms" %
                      (arm.upper(), s["fields_recovered"], s["fields_total"],
                       s["trials_fully_correct"], len(rec["runs"]) // 2,
                       s["median_margin_ms"]))
            print("  salt-ON is the negative control; chance = %.2f fields per 3-field trial"
                  % rec["salt_on"]["chance_level_fields"])
            print()
            res["recovery"] = rec

        cl = d.get("concurrent_load")
        if cl:
            print("=" * 88)
            print("DETECTOR UNDER CONCURRENT LOAD (real servers are not idle)")
            print("=" * 88)
            t = []
            for k, r in cl.items():
                a = r["detector"]
                print("  %-12s n=%d  hit p50 %7.2f ms  miss p50 %7.2f ms  delta %+7.2f ms  "
                      "KS %.3f  AUC %.3f  overlap %.3f" %
                      (k, r["n"], r["hit_p50_ms"], r["miss_p50_ms"], a["delta_p50_ms"],
                       a["ks_stat"], a["auc"], a["kde_overlap"]))
                t.append({"threads": r["concurrent_threads"], "delta_ms": a["delta_p50_ms"],
                          "auc": a["auc"], "hit_p50_ms": r["hit_p50_ms"],
                          "miss_p50_ms": r["miss_p50_ms"]})
            print()
            res["concurrent_load"] = t
    else:
        print("(exp3_text2.json not present yet)")

    res["mm_extra"] = analyse_positive("exp3_mm_extra.json", "IMAGE second pass", [
        ("same image, DIFFERENT prompt text", "same_image_different_prompt",
         "victim_image_attacker_prompt", "never_sent_image_attacker_prompt"),
        ("one pixel different", "one_pixel_different", "victim_image_resubmitted",
         "never_sent_image"),
    ])

    if os.path.exists("exp3_mm_extra.json"):
        d = json.load(open("exp3_mm_extra.json"))
        de = d.get("deep_eviction")
        if de:
            print("=" * 88)
            print("IMAGE: deep encoder-cache eviction (does the residual go to ~0?)")
            print("=" * 88)
            print("  before flood: victim image %7.2f ms | after flood %7.2f ms | "
                  "never-sent control %7.2f ms" %
                  (de["before_flood_victim_image"]["ttft_ms"],
                   de["after_flood_victim_image"]["ttft_ms"],
                   de["never_sent_control"]["ttft_ms"]))
            print("  attacker advantage %.1f ms -> %.1f ms   (residual fraction %.1f%%)" %
                  (de["advantage_before_ms"], de["advantage_after_ms"],
                   de["residual_fraction"] * 100.0))
            print("  sanity, same-tenant reuse after the flood: warm %.1f ms -> hit %.1f ms "
                  "(cached tokens %s)" %
                  (de["sanity_same_tenant_reuse_still_works"]["warm_ttft_ms"],
                   de["sanity_same_tenant_reuse_still_works"]["hit_ttft_ms"],
                   de["sanity_same_tenant_reuse_still_works"]["hit_cached_tokens"]))
            print()
            res["deep_eviction"] = de

    if os.path.exists("exp3_audio_extra.json"):
        d = json.load(open("exp3_audio_extra.json"))
        print("=" * 88)
        print("AUDIO: how much re-encoding does the channel survive?")
        print("=" * 88)
        base = d["baseline_never_sent_wav"].get("cached_tokens_p50")
        print("  template baseline (never-sent clip) : cached_tokens %s" % base)
        print("  identical WAV re-sent               : cached_tokens %s" %
              d["baseline_identical_wav_resend"].get("cached_tokens_p50"))
        print("  (cached_tokens above the baseline == it reused the ORIGINAL clip's entry)")
        for k, v in d["reencode_variants"].items():
            if "error" in v:
                print("  %-36s: %s" % (k, v["error"])); continue
            print("  %-36s: cached_tokens %-5s reuses original entry: %s" %
                  (k, v["first_send_cached_tokens"], v["reuses_original_entry"]))
        print()
        res["reencode_variants"] = d["reencode_variants"]

    json.dump(res, open("rigor_summary.json", "w"), indent=2, default=float)
    print("wrote rigor_summary.json")


# ---------------------------------------------------------------------------------
# Detector characterisation in the style of Gu et al. (ICML'25), whose Figure 4
# reports precision-recall curves for "distinguishing times from the cache hit and
# cache miss procedures". We have raw per-trial latencies, so we can report the full
# operating characteristic instead of a single summary statistic: the attacker scores
# a request by how FAST it was and calls it cached if the latency is below a threshold.
# ---------------------------------------------------------------------------------

def roc_pr(hit, miss):
    """hit = latencies that really were cache hits (positive class);
    miss = latencies that really were misses (negative class).
    Score = -latency (lower latency => more likely cached)."""
    hit = np.asarray(hit, float); miss = np.asarray(miss, float)
    P, N = len(hit), len(miss)
    y = np.r_[np.ones(P), np.zeros(N)]
    sc = np.r_[-hit, -miss]
    o = np.argsort(-sc)
    y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / P; fpr = fp / N
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tpr
    roc_auc = float(np.trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    # step-wise average precision, guarding the ends
    r = np.r_[0.0, recall]; pr = np.r_[1.0, precision]
    ap = float(np.sum(np.diff(r) * pr[1:]))
    j = tpr - fpr
    k = int(np.argmax(j))
    return {"roc_auc": roc_auc, "pr_auc": ap, "baseline_pr_auc": float(P / (P + N)),
            "best_threshold_ms": float(sc[o][k]) * -1.0,
            "tpr_at_best": float(tpr[k]), "fpr_at_best": float(fpr[k]),
            "precision_at_best": float(precision[k]), "youden_j": float(j[k]),
            "n_hit": P, "n_miss": N}


def detector_operating_points(title, specs):
    print("=" * 88)
    print(title)
    print("=" * 88)
    rows = []
    for label, hit, miss in specs:
        if not hit or not miss:
            print("  %-42s (no raw samples)" % label); continue
        r = roc_pr(hit, miss)
        print("  %-40s ROC AUC %.3f  PR AUC %.3f (base %.3f)  best thr %6.1f ms -> "
              "TPR %.2f / FPR %.2f (precision %.2f)" %
              (label, r["roc_auc"], r["pr_auc"], r["baseline_pr_auc"],
               r["best_threshold_ms"], r["tpr_at_best"], r["fpr_at_best"],
               r["precision_at_best"]))
        rows.append({"label": label, **r})
    if not rows:
        print("  (nothing to report)")
    print()
    return rows


def run_detector_char():
    out = {}
    if os.path.exists("exp3_text2.json"):
        d = json.load(open("exp3_text2.json"))
        specs = []
        for L, r in d["by_prefix_tokens"].items():
            specs.append(("text %s tok, defense OFF" % L,
                          r["off_hit"].get("raw_ttft_ms"),
                          r["off_control_unseen"].get("raw_ttft_ms")))
            specs.append(("text %s tok, defense ON" % L,
                          r["on_attacker"].get("raw_ttft_ms"),
                          r["on_attacker_control"].get("raw_ttft_ms")))
        out["text"] = detector_operating_points("DETECTOR CHARACTERISATION (Gu et al. Fig-4 style)", specs)

    if os.path.exists("exp3_mm_extra.json"):
        x = json.load(open("exp3_mm_extra.json"))
        out["image"] = detector_operating_points("IMAGE DETECTOR CHARACTERISATION", [
            ("image, same image / different prompt",
             x["same_image_different_prompt"]["victim_image_attacker_prompt"].get("raw_ttft_ms"),
             x["same_image_different_prompt"]["never_sent_image_attacker_prompt"].get("raw_ttft_ms")),
            ("image, one pixel different",
             x["one_pixel_different"]["victim_image_resubmitted"].get("raw_ttft_ms"),
             x["one_pixel_different"]["never_sent_image"].get("raw_ttft_ms")),
        ])

    if os.path.exists("exp3_text2.json"):
        d = json.load(open("exp3_text2.json"))
        out["under_load"] = detector_operating_points("DETECTOR CHARACTERISATION UNDER LOAD", [
            ("load %s threads, 1024 tok" % k.split("_")[1],
             r.get("raw_hit_ttft_ms"), r.get("raw_miss_ttft_ms"))
            for k, r in d.get("concurrent_load", {}).items()])

    json.dump(out, open("detector_characterisation.json", "w"), indent=2, default=float)
    print("wrote detector_characterisation.json")


run_detector_char()

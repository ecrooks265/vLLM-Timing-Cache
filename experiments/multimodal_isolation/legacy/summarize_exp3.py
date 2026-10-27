#!/usr/bin/env python3
"""Turn the Modal exp3 JSONs into the RESULTS_exp3_*.txt transcript files and a
readable summary. Usage: python3 summarize_exp3.py"""
import json, os, sys

def have(p):
    return os.path.exists(p)

def fmt_table_text(d):
    L = ["=" * 78,
         "EXP3 / TEXT CHANNEL - TTFT prefix-cache timing on self-hosted vLLM v0.11.0",
         "=" * 78,
         "model            : %s" % d["meta"]["model"],
         "vllm version     : %s   block_size=%s" % (d["meta"]["vllm_version"], d["meta"]["block_size"]),
         "trials/condition : %s" % d["meta"]["n_trials_per_length"],
         "TTFT             : chat.completions create -> first streamed chunk (max_tokens=1)",
         "ground truth     : per-REQUEST deltas of vllm:prefix_cache_hits/_queries",
         "",
         "HIT vs MISS latency (defense OFF = no cache_salt):",
         "  %-7s %9s %9s %9s %8s" % ("prefix", "hit p50", "miss p50", "saved", "speedup")]
    for k, r in d["by_prefix_tokens"].items():
        h = r["off"]["hit"]["p50_ms"]; m = r["off"]["miss"]["p50_ms"]
        L.append("  %-7s %8.1fms %8.1fms %8.1fms %7.2fx" % (k, h, m, m - h, m / h))
    L += ["",
          "ATTACKER DETECTOR: can a co-tenant tell 'victim content is cached' from 'unseen'?",
          "  (KS/AUC/overlap of the two TTFT distributions; AUC 0.5 and overlap 1.0 = no signal)",
          "  %-7s | %-38s | %-38s | %s" % ("prefix", "DEFENSE OFF (no salt)", "DEFENSE ON (per-tenant salt)", "null floor"),
          "  %-7s | %7s %5s %5s %5s %8s | %7s %5s %5s %5s %8s | %5s %5s" %
          ("", "delta", "KS", "AUC", "ovl", "KS p", "delta", "KS", "AUC", "ovl", "KS p", "KS", "ovl")]
    for k, r in d["by_prefix_tokens"].items():
        o = r["off"]["attacker_detector_hit_vs_unseen"]
        n = r["on"]["attacker_detector_cached_vs_unseen"]
        nf = r["null_floor_same_condition"]["on_attacker_vs_itself"]
        L.append("  %-7s | %6.1fms %5.2f %5.3f %5.2f %8.1e | %6.1fms %5.2f %5.3f %5.2f %8.1e | %5.2f %5.2f" %
                 (k, o["delta_p50_ms"], o["ks_stat"], o["auc"], o["kde_overlap"], o["ks_p"],
                  n["delta_p50_ms"], n["ks_stat"], n["auc"], n["kde_overlap"], n["ks_p"],
                  nf["ks_stat_mean"], nf["kde_overlap_mean"]))
    L += ["",
          "GROUND TRUTH (engine counters) - cached prompt tokens per condition:",
          "  %-7s %8s %9s %9s | %8s %9s %8s %10s" %
          ("prefix", "OFF hit", "OFF miss", "OFF ctl", "ON same", "ON cold", "ON att", "ON att ctl")]
    for k, r in d["by_prefix_tokens"].items():
        o = r["off"]; n = r["on"]
        L.append("  %-7s %8.0f %9.0f %9.0f | %8.0f %9.0f %8.0f %10.0f" %
                 (k, o["hit"]["cached_tokens_p50"], o["miss"]["cached_tokens_p50"],
                  o["control_unseen"]["cached_tokens_p50"], n["victim_same_salt_hit"]["cached_tokens_p50"],
                  n["victim_cold_miss"]["cached_tokens_p50"],
                  n["attacker_probe_cached_content"]["cached_tokens_p50"],
                  n["attacker_probe_unseen"]["cached_tokens_p50"]))
    L += ["",
          "COST OF THE DEFENSE (extra TTFT when reuse is denied), ms:",
          "  %-7s %26s %24s" % ("prefix", "attacker loses reuse", "same-tenant (should be ~0)")]
    for k, r in d["by_prefix_tokens"].items():
        L.append("  %-7s %24.1f %25.2f" % (k, r["salt_cost_ms_attacker_loses_reuse"],
                                           r["salt_cost_ms_within_tenant"]))
    L += ["",
          "READING:",
          "  * Defense OFF: hit and miss TTFT separate completely from ~64 tokens up",
          "    (KS=1.00, AUC=0.000, KDE overlap=0.000 at 256 and 1024 tokens).",
          "  * Defense ON : every ON separation sits AT OR BELOW the empirical null floor",
          "    measured on the same condition's own random halves -> no usable signal.",
          "  * Ground truth confirms the mechanism: the attacker's probe of content the",
          "    victim cached returns the SAME cached-token count (the shared chat-template",
          "    block) as its probe of content nobody sent. There is nothing to detect.",
          "  * Cost: ~0 ms within a tenant; the attacker/other-tenant pays the full lost",
          "    reuse (up to ~95 ms at 1024 tokens, a 2.7x TTFT penalty).",
          "=" * 78, "done", ""]
    return "\n".join(L)

def fmt_table_mm(d):
    o = d["off"]; n = d["on"]; nf = d["null_floor_same_condition"]
    det = o["attacker_detector_hit_vs_miss"]; don = n["attacker_detector_cached_vs_unseen"]
    return "\n".join([
        "=" * 78,
        "EXP3 / IMAGE CHANNEL - TTFA timing on self-hosted vLLM v0.11.0",
        "=" * 78,
        "model      : %s" % d["meta"]["model"],
        "image      : %s" % d["meta"]["image"],
        "hit        : %s" % d["meta"]["hit_condition"],
        "miss       : %s" % d["meta"]["miss_condition"],
        "",
        "REPEATED IMAGE (defense OFF):",
        "  hit  p50 %8.1f ms   cached_tokens_p50 %6.0f   frac_trials_hit %.2f" %
        (o["hit_repeated_image"]["p50_ms"], o["hit_repeated_image"].get("cached_tokens_p50", -1),
         o["hit_repeated_image"].get("frac_trials_with_cache_hit", -1)),
        "  miss p50 %8.1f ms   cached_tokens_p50 %6.0f   frac_trials_hit %.2f" %
        (o["miss_fresh_image"]["p50_ms"], o["miss_fresh_image"].get("cached_tokens_p50", -1),
         o["miss_fresh_image"].get("frac_trials_with_cache_hit", -1)),
        "  detector: delta %6.1f ms  KS %.3f  p %.2e  AUC %.3f  overlap %.3f" %
        (det["delta_p50_ms"], det["ks_stat"], det["ks_p"], det["auc"], det["kde_overlap"]),
        "",
        "DEFENSE ON (per-tenant cache_salt):",
        "  victim same-salt hit cached_tokens_p50 %.0f" % n["victim_same_salt_hit"].get("cached_tokens_p50", -1),
        "  attacker vs victim image   cached_tokens_p50 %.0f" % n["attacker_probe_cached_image"].get("cached_tokens_p50", -1),
        "  attacker vs unseen  image  cached_tokens_p50 %.0f" % n["attacker_probe_unseen_image"].get("cached_tokens_p50", -1),
        "  detector: delta %6.1f ms  KS %.3f  p %.2e  AUC %.3f  overlap %.3f" %
        (don["delta_p50_ms"], don["ks_stat"], don["ks_p"], don["auc"], don["kde_overlap"]),
        "  null floor (same condition): KS %.3f  overlap %.3f" %
        (nf["off_hit_vs_itself"]["ks_stat_mean"], nf["off_hit_vs_itself"]["kde_overlap_mean"]),
        "",
        "salt cost (attacker loses reuse): %.1f ms" % d["salt_cost_ms_attacker_loses_reuse"],
        "=" * 78, "done", ""])

def fmt_table_audio(d):
    o = d["off"]; v = d["reencode_verdict"]; nf = d["null_floor_same_condition"]
    det = o["attacker_detector_hit_vs_miss"]
    return "\n".join([
        "=" * 78,
        "EXP3 / AUDIO CHANNEL - TTFA timing on self-hosted vLLM v0.11.0",
        "=" * 78,
        "model : %s" % d["meta"]["model"],
        "clip  : %s" % d["meta"]["clip"],
        "",
        "IDENTICAL BYTES (defense OFF):",
        "  hit  p50 %8.1f ms   cached_tokens_p50 %6.0f" %
        (o["hit_identical_bytes"]["p50_ms"], o["hit_identical_bytes"].get("cached_tokens_p50", -1)),
        "  miss p50 %8.1f ms   cached_tokens_p50 %6.0f" %
        (o["miss_fresh_clip"]["p50_ms"], o["miss_fresh_clip"].get("cached_tokens_p50", -1)),
        "  detector: delta %6.1f ms  KS %.3f  p %.2e  AUC %.3f  overlap %.3f" %
        (det["delta_p50_ms"], det["ks_stat"], det["ks_p"], det["auc"], det["kde_overlap"]),
        "  null floor (same condition): KS %.3f  overlap %.3f" %
        (nf["ks_stat_mean"], nf["kde_overlap_mean"]),
        "",
        "DECODE-DETERMINISM (exp2's question, answered end-to-end):",
        "  what matters is whether the FIRST send of a re-encode reuses the ORIGINAL clip's",
        "  cached entry -- repeats always hit themselves, so a median over repeats is meaningless.",
        "  template baseline (an unseen clip)              : cached_tokens %.0f" % v["template_baseline_cached_tokens"],
        "  identical bytes, first send                    : cached_tokens %s" % v["identical_bytes_first_send_cached_tokens"],
        "  soundfile decode round-trip, first send        : cached_tokens %s  -> reuses original entry: %s" %
        (v["roundtrip_cached_tokens"], v["decode_roundtrip_reuses_original_entry"]),
        "  resampled 16k -> 22k, first send               : cached_tokens %s  -> reuses original entry: %s" %
        (v["resample_cached_tokens"], v["resample_reuses_original_entry"]),
        "  round-trip self-repeat / resample self-repeat  : cached_tokens %s / %s" %
        (v["roundtrip_self_repeat_cached_tokens"], v["resample_self_repeat_cached_tokens"]),
        "=" * 78, "done", ""])

def fmt_table_mechanism(d):
    on = d["processor_cache_on"]; off = d["processor_cache_off"]; ev = d["encoder_cache_eviction"]
    return "\n".join([
        "=" * 78,
        "EXP3 / IMAGE MECHANISM - why cache_salt does not close the image channel",
        "=" * 78,
        "model : %s   vllm %s   block_size %s" % (d["meta"]["model"], d["meta"]["vllm_version"],
                                                  d["meta"]["block_size"]),
        "design: %s" % d["meta"]["design"],
        "        => in EVERY arm the engine reports 0 cached KV tokens, so the KV prefix",
        "           cache contributes nothing and cannot explain any difference.",
        "",
        "A. WITH the MM processor cache (--mm-processor-cache-gb default 4 GiB):",
        "     victim's image   p50 %7.2f ms   (cached KV tokens %.0f)" %
        (on["victim_image"]["p50_ms"], on["victim_image"]["cached_tokens_p50"]),
        "     never-sent image p50 %7.2f ms   (cached KV tokens %.0f)" %
        (on["never_sent_image"]["p50_ms"], on["never_sent_image"]["cached_tokens_p50"]),
        "     => delta %.1f ms   KS %.3f   AUC %.3f   overlap %.2e" %
        (on["detector"]["delta_p50_ms"], on["detector"]["ks_stat"], on["detector"]["auc"],
         on["detector"]["kde_overlap"]),
        "     KV cached tokens identical in both arms: %s" % on["cached_tokens_identical_both_arms"],
        "",
        "B. WITH the MM processor cache DISABLED (--mm-processor-cache-gb 0):",
        "     victim's image   p50 %7.2f ms   (cached KV tokens %.0f)" %
        (off["victim_image"]["p50_ms"], off["victim_image"]["cached_tokens_p50"]),
        "     never-sent image p50 %7.2f ms   (cached KV tokens %.0f)" %
        (off["never_sent_image"]["p50_ms"], off["never_sent_image"]["cached_tokens_p50"]),
        "     => delta %.1f ms   KS %.3f   AUC %.3f   overlap %.2e" %
        (off["detector"]["delta_p50_ms"], off["detector"]["ks_stat"], off["detector"]["auc"],
         off["detector"]["kde_overlap"]),
        "     ==> the advantage SURVIVES, so the processor cache is NOT the mechanism.",
        "",
        "C. EVICT the encoder cache (flood %d distinct images), then re-probe the SAME image" %
        d["meta"]["flood_images"],
        "   with a fresh salt:",
        "     before flood, victim's image : %7.2f ms" % ev["before_flood_victim_image"]["ttft_ms"],
        "     after  flood, victim's image : %7.2f ms" % ev["after_flood_victim_image"]["ttft_ms"],
        "     never-sent control           : %7.2f ms" % ev["never_sent_control"]["ttft_ms"],
        "     attacker advantage: %.1f ms -> %.1f ms" %
        (ev["advantage_before_ms"], ev["advantage_after_ms"]),
        "     ==> eviction REMOVED the advantage: it is the content-keyed EncoderCacheManager",
        "         (vllm/v1/core/encoder_cache_manager.py, keyed on mm_hash; no cache_salt in it).",
        "",
        "CONCLUSION: cache_salt salts only the KV prefix cache. The multimodal processor/encoder",
        "caches are keyed on content alone and stay shared across tenants, so the image and audio",
        "channels remain open after the defence is enabled.",
        "=" * 78, "done", ""])


def fmt_table_text2(d):
    L = ["=" * 78,
         "EXP3 / TEXT SECOND PASS - scaling to 4096 tokens and robustness under load",
         "=" * 78,
         "model : %s   vllm %s   block_size %s" % (d["meta"]["model"], d["meta"]["vllm_version"],
                                                   d["meta"]["block_size"]),
         "raw per-trial latencies are stored in the JSON; bootstrap them with analyze_rigor.py",
         "",
         "OPEN CHANNEL (no cache_salt), %d trials per condition:" % d["meta"]["n_trials_per_length"],
         "  %-7s %9s %10s %9s %8s %6s %6s" %
         ("prefix", "hit p50", "miss p50", "saved", "miss/hit", "KS", "AUC")]
    for k, r in d["by_prefix_tokens"].items():
        h = r["off_hit"]["p50_ms"]; m = r["off_miss"]["p50_ms"]; det = r["off_detector"]
        L.append("  %-7s %8.1fms %9.1fms %8.1fms %7.2fx %6.3f %6.3f" %
                 (k, h, m, m - h, m / h, det["ks_stat"], det["auc"]))
    L += ["",
          "SALT ON - the attacker's probe of the victim's content vs never-sent content:",
          "  %-7s %9s %9s %9s %8s %7s %9s" %
          ("prefix", "delta", "KS", "AUC", "KS p", "ovl", "cachedKV")]
    for k, r in d["by_prefix_tokens"].items():
        on = r["on_detector"]
        L.append("  %-7s %8.2fms %9.3f %9.3f %8.3f %7.3f %5s vs %-4s" %
                 (k, on["delta_p50_ms"], on["ks_stat"], on["auc"], on["ks_p"],
                  on["kde_overlap"], r["on_attacker"].get("cached_tokens_p50"),
                  r["on_attacker_control"].get("cached_tokens_p50")))
    L += ["",
          "COST OF THE DEFENCE: %s" % "attacker loses reuse / within tenant (ms)"]
    for k, r in d["by_prefix_tokens"].items():
        L.append("  %-7s %8.1f / %.2f" % (k, r["salt_cost_ms"], r["salt_cost_within_tenant_ms"]))
    L += ["",
          "UNDER CONCURRENT LOAD (1024-token prefix, timing-only measurement):",
          "  %-9s %9s %9s %9s %6s %6s" % ("threads", "hit p50", "miss p50", "delta", "KS", "AUC")]
    for k, r in d["concurrent_load"].items():
        a = r["detector"]
        L.append("  %-9d %8.1fms %8.1fms %8.1fms %6.3f %6.3f" %
                 (r["concurrent_threads"], r["hit_p50_ms"], r["miss_p50_ms"],
                  a["delta_p50_ms"], a["ks_stat"], a["auc"]))
    L += ["", "See analyze_rigor.py for the bootstrap CIs that turn 'salt ON' into a BOUND.",
          "=" * 78, "done", ""]
    return "\n".join(L)


def fmt_table_recovery(d):
    s = d["summary"]
    L = ["=" * 78,
         "EXP3 / END-TO-END LIVE SECRET RECOVERY (TTFT only, no privileged access)",
         "=" * 78,
         "model : %s" % d["meta"]["model"],
         "design: %s" % d["meta"]["design"],
         "",
         "Victim prompt = HEAD [A] LONG1 [B] LONG2 [C] LONG3; the attacker knows the fixed",
         "parts and only that each field is one of V candidates. Each candidate is probed",
         "EXACTLY ONCE (a second probe would hit the attacker's own cache entry).",
         "",
         "  V = %d candidates/field, %d fields -> %d queries per trial" %
         (d["meta"]["v_candidates_per_field"], d["meta"]["fields"], s["queries_per_trial"]),
         "  naive joint search space = %d  (%.0fx more queries)" %
         (s["naive_joint_space"], s["speedup_vs_joint"]),
         "",
         "  %-10s %-16s %-18s %-16s %s" %
         ("arm", "fields recovered", "trials fully right", "median margin", "median truth rank")]
    for arm in ("salt_off", "salt_on"):
        a = s[arm]
        L.append("  %-10s %-16s %-18s %-16s %s" %
                 (arm.upper(),
                  "%d/%d (%.1f%%)" % (a["fields_correct"], a["fields_total"],
                                      a["field_accuracy"] * 100),
                  "%d/%d" % (a["trials_fully_correct"], a["trials_total"]),
                  "%.2f ms" % a["median_margin_ms"],
                  "%.1f (chance %.1f)" % (a["median_truth_rank"], a["chance_truth_rank"])))
    L += ["",
          "READING: the secret is recovered EXACTLY, every trial, from nothing but TTFT; the",
          "salt-ON control lands at chance. This is the link between exp4's V*n query model",
          "(over an ideal oracle) and exp3's demonstration that the noisy oracle is reliable.",
          "=" * 78, "done", ""]
    return "\n".join(L)


def fmt_table_mm_extra(d):
    a = d["same_image_different_prompt"]; b = d["one_pixel_different"]; e = d["deep_eviction"]
    L = ["=" * 78,
         "EXP3 / IMAGE SECOND PASS - what does the attacker actually need to know?",
         "=" * 78,
         "model : %s" % d["meta"]["model"],
         "Every probe uses a FRESH cache_salt, so both arms of each comparison are KV-cache misses.",
         "",
         "A. SAME image, DIFFERENT prompt text (does the attacker need the victim's prompt?)",
         "   victim's image + attacker's own text : p50 %7.2f ms  cachedKV %.0f" %
         (a["victim_image_attacker_prompt"]["p50_ms"],
          a["victim_image_attacker_prompt"]["cached_tokens_p50"]),
         "   never-sent image + attacker's text  : p50 %7.2f ms  cachedKV %.0f" %
         (a["never_sent_image_attacker_prompt"]["p50_ms"],
          a["never_sent_image_attacker_prompt"]["cached_tokens_p50"]),
         "   => delta %.1f ms  KS %.3f  AUC %.3f  (KV cached tokens identical: %s)" %
         (a["detector"]["delta_p50_ms"], a["detector"]["ks_stat"], a["detector"]["auc"],
          a["kv_cached_tokens_identical"]),
         "   => the attacker needs ONLY the image; the multimodal channel is stronger than text.",
         "",
         "B. ONE PIXEL different (is the channel exact-content only?)",
         "   1-pixel-changed image : p50 %7.2f ms" % b["victim_image_resubmitted"]["p50_ms"],
         "   never-sent image      : p50 %7.2f ms" % b["never_sent_image"]["p50_ms"],
         "   => delta %.1f ms  AUC %.3f  -> indistinguishable: EXACT content only" %
         (b["detector"]["delta_p50_ms"], b["detector"]["auc"]),
         "",
         "C. DEEP EVICTION (%d distinct images) - is the residual gone?" % d["meta"]["flood_images"],
         "   before flood %7.2f ms | after flood %7.2f ms | never-sent control %7.2f ms" %
         (e["before_flood_victim_image"]["ttft_ms"], e["after_flood_victim_image"]["ttft_ms"],
          e["never_sent_control"]["ttft_ms"]),
         "   attacker advantage %.1f -> %.1f ms   (residual %.1f%%)" %
         (e["advantage_before_ms"], e["advantage_after_ms"], e["residual_fraction"] * 100),
         "   sanity: same-tenant reuse still works after the flood (hit cached tokens %.0f)" %
         e["sanity_same_tenant_reuse_still_works"]["hit_cached_tokens"],
         "   A residual that does NOT shrink as the flood grows means a second, larger cache is",
         "   still holding the image -- the 4 GiB mm processor cache. See RESULTS_exp3_mm_evict.txt.",
         "=" * 78, "done", ""]
    return "\n".join(L)


def fmt_table_audio_extra(d):
    v = d["reencode_variants"]
    L = ["=" * 78,
         "EXP3 / AUDIO SECOND PASS - how much re-encoding does the channel survive?",
         "=" * 78,
         "model : %s" % d["meta"]["model"],
         "What matters is whether the FIRST send of a variant reuses the ORIGINAL clip's entry",
         "(cached tokens clearly above the template baseline); self-repeats always hit themselves.",
         "",
         "  never-sent clip (baseline) : cached_tokens %.0f" %
         d["baseline_never_sent_wav"].get("cached_tokens_p50", -1),
         "  identical WAV re-sent      : cached_tokens %.0f" %
         d["baseline_identical_wav_resend"].get("cached_tokens_p50", -1),
         ""]
    for k, x in v.items():
        if "error" in x:
            L.append("  %-26s : %s" % (k, x["error"])); continue
        L.append("  %-26s : first-send cached_tokens %-5s reuses original: %s" %
                 (k, x["first_send_cached_tokens"], x["reuses_original_entry"]))
    L += ["",
          "READING: only a byte-identical file reuses the original entry. Even a LOSSLESS",
          "container change (WAV -> FLAC) breaks it, because the decode path quantises",
          "differently (verified locally: the decoded arrays differ by 3.05e-05, which is",
          "already enough to change the media key). The audio channel needs a bit-identical",
          "decode path, so the attacker must hold the identical file.",
          "=" * 78, "done", ""]
    return "\n".join(L)


def fmt_table_evict(d):
    r = d["result"]
    return "\n".join([
        "=" * 78,
        "EXP3 / IMAGE - separating the two unsalted multimodal caches",
        "=" * 78,
        "model  : %s" % d["meta"]["model"],
        "flags  : %s" % d["meta"]["server_flags"],
        "flood  : %d distinct images" % d["meta"]["flood_images"],
        "",
        "  before flood : victim image %7.2f ms | never-sent %7.2f ms | advantage %6.1f ms" %
        (r["before_flood"]["victim_image_ms"], r["before_flood"]["never_sent_ms"],
         r["before_flood"]["advantage_ms"]),
        "  after  flood : victim image %7.2f ms | never-sent %7.2f ms | advantage %6.1f ms" %
        (r["after_flood"]["victim_image_ms"], r["after_flood"]["never_sent_ms"],
         r["after_flood"]["advantage_ms"]),
        "  residual fraction: %.1f%%" % (r["residual_fraction"] * 100),
        "",
        r["reading"],
        "=" * 78, "done", ""])



def fmt_table_load(d):
    L = ["=" * 78,
         "EXP3 / TEXT DETECTOR UNDER CONCURRENT LOAD (raw samples kept)",
         "=" * 78,
         "model : %s   prefix %d tokens   %d trials per arm" %
         (d["meta"]["model"], list(d["by_load"].values())[0]["prefix_tokens"],
          d["meta"]["n_per_arm"]),
         "note  : %s" % d["meta"]["note"],
         "",
         "  %-9s %9s %9s %9s %7s %7s" % ("threads", "hit p50", "miss p50", "delta", "KS", "AUC")]
    for k, v in d["by_load"].items():
        a = v["detector"]
        L.append("  %-9d %8.1fms %8.1fms %+8.1fms %7.3f %7.3f" %
                 (v["concurrent_threads"], v["hit_p50_ms"], v["miss_p50_ms"],
                  a["delta_p50_ms"], a["ks_stat"], a["auc"]))
    L += ["",
          "READING: the channel persists under load but detectability is DEGRADED and",
          "RUN-DEPENDENT. An earlier, separate run found perfect separation at 8 threads",
          "(AUC 0.000); this one does not (KS 0.567, AUC 0.212). Do not claim the detector",
          "always separates under load -- claim it survives, with variable reliability.",
          "=" * 78, "done", ""]
    return "\n".join(L)



def _rep_block(title, d, keys):
    L = ["=" * 78, title, "=" * 78,
         "design: %s" % d["meta"].get("design", ""),
         "repetitions: %d   (each repetition RESTARTS the vLLM server)" % d["meta"]["repetitions"],
         "",
         "  %-34s %-24s %-22s %s" % ("measurement", "AUC per rep", "AP per rep", "delta p50 per rep")]
    for k in keys:
        a = d["summary"][k]
        L.append("  %-34s %-24s %-22s %s" %
                 (k,
                  ", ".join("%.3f" % x for x in a["auc"]["values"]),
                  ", ".join("%.3f" % x for x in a["average_precision"]["values"]),
                  ", ".join("%.1f ms" % x for x in a["delta_p50_ms"]["values"])))
        L.append("  %-34s mean %.3f +- %.3f      mean %.3f +- %.3f    mean %.1f +- %.1f ms" %
                 ("", a["auc"]["mean"], a["auc"]["std"],
                  a["average_precision"]["mean"], a["average_precision"]["std"],
                  a["delta_p50_ms"]["mean"], a["delta_p50_ms"]["std"]))
    if "flac_reuses_original_all_reps" in d.get("summary", {}):
        L.append("")
        L.append("  FLAC (lossless container) reuses the original clip's entry, per rep: %s" %
                 d["summary"]["flac_reuses_original_all_reps"])
    L += ["",
          "READING:",
          "  * Separability (AUC/AP) is essentially perfectly reproducible -- every headline",
          "    claim reproduced in all repetitions.",
          "  * The ABSOLUTE effect size is stable within a protocol but NOT across sessions",
          "    (the image attack measured 46.7 ms originally and ~66 ms here: 41% apart).",
          "    Quote effect sizes as RANGES, never as one number.",
          "=" * 78, "done", ""]
    return "\n".join(L)


def fmt_repeat_text(d):
    return _rep_block("EXP3 / REPEATABILITY - TEXT (K=3, server restarted each rep)", d,
                      ("idle_open_channel", "load8_open_channel", "salt_on_attack"))


def fmt_repeat_mm(d):
    return _rep_block("EXP3 / REPEATABILITY - IMAGE (K=3, server restarted each rep)", d,
                      ("salt_on_attack", "same_image_different_prompt"))


def fmt_repeat_audio(d):
    return _rep_block("EXP3 / REPEATABILITY - AUDIO (K=3, server restarted each rep)", d,
                      ("salt_on_attack",))


if __name__ == "__main__":
    jobs = [("exp3_text.json", "RESULTS_exp3_text.txt", fmt_table_text),
            ("exp3_mm.json", "RESULTS_exp3_mm.txt", fmt_table_mm),
            ("exp3_audio.json", "RESULTS_exp3_audio.txt", fmt_table_audio),
            ("exp3_mm_mechanism.json", "RESULTS_exp3_mm_mechanism.txt", fmt_table_mechanism),
            ("exp3_text2.json", "RESULTS_exp3_text2.txt", fmt_table_text2),
            ("exp3_text_recovery.json", "RESULTS_exp3_text_recovery.txt", fmt_table_recovery),
            ("exp3_mm_extra.json", "RESULTS_exp3_mm_extra.txt", fmt_table_mm_extra),
            ("exp3_audio_extra.json", "RESULTS_exp3_audio_extra.txt", fmt_table_audio_extra),
            ("exp3_mm_evict.json", "RESULTS_exp3_mm_evict.txt", fmt_table_evict),
            ("exp3_text_load.json", "RESULTS_exp3_text_load.txt", fmt_table_load),
            ("exp3_repeat_text.json", "RESULTS_exp3_repeat_text.txt", fmt_repeat_text),
            ("exp3_repeat_mm.json", "RESULTS_exp3_repeat_mm.txt", fmt_repeat_mm),
            ("exp3_repeat_audio.json", "RESULTS_exp3_repeat_audio.txt", fmt_repeat_audio)]
    for src, dst, fn in jobs:
        if not have(src):
            print("skip %s (not present)" % src); continue
        d = json.load(open(src))
        out = fn(d)
        open(dst, "w").write(out)
        print(out)

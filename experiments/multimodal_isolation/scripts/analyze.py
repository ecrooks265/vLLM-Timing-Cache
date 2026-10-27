#!/usr/bin/env python3
"""Analyze only NEW first-content measurements; legacy is not silently pooled.

Thresholds are learned from calibration images only, per server session. ROC AUC
and AP have cached=positive, score=-first_content_ms. Bootstrap resamples whole
image IDs and whole repetitions, preserving complementary-pass pairing.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def auc_ap(labels, latency):
    y, score = np.asarray(labels, int), -np.asarray(latency, float)
    n1, n0 = int(y.sum()), int((1-y).sum())
    if not n1 or not n0:
        return None, None
    auc = (rankdata(score)[y == 1].sum() - n1*(n1+1)/2) / (n1*n0)
    # Average precision with tied scores grouped, rather than arbitrary tie order.
    order = np.argsort(-score, kind='stable')
    yy, ss = y[order], score[order]
    ends = np.r_[np.where(np.diff(ss) != 0)[0], len(yy)-1]
    tp = np.cumsum(yy)[ends]
    precision = tp/(ends+1)
    recall = tp/n1
    ap = np.sum(np.diff(np.r_[0, recall])*precision)
    return float(auc), float(ap)


def threshold(rows):
    times = sorted(set(r['first_content_ms'] for r in rows))
    if not rows or len(set(r['victim_primed'] for r in rows)) != 2:
        raise ValueError('Both calibration classes required')
    candidates = [times[0]-1e-6] + times + [times[-1]+1e-6]
    def balanced(t):
        pos = [r for r in rows if r['victim_primed']]
        neg = [r for r in rows if not r['victim_primed']]
        return .5*(np.mean([r['first_content_ms'] <= t for r in pos]) +
                   np.mean([r['first_content_ms'] > t for r in neg]))
    return float(max(candidates, key=balanced))


def paired(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['rep'], r['image_id']), {})[r['victim_primed']] = r
    if any(set(v) != {0, 1} for v in groups.values()):
        raise ValueError('Missing complementary-pass pair')
    return groups


def boot_ci(rows, count=1000):
    groups = paired(rows)
    reps = sorted(set(k[0] for k in groups))
    ids = sorted(set(k[1] for k in groups))
    rng = np.random.default_rng(795)
    stats = []
    for _ in range(count):
        # Crossed resampling: keep repeated images across the sampled sessions.
        rr = rng.choice(reps, len(reps), replace=True)
        ii = rng.choice(ids, len(ids), replace=True)
        selected = [groups[(int(rep), ident)] for rep in rr for ident in ii]
        vals = [v[label] for v in selected for label in (0, 1)]
        auc, ap = auc_ap([r['victim_primed'] for r in vals], [r['first_content_ms'] for r in vals])
        delta = np.median([v[0]['first_content_ms']-v[1]['first_content_ms'] for v in selected])
        stats.append([auc, ap, float(delta)])
    return {k: [float(x) for x in np.percentile(np.asarray(stats)[:, j], [2.5, 97.5])]
            for j, k in enumerate(['roc_auc', 'average_precision', 'paired_delta_ms'])}


def cost_ci(roles, count=1000):
    reps = sorted({k[0] for k in roles['cold']})
    ids = sorted({k[1] for k in roles['cold']})
    rng = np.random.default_rng(796)
    values = {'same_tenant': [], 'cross_tenant': []}
    for _ in range(count):
        rr = rng.choice(reps, len(reps), replace=True)
        ii = rng.choice(ids, len(ids), replace=True)
        for role in values:
            values[role].append(float(np.median([roles['cold'][(int(r),i)]-roles[role][(int(r),i)] for r in rr for i in ii])))
    return {role: [float(v) for v in np.percentile(samples, [2.5,97.5])] for role,samples in values.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--bootstrap', type=int, default=1000)
    args = ap.parse_args()
    path = args.run_dir.resolve()
    cfg = json.loads((path/'config.json').read_text())
    meta = json.loads((path/'meta.json').read_text())
    data = [json.loads(line) for line in (path/'requests.jsonl').read_text().splitlines() if line]
    requests = [r for r in data if r['kind'] == 'request']
    valid = [r for r in requests if r.get('ok')]
    calibration = [r for r in valid if r['phase'] == 'attack_probe' and r['split'] == 'calibration']
    evaluation = [r for r in valid if r['phase'] == 'attack_probe' and r['split'] == 'evaluation']
    timings_ok = bool(valid) and all(0 <= r['first_chunk_ms'] <= r['first_content_ms'] <= r['completion_ms'] for r in valid)
    all_counters_observed = bool(valid) and all(r['kv_counter_status'] == 'observed_delta' for r in valid)
    expected_eval = cfg['repetitions'] * len(cfg['salt_modes']) * 2 * cfg['evaluation_images'] if 'attack' in cfg['phases'] else 0
    expected_cost = cfg['repetitions'] * len(cfg['salt_modes']) * 3 * cfg['cost_images'] if 'cost' in cfg['phases'] else 0
    gates = {'complete': meta['status'] == 'complete', 'no_request_failures': len(valid) == len(requests),
             'timing_order_valid': timings_ok, 'kv_counters_observed': all_counters_observed,
             'evaluation_count_correct': len(evaluation) == expected_eval,
             'performance_count_correct': sum(r['phase']=='cost_probe' for r in valid) == expected_cost,
             'manifest_has_no_cross_split_duplicates': len({r['sha256'] for r in json.loads((path/'dataset_manifest.json').read_text())['images']}) == len(json.loads((path/'dataset_manifest.json').read_text())['images'])}
    out = {'source_run': str(path), 'stage': cfg['stage'], 'technical_gates': gates,
           'pilot_not_final_evidence': cfg['stage'] == 'pilot',
           'metric': 'first nonempty streamed content; score=-latency; cached=positive',
           'bootstrap': 'crossed image-ID and repetition resampling; 3 repetitions limit session inference',
           'detection': {}, 'performance': {},
           'timing_calibration': {'median_content_minus_chunk_ms': float(np.median([r['first_content_ms']-r['first_chunk_ms'] for r in valid])) if valid else None}}
    if not all(gates.values()):
        (path/'analysis.json').write_text(json.dumps(out, indent=2))
        if cfg['stage'] == 'pilot':
            (ROOT/'results/pilot_gate.json').write_text(json.dumps({'pass': False, 'run': str(path), 'gates': gates}, indent=2))
        raise SystemExit('Technical gate failed. Saved analysis.json; no final figures or main launch.')
    for mode in cfg['salt_modes']:
        rr = [r for r in evaluation if r['mode'] == mode]
        cal = [r for r in calibration if r['mode'] == mode]
        # Cost-only follow-up is supported; it need not repeat the attack phase.
        if rr:
            cutoffs = {s: threshold([r for r in cal if r['session'] == s]) for s in {r['session'] for r in rr}}
            auc, apv = auc_ap([r['victim_primed'] for r in rr], [r['first_content_ms'] for r in rr])
            pairs = paired(rr)
            deltas = [p[0]['first_content_ms']-p[1]['first_content_ms'] for p in pairs.values()]
            pos = [r for r in rr if r['victim_primed']]
            neg = [r for r in rr if not r['victim_primed']]
            per_rep = []
            for rep in range(cfg['repetitions']):
                sub = [r for r in rr if r['rep'] == rep]
                aa, pp = auc_ap([r['victim_primed'] for r in sub], [r['first_content_ms'] for r in sub])
                per_rep.append({'rep': rep, 'roc_auc': aa, 'average_precision': pp,
                    'paired_delta_ms': float(np.median([p[0]['first_content_ms']-p[1]['first_content_ms'] for (k, _), p in pairs.items() if k == rep]))})
            out['detection'][mode] = {'roc_auc': auc, 'average_precision': apv,
                'paired_delta_ms': float(np.median(deltas)), 'ci95': boot_ci(rr, args.bootstrap),
                'per_rep': per_rep, 'calibration_thresholds_ms': cutoffs,
                'tpr': float(np.mean([r['first_content_ms'] <= cutoffs[r['session']] for r in pos])),
                'fpr': float(np.mean([r['first_content_ms'] <= cutoffs[r['session']] for r in neg])),
                'kv_hits_cached_p50': float(np.median([r['kv_hits_delta'] for r in pos])),
                'kv_hits_unseen_p50': float(np.median([r['kv_hits_delta'] for r in neg])),
                'evaluation_images': cfg['evaluation_images'], 'requests': len(rr)}
        cr = [r for r in valid if r['phase'] == 'cost_probe' and r['mode'] == mode]
        roles = {role: {(r['rep'], r['image_id']): r['first_content_ms'] for r in cr if r['role'] == role}
                 for role in ['cold', 'same_tenant', 'cross_tenant']}
        if roles['cold']:
            if not (roles['cold'].keys() == roles['same_tenant'].keys() == roles['cross_tenant'].keys()):
                raise ValueError('Performance paired image set differs')
            out['performance'][mode] = {'role_p50_ms': {role: float(np.median(list(values.values()))) for role, values in roles.items()},
                'same_tenant_saving_paired_p50_ms': float(np.median([v-roles['same_tenant'][k] for k,v in roles['cold'].items()])),
                'cross_tenant_saving_paired_p50_ms': float(np.median([v-roles['cross_tenant'][k] for k,v in roles['cold'].items()])),
                'paired_saving_ci95_ms': cost_ci(roles, args.bootstrap),
                'per_rep': {str(rep): {role: float(np.median([v for (rp, _),v in values.items() if rp == rep])) for role, values in roles.items()} for rep in range(cfg['repetitions'])}}
    (path/'analysis.json').write_text(json.dumps(out, indent=2))
    lines = ['# New image isolation results', '',
             '**PILOT ONLY — not final evidence.**' if cfg['stage']=='pilot' else 'Measured new run; legacy results are separate.', '',
             '| Salt mode | ROC AUC | AP | Paired cold−cached ms | TPR | FPR |',
             '|---|---:|---:|---:|---:|---:|']
    for mode, r in out['detection'].items():
        lines.append(f"| {mode} | {r['roc_auc']:.3f} | {r['average_precision']:.3f} | {r['paired_delta_ms']:.2f} | {r['tpr']:.3f} | {r['fpr']:.3f} |")
    lines += ['', '| Salt mode | Cold ms | Same-tenant ms | Cross-tenant ms | Paired cross saving ms |',
              '|---|---:|---:|---:|---:|']
    for mode, r in out['performance'].items():
        v = r['role_p50_ms']
        lines.append(f"| {mode} | {v['cold']:.2f} | {v['same_tenant']:.2f} | {v['cross_tenant']:.2f} | {r['cross_tenant_saving_paired_p50_ms']:.2f} |")
    lines += ['', 'All timing is client-side localhost first-content latency. AP baseline is 0.5 for this balanced design.',
              'No claim of zero leakage follows from a nonsignificant test. KV counters do not directly observe encoder hits.',
              'One leaf-photo dataset, one VLM, one vLLM version. This is membership testing, not image reconstruction.', '',
              'See analysis.json for confidence intervals, per-repetition results, and performance values.']
    (path/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    for mode, color in zip(cfg['salt_modes'], ['#2563eb', '#db2777']):
        if mode in out['detection']:
            r = out['detection'][mode]
            i = cfg['salt_modes'].index(mode)
            lo, hi = r['ci95']['roc_auc']
            axes[0].errorbar(i, r['roc_auc'], yerr=[[max(0,r['roc_auc']-lo)], [max(0,hi-r['roc_auc'])]], fmt='o', color=color, capsize=5)
        if mode in out['performance']:
            p = out['performance'][mode]['role_p50_ms']
            axes[1].plot(list(p), list(p.values()), marker='o', color=color, label=mode)
    axes[0].set(xticks=range(len(cfg['salt_modes'])), xticklabels=cfg['salt_modes'], ylim=(0,1.05), ylabel='ROC AUC (cached positive; -latency)', title='Held-out image membership')
    axes[0].axhline(.5, color='gray', linestyle='--')
    axes[1].set(ylabel='Median first-content latency (ms)', title='Separate performance sessions')
    if out['performance']:
        axes[1].legend()
    if cfg['stage']=='pilot':
        fig.suptitle('PILOT — technical validation only')
    fig.tight_layout()
    fig.savefig(path/'isolation_and_cost.png', dpi=160)
    plt.close(fig)
    if cfg['stage'] == 'pilot':
        (ROOT/'results/pilot_gate.json').write_text(json.dumps({'pass': True, 'run': str(path), 'gates': gates}, indent=2))
    print(path/'REPORT.md')


if __name__ == '__main__':
    main()

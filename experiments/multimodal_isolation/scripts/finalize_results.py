#!/usr/bin/env python3
"""Create a handoff report and latency distribution from a gated main run."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
p=Path(sys.argv[1]).resolve()
a=json.loads((p/'analysis.json').read_text()); meta=json.loads((p/'meta.json').read_text()); cfg=json.loads((p/'config.json').read_text())
assert cfg['stage']=='main' and all(a['technical_gates'].values())
rows=[json.loads(l) for l in (p/'requests.jsonl').read_text().splitlines()]
fig,axes=plt.subplots(1,2,figsize=(11,4.7),sharey=True)
for ax,mode in zip(axes,['shared','distinct']):
    vals=[]; labels=[]
    for rep in range(3):
        for cached in [0,1]:
            vals.append([r['first_content_ms'] for r in rows if r.get('phase')=='attack_probe' and r.get('split')=='evaluation' and r['mode']==mode and r['rep']==rep and r['victim_primed']==cached])
            labels.append(f'R{rep+1}\n'+('Primed' if cached else 'Unseen'))
    bp=ax.boxplot(vals,tick_labels=labels,patch_artist=True,showfliers=True)
    for box,color in zip(bp['boxes'],['#BBBBBB','#C41230']*3):box.set_facecolor(color)
    ax.set_title(f'{mode.capitalize()} salt: 40 held-out images per class per repetition')
    ax.set_ylabel('First-content latency (ms)');ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(p/'detection_latency.png',dpi=170);plt.close(fig)
# Matched image/repetition comparison across separate performance sessions.
perf={m:{(r['rep'],r['image_id']):r['first_content_ms'] for r in rows if r.get('phase')=='cost_probe' and r['role']=='cross_tenant' and r['mode']==m} for m in ['shared','distinct']}
keys=sorted(perf['shared']);assert set(keys)==set(perf['distinct'])
delta=np.array([perf['distinct'][k]-perf['shared'][k] for k in keys]); rng=np.random.default_rng(797)
reps=sorted({k[0] for k in keys}); ids=sorted({k[1] for k in keys}); boot=[]
for _ in range(1000):
    rr=rng.choice(reps,len(reps),replace=True);ii=rng.choice(ids,len(ids),replace=True)
    boot.append(np.median([perf['distinct'][(int(r),i)]-perf['shared'][(int(r),i)] for r in rr for i in ii]))
comp={'matched_distinct_minus_shared_cross_tenant_ms':float(np.median(delta)),'ci95_ms':np.percentile(boot,[2.5,97.5]).tolist(),'interpretation':'Separate randomized sessions. A near-zero difference does not establish equivalence.'}
(p/'salt_latency_comparison.json').write_text(json.dumps(comp,indent=2))
text=f'''# Ruilin 的多模态隔离实验结果

正式运行：`{p.name}`。三轮已完成，7 项技术检查全部通过，1704 个请求无失败。30 个独立服务器会话。A100-SXM4-40GB 已停止。

## 可以交付的结论

在 vLLM 0.11.0、Qwen2.5-VL-3B、当前处理配置下，不同 `cache_salt` 抑制了图片相关 KV 前缀命中，但没有消除已知图片的成员检测信号。重复图片仍有明显延迟收益。这个结论只适用于这里的配置和数据，不是对所有服务的审计。

## 检测结果

40 张公开叶片照片用于评估，8 张另外用于校准。每张评估图在每轮两个全新服务器中各出现一次，一次被受害者预提交、一次未提交。每个服务器会话内攻击者只探测同一候选一次。三轮复用了相同图片，不能算作 120 张不同图片。

| Salt | 三轮 AUC | AP | 配对未提交−已提交延迟 | 95% bootstrap 区间 | 校准阈值 TPR | FPR |
|---|---|---|---|---|---|---|
'''
for mode in ['shared','distinct']:
    d=a['detection'][mode];lo,hi=d['ci95']['paired_delta_ms']
    text+=f"| {mode} | {', '.join(str(v['roc_auc']) for v in d['per_rep'])} | {d['average_precision']:.3f} | {d['paired_delta_ms']:.2f} ms | [{lo:.2f}, {hi:.2f}] ms | {d['tpr']:.1%} | {d['fpr']:.1%} |\n"
text+='''
AUC 衡量排序分离，TPR/FPR 是校准图片上选定阈值的实际检测表现。阈值规则偏保守，评估中有漏报。不能用 AUC=1 代替 100% 检出率。经验 bootstrap 的 AUC 区间 [1,1] 反映本样本的完美排序，不保证总体完美。只有三轮，服务器会话的不确定性估计有限。

KV 命中 token 中位数：共享 salt 的已提交图片 160、未提交图片 16；不同 salt 两组均为 16。16 是共同模板前缀的基线，不是图片 KV 的跨租户命中。计数在完整 SSE 结束后读取，首个非空内容比首个 chunk 的中位时间晚约 0.44 ms。

## 延迟收益与防御成本

性能使用另外 12 张图片，各条件在独立服务器会话中测量。表中为首个非空内容延迟，单位 ms。

| Salt | 冷请求中位数 | 同租户复用 | 跨租户复用 | 配对跨租户节省 | 节省的 95% 区间 |
|---|---:|---:|---:|---:|---|
'''
for mode in ['shared','distinct']:
    d=a['performance'][mode];v=d['role_p50_ms'];lo,hi=d['paired_saving_ci95_ms']['cross_tenant']
    text+=f"| {mode} | {v['cold']:.2f} | {v['same_tenant']:.2f} | {v['cross_tenant']:.2f} | {d['cross_tenant_saving_paired_p50_ms']:.2f} | [{lo:.2f}, {hi:.2f}] |\n"
lo,hi=comp['ci95_ms'];d=comp['matched_distinct_minus_shared_cross_tenant_ms']
text+=f'''
跨租户重复请求的不同 salt−共享 salt 配对差为 {d:.2f} ms，bootstrap 95% 区间 [{lo:.2f}, {hi:.2f}] ms。这些独立会话未显示明确的大额额外延迟，但没有预设等效性界限，不能声称“隔离完全免费”。没有实际测吞吐，不能声称得到吞吐成本。

## 机制证据与局限

运行时 KV 计数支持“不同 salt 下没有额外图片 KV 命中，同时延迟信号仍在”。`static_source_evidence.json` 保存了本次已安装 vLLM 的相关源码行与 SHA：KV 路径加入 cache_salt，多模态 processor 路径按多模态 item hash 查询缓存。静态代码支持机制解释，但不是逐请求 processor/encoder 命中的直接观测。直接观测补丁未验证，没有安装到正式计时进程。

执行期间做过简短的只读硬件/源码检查，未打 observer patch；这些辅助进程可能造成短暂系统扰动。计时在 localhost，单客户端串行，没有模拟生产并发和公网抖动。

数据仅一个自然图片领域，一个 VLM、vLLM 版本和设备。攻击者预先知道候选图片；结果是成员检测，没有恢复未知图片，也没有估算泄露多少比特。旧文字/音频结果分开保留，不能混入新统计。

## 环境与费用

vLLM 0.11.0，transformers 4.55.2，torch 2.8.0+cu128，CUDA 12.8，模型 revision `{meta['model_cache_revision']}`。完整 flags、source 快照、逐请求原始数据、图片来源/hash、硬件记录均在运行目录。

正式函数用时 {(meta['end_epoch']-meta['start_epoch'])/60:.2f} 分钟。按公开 A10040GB+2CPU+32GiB 价格、假定全额资源计费，正式运行资源估算约 ${(meta['end_epoch']-meta['start_epoch'])*(.000583+2*.0000131+32*.00000222):.2f}。这不是账单实付金额，不包含之前 L4 尝试、pilot、构建及长期存储。保守预算账本预留总额 $21.70，也不是消费金额。没有继续启动收费实验。

## 与 Evan 的交付衔接

Evan 负责文字内容恢复与共享计算资源争用；这里补充多模态缓存隔离边界和复用收益。合并时使用独立实验目录与结果说明，注明双方 vLLM/硬件/模型/计时口径的不同。不要把不同栈的数据混成一条对照曲线。建议最终汇报以“KV salting 的覆盖范围有限”为共同防御讨论，再分别展示恢复、成员检测和争用泄露的证据。

核心 A/B/C 已完成。D 的直接命中观测留作可选后续，不影响当前交付。下一步优先整理图表并集成到项目，而不是扩大模型或重跑音频。
'''
(root/'docs/FINAL_RESULTS_zh.md').write_text(text)
print(root/'docs/FINAL_RESULTS_zh.md')

"""Pure protocol logic: allocation, streaming timing, and metrics parsing.

Adapted conceptually from legacy/modal_exp3_ttft.py; legacy is kept unmodified.
No remote service or GPU is initialized on import.
"""
import json
import random
import time


def assignment(items, rep, pass_index, seed):
    """Balanced labels, complementary across two fresh-server passes.

    Every image is observed once cold and once victim-primed per mode/repetition.
    Each pass has only one attacker probe per image. Shuffle order independently.
    """
    if len(items) % 2:
        raise ValueError("An even number of images is required")
    shuffled = list(items)
    random.Random(seed + rep * 100).shuffle(shuffled)
    labels = {x['id']: int(i < len(shuffled) // 2) ^ pass_index
              for i, x in enumerate(shuffled)}
    order = list(items)
    random.Random(seed + rep * 100 + pass_index + 5000).shuffle(order)
    return [(x, labels[x['id']]) for x in order]


def parse_metrics(text):
    totals = {}
    for name in ('prefix_cache_hits', 'prefix_cache_queries'):
        vals = []
        for line in text.splitlines():
            if not line or line.startswith('#'):
                continue
            metric, value, *_ = line.split()
            bare = metric.split('{')[0]
            if bare in ('vllm:' + name, 'vllm:' + name + '_total'):
                vals.append(float(value))
        totals[name] = sum(vals) if vals else None
    return totals


def consume_sse(lines, started, clock=time.perf_counter):
    """Drain the entire response; distinguish role-only and content chunks.

    Tests inject a clock. A truncated stream, no content, or missing DONE is an
    error. The caller closes HTTP and reads counters AFTER full stream completion.
    """
    first_chunk = first_content = None
    usage = None
    done = False
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        if not line.startswith('data:'):
            continue
        payload = line[5:].strip()
        if payload == '[DONE]':
            done = True
            break
        obj = json.loads(payload)
        if 'error' in obj:
            raise RuntimeError(str(obj['error']))
        now = clock()
        if first_chunk is None:
            first_chunk = now
        if obj.get('usage'):
            usage = obj['usage']
        for choice in obj.get('choices', []):
            if choice.get('delta', {}).get('content') and first_content is None:
                first_content = now
    completed = clock()
    if not done or first_content is None:
        raise RuntimeError('Stream incomplete or no nonempty content token received')
    return {'first_chunk_ms': (first_chunk - started) * 1000,
            'first_content_ms': (first_content - started) * 1000,
            'completion_ms': (completed - started) * 1000, 'usage': usage}

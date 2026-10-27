"""Sequential self-hosted vLLM experiment. Called inside one Modal L4 function.

Do not infer encoder hits from KV counters. Instrumented files are not applied in
the timing run. No recovery, candidate search, or contention workload is included.
"""
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import time
import uuid
import requests
from src.protocol import assignment, consume_sse, parse_metrics

BASE = 'http://127.0.0.1:8000'
PROMPT = 'Describe the visible objects briefly.'


class Study:
    def __init__(self, cfg, dataset_root, out, checkpoint=lambda: None):
        self.cfg, self.root, self.out = cfg, Path(dataset_root), Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.deadline = time.monotonic() + cfg['max_wall_seconds']
        self.checkpoint = checkpoint
        self.rows = self.out / 'requests.jsonl'
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.items = self.manifest['images']
        self.proc = None
        self.request_seq = 0
        self.invalid = 0
        (self.out / 'config.json').write_text(json.dumps(cfg, indent=2))
        (self.out / 'dataset_manifest.json').write_text(json.dumps(self.manifest, indent=2))

    def guard(self):
        if time.monotonic() > self.deadline:
            raise TimeoutError('Study wall-time limit reached; stop without automatic retries')

    def event(self, record):
        with self.rows.open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')

    def start(self, session):
        self.guard()
        self.stop()
        log = self.out / ('server-' + session + '.log')
        flags = ['--enable-prefix-caching', '--port', '8000', '--block-size', '16',
                 '--max-model-len', str(self.cfg['max_model_len']),
                 '--max-num-batched-tokens', str(self.cfg['max_num_batched_tokens']),
                 '--max-num-seqs', str(self.cfg['max_num_seqs']), '--enforce-eager',
                 '--gpu-memory-utilization', str(self.cfg['gpu_memory_utilization']),
                 '--mm-processor-cache-gb', str(self.cfg['processor_cache_gb'])]
        self.command = ['vllm', 'serve', self.cfg['model'], *flags]
        self.log_handle = log.open('wb')
        self.proc = subprocess.Popen(self.command, stdout=self.log_handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        startup = time.monotonic() + self.cfg['startup_timeout_seconds']
        while time.monotonic() < startup:
            self.guard()
            if self.proc.poll() is not None:
                raise RuntimeError('Server exited. Inspect ' + str(log))
            try:
                if requests.get(BASE + '/health', timeout=2).status_code == 200:
                    self.event({'kind': 'server_ready', 'session': session,
                                'command': self.command, 'epoch': time.time()})
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError('Server startup timed out; no fallback GPU or retry')

    def stop(self):
        if self.proc is not None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=15)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=5)
                except ProcessLookupError:
                    pass
            self.proc = None
            self.log_handle.close()

    def metrics(self):
        try:
            r = requests.get(BASE + '/metrics', timeout=5)
            r.raise_for_status()
            return parse_metrics(r.text)
        except requests.RequestException:
            return {'prefix_cache_hits': None, 'prefix_cache_queries': None}

    def request(self, image, salt, **tags):
        self.guard()
        path = self.root / image['path']
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != image['sha256']:
            raise ValueError('Dataset image hash changed: ' + image['id'])
        url = 'data:image/png;base64,' + base64.b64encode(content).decode()
        payload = {'model': self.cfg['model'], 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': PROMPT},
            {'type': 'image_url', 'image_url': {'url': url}}]}],
            'max_tokens': 1, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True}, 'cache_salt': salt}
        before = self.metrics()
        self.request_seq += 1
        record = {'kind': 'request', 'seq': self.request_seq, 'image_id': image['id'],
                  'image_sha256': image['sha256'], 'salt': salt, **tags}
        started = time.perf_counter()
        try:
            with requests.post(BASE + '/v1/chat/completions', json=payload, stream=True,
                               timeout=self.cfg['request_timeout_seconds']) as resp:
                resp.raise_for_status()
                # chunk_size=1 avoids requests' default 512-byte buffering of SSE.
                record.update(consume_sse(resp.iter_lines(chunk_size=1), started))
            after = self.metrics()
            until = time.monotonic() + self.cfg['metrics_wait_seconds']
            q0 = before['prefix_cache_queries']
            while q0 is not None and after['prefix_cache_queries'] is not None and after['prefix_cache_queries'] <= q0 and time.monotonic() < until:
                time.sleep(.1)
                after = self.metrics()
            record['kv_before'], record['kv_after'] = before, after
            valid = all(v is not None for v in (*before.values(), *after.values()))
            valid = valid and after['prefix_cache_queries'] > before['prefix_cache_queries']
            record['kv_counter_status'] = 'observed_delta' if valid else 'missing_or_delayed'
            record['kv_hits_delta'] = (after['prefix_cache_hits'] - before['prefix_cache_hits']) if valid else None
            record['kv_queries_delta'] = (after['prefix_cache_queries'] - before['prefix_cache_queries']) if valid else None
            record['ok'] = True
        except Exception as exc:
            record.update(ok=False, error=repr(exc))
            self.invalid += 1
            self.event(record)
            raise
        self.event(record)
        return record

    def warmup(self, rep, mode, session):
        # Not used in calibration/evaluation/performance. Warm execution paths and
        # BOTH tenant namespaces symmetrically before testing any candidate image.
        image = next(i for i in self.items if i['split'] == 'warmup')
        v, a = self.salts(mode, session)
        for salt in [v, a, v, a]:
            self.request(image, salt, phase='warmup', rep=rep, mode=mode, session=session)

    @staticmethod
    def salts(mode, session):
        return (session + '-victim', session + ('-victim' if mode == 'shared' else '-attacker'))

    def attack(self, rep, mode, pass_index):
        session = f'r{rep}-{mode}-attack-p{pass_index}'
        self.start(session)
        try:
            self.warmup(rep, mode, session)
            vs, ats = self.salts(mode, session)
            for split, count in [('calibration', self.cfg['calibration_images']),
                                 ('evaluation', self.cfg['evaluation_images'])]:
                items = [x for x in self.items if x['split'] == split][:count]
                if len(items) != count:
                    raise ValueError('Not enough ' + split + ' images')
                for item, cached in assignment(items, rep, pass_index, self.cfg['seed']):
                    tags = dict(rep=rep, mode=mode, session=session, pass_index=pass_index,
                                split=split, victim_primed=cached)
                    if cached:
                        self.request(item, vs, phase='victim_prime', **tags)
                    else:
                        # Matched sham victim request to equalize activity/order.
                        warm = next(x for x in self.items if x['split'] == 'warmup')
                        self.request(warm, vs, phase='sham_prime', **tags)
                    # One attacker request to this image in this server session.
                    self.request(item, ats, phase='attack_probe', **tags)
            self.checkpoint()
        finally:
            self.stop()

    def cost(self, rep, mode, role):
        # Separate fresh sessions for cold/same-tenant/cross-tenant, with identical
        # image sets. Same-tenant probes can never warm the cross-tenant condition.
        session = f'r{rep}-{mode}-cost-{role}'
        self.start(session)
        try:
            self.warmup(rep, mode, session)
            vs, ats = self.salts(mode, session)
            items = [x for x in self.items if x['split'] == 'performance'][:self.cfg['cost_images']]
            if len(items) != self.cfg['cost_images']:
                raise ValueError('Not enough performance images')
            random.Random(self.cfg['seed'] + rep).shuffle(items)
            for item in items:
                tags = dict(rep=rep, mode=mode, session=session, role=role, split='performance')
                if role != 'cold':
                    self.request(item, vs, phase='cost_prime', **tags)
                else:
                    warm = next(x for x in self.items if x['split'] == 'warmup')
                    self.request(warm, vs, phase='cost_sham', **tags)
                self.request(item, vs if role == 'same_tenant' else ats,
                             phase='cost_probe', **tags)
            self.checkpoint()
        finally:
            self.stop()

    def run(self):
        snapshot = self.out / 'source'
        snapshot.mkdir()
        for source in Path(__file__).parent.glob('*.py'):
            shutil.copy2(source, snapshot / source.name)
        meta = {'schema_version': 1, 'versions': {p: importlib.metadata.version(p)
                for p in ['vllm', 'transformers', 'requests']},
                'start_epoch': time.time(), 'timing_metric': 'first_content_ms',
                'instrumented': False, 'scope': 'image_membership_and_reuse_only'}
        (self.out / 'meta.json').write_text(json.dumps(meta, indent=2))
        try:
            if self.cfg.get('instrumentation'):
                raise ValueError('Instrumented timing is forbidden in this runner')
            for rep in range(self.cfg['repetitions']):
                modes = list(self.cfg['salt_modes'])
                random.Random(self.cfg['seed'] + rep).shuffle(modes)
                for mode in modes:
                    if 'attack' in self.cfg['phases']:
                        passes = [0, 1]
                        random.Random(self.cfg['seed'] + rep + 99).shuffle(passes)
                        for pi in passes:
                            self.attack(rep, mode, pi)
                    if 'cost' in self.cfg['phases']:
                        roles = ['cold', 'same_tenant', 'cross_tenant']
                        random.Random(self.cfg['seed'] + rep + 77).shuffle(roles)
                        for role in roles:
                            self.cost(rep, mode, role)
            meta['status'] = 'complete'
        except Exception as exc:
            meta.update(status='failed', error=repr(exc))
            raise
        finally:
            self.stop()
            ref = Path(os.environ.get('HF_HOME', '/root/.cache/huggingface')) / 'hub' / ('models--' + self.cfg['model'].replace('/', '--')) / 'refs/main'
            meta['model_cache_revision'] = ref.read_text().strip() if ref.exists() else None
            meta['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob('*.py')}
            meta.update(end_epoch=time.time(), requests=self.request_seq, invalid=self.invalid)
            (self.out / 'meta.json').write_text(json.dumps(meta, indent=2))
            self.checkpoint()
        return meta

"""One explicit A100 allocation. No GPU use on import; no automatic retries.

Run through scripts/run_stage.py so a cumulative budget reservation is recorded.
"""
import json
from pathlib import Path
import time
import uuid
import modal

ROOT = Path(__file__).resolve().parent
app = modal.App('pcache-multimodal-isolation')
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('vllm[audio]==0.11.0', 'transformers==4.55.2',
                      'openai>=1.99.1,<2', 'numpy', 'scipy', 'requests', 'pillow', 'soundfile')
         .add_local_dir(ROOT / 'src', '/root/study/src')
         .add_local_dir(ROOT / 'data', '/root/study/data'))
# Reuse the existing pcache-out volume, under a NEW namespace. Old files untouched.
volume = modal.Volume.from_name('pcache-out', create_if_missing=False)


@app.function(image=image, gpu='A100', cpu=2, memory=32768,
              volumes={'/out': volume}, timeout=7500, retries=0,
              max_containers=1, scaledown_window=2)
def execute(cfg):
    import sys
    sys.path.insert(0, '/root/study')
    from src.runner import Study
    # Persist downloaded weights across sequential sessions/runs; no private token.
    import os
    os.environ['HF_HOME'] = '/out/multimodal_starter/model_cache'
    run_id = cfg['stage'] + '-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    target = Path('/out/multimodal_starter/runs') / run_id
    study = Study(cfg, '/root/study/data', target, checkpoint=volume.commit)
    try:
        study.run()
    finally:
        volume.commit()
    return {'remote_run_dir': str(target), 'run_id': run_id}


@app.local_entrypoint()
def main(config: str = 'configs/pilot.json', receipt: str = 'results/last_receipt.json'):
    cfg = json.loads(Path(config).read_text())
    if cfg['gpu'] != 'A100' or cfg.get('instrumentation'):
        raise ValueError('This entrypoint only supports uninstrumented A100 timing runs')
    if cfg['max_wall_seconds'] > 7200:
        raise ValueError('Maximum study wall time is two hours per invocation')
    result = execute.remote(cfg)
    dest = Path(receipt)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

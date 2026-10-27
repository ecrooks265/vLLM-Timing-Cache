#!/usr/bin/env python3
"""Reserve budget BEFORE any Modal launch, serialize the full remote invocation.

Reservations are never automatically refunded (including failed launches).
This is a conservative local guard, NOT a provider billing cap. It excludes other
projects/account spend. See docs/BUDGET.md for the exact limits.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def reservation(cfg):
    # Remote function timeout includes cold model loading; reserve the full 7500s,
    # not just the smaller experiment deadline. Extra $1 covers CPU image build.
    return round(7500 / 3600 * cfg['assumed_all_in_usd_per_hour'] + 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['pilot', 'main'])
    ap.add_argument('--check', action='store_true', help='No launch or reservation')
    ap.add_argument('--wait-for-gpu', action='store_true', help='Keep waiting for L4 capacity; user-requested queue wait, compute limits unchanged')
    args = ap.parse_args()
    cfg_path = ROOT / 'configs' / (args.stage + '.json')
    cfg = json.loads(cfg_path.read_text())
    if cfg['budget_cap_usd'] > 40 or cfg['assumed_all_in_usd_per_hour'] < 1.25:
        raise ValueError('Budget cap must be <= $40; conservative rate >= $1.25/hour')
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    with (ROOT / '.budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = results / 'budget_ledger.json'
        entries = json.loads(ledger.read_text()) if ledger.exists() else []
        total = sum(x['reserved_usd'] for x in entries)
        amount = reservation(cfg)
        print(json.dumps({'previous_reserved_usd': total, 'next_reserved_usd': amount,
                          'after_usd': round(total + amount, 2), 'cap_usd': 40}, indent=2))
        if total + amount > 40:
            raise RuntimeError('Local cumulative budget exhausted; do not launch')
        if args.check:
            return
        if not (ROOT / 'data/manifest.json').exists():
            raise RuntimeError('First run scripts/prepare_images.py')
        if args.stage == 'main':
            gate = results / 'pilot_gate.json'
            if not gate.exists() or not json.loads(gate.read_text()).get('pass'):
                raise RuntimeError('Analyze the pilot first; pilot_gate.json must pass')
        modal_cli = shutil.which('modal')
        if not modal_cli:
            raise RuntimeError('modal CLI missing; install requirements in your venv')
        ident = str(int(time.time()))
        receipt = results / ('receipt-' + args.stage + '-' + ident + '.json')
        entry = {'id': ident, 'stage': args.stage, 'reserved_usd': amount,
                 'rate_assumption': cfg['assumed_all_in_usd_per_hour'],
                 'epoch': time.time(), 'status': 'reserved', 'receipt': receipt.name}
        entries.append(entry)
        ledger.write_text(json.dumps(entries, indent=2))
        cmd = [modal_cli, 'run', str(ROOT / 'modal_study.py'), '--config', str(cfg_path),
               '--receipt', str(receipt)]
        try:
            with (results / ('launch-' + ident + '.log')).open('w') as log:
                p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                print('Follow log:', log.name, flush=True)
                queued_since = time.monotonic()
                queue_checked = False
                try:
                    while p.poll() is None:
                        # A bounded queue check, not a repeated application poll.
                        if not args.wait_for_gpu and not queue_checked and time.monotonic()-queued_since > 300:
                            queue_checked = True
                            contents = Path(log.name).read_text(errors='replace')
                            ids = re.findall(r'\bap-[A-Za-z0-9]+\b', contents)
                            if ids and 'remote_run_dir' not in contents:
                                snapshot = subprocess.run([modal_cli, 'app', 'list', '--json'], capture_output=True, text=True)
                                if snapshot.returncode == 0:
                                    apps = json.loads(snapshot.stdout)
                                    target = next((a for a in apps if a.get('App ID') == ids[0]), None)
                                    if target and str(target.get('Tasks')) == '0':
                                        subprocess.run([modal_cli, 'app', 'stop', ids[0]], check=True)
                                        print('No allocated task after five minutes; stopped only this study app.', flush=True)
                                        p.send_signal(signal.SIGINT)
                        time.sleep(1)
                    code = p.returncode
                except KeyboardInterrupt:
                    p.send_signal(signal.SIGINT)
                    p.wait(timeout=30)
                    entry['status'] = 'interrupted'
                    raise
            entry['status'] = 'completed' if code == 0 else 'failed'
            entry['exit_code'] = code
            if code == 0:
                result = json.loads(receipt.read_text())
                remote = 'multimodal_starter/runs/' + result['run_id']
                dest = results / 'runs' / result['run_id']
                dest.mkdir(parents=True, exist_ok=True)
                subprocess.run([modal_cli, 'volume', 'get', 'pcache-out', remote + '/', str(dest)], check=True)
                nested = dest / result['run_id']
                if nested.is_dir() and not (dest / 'config.json').exists():
                    for child in list(nested.iterdir()):
                        child.rename(dest / child.name)
                    nested.rmdir()
                print('Downloaded:', dest)
            else:
                raise SystemExit(code)
        finally:
            ledger.write_text(json.dumps(entries, indent=2))


if __name__ == '__main__':
    main()

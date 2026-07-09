"""Lambda Cloud driver: launch GPU -> run the 8B probe rung -> fetch results -> terminate.

Fully autonomous given LAMBDA_API_KEY in the environment (optionally HF_TOKEN for the
gated meta-llama/Llama-3.1-8B-Instruct; otherwise runs ungated Qwen2.5-7B-Instruct).

    python lambda_driver.py            # full lifecycle, ~1-2h, ~$1-3 on an A10
    python lambda_driver.py --dry-run  # show instance choice + plan, launch nothing

Safety: instance id + console URL are printed immediately after launch; termination
runs in a finally block; if this process dies anyway, kill the instance at
https://cloud.lambdalabs.com/instances. Never leaves files with secrets on disk.
"""
import argparse, json, os, subprocess, sys, time, urllib.request

API = 'https://cloud.lambdalabs.com/api/v1'
KEY_PATH = os.path.expanduser('~/.ssh/lambda_semdiff')
PREFERRED = ['gpu_1x_a10', 'gpu_1x_a100_sxm4', 'gpu_1x_a100', 'gpu_1x_gh200',
             'gpu_1x_h100_pcie']
PUSH = ['laws.py', 'magma.py', 'dataset.py', 'pilot.py', 'verify_etp.py',
        'hf_backend.py', 'probes.py', 'preflight.py', 'data.jsonl', 'data-etp.jsonl']
PULL_GLOBS = ['results-hf-*.jsonl', 'acts-hf-*.npz', 'probe-report-hf-*.md',
              'steer-*.json*', 'overopt-*.jsonl', 'overopt-*.npz',
              'results-geo.jsonl', 'acts-geo.npz', 'acts-steer-*.npz']

def api(path, payload=None):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Authorization': f"Bearer {os.environ['LAMBDA_API_KEY']}",
                 'Content-Type': 'application/json',
                 'User-Agent': 'semdiff-pilot/1.0'},  # urllib default UA gets 403'd
        method='POST' if payload is not None else 'GET')
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())['data']

STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lambda_state.json')

def _state():
    try:
        return json.load(open(STATE))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _record(iid, action):
    s = _state()
    if action == 'launch':
        s[iid] = time.time()
    else:
        s.pop(iid, None)
    json.dump(s, open(STATE, 'w'))

def reap(ttl_hours=3.0, force=False):
    """Terminate semdiff-named instances: recorded ones past TTL, and (with force)
    ALL semdiff-named ones. Orphan insurance for when a driver process dies before
    its finally-block. Run: python lambda_driver.py --reap [--force]"""
    state = _state()
    victims = []
    for inst in api('/instances'):
        if not inst.get('name', '').startswith('semdiff'):
            continue  # never touch instances we did not name
        if inst.get('status') in ('terminating', 'terminated'):
            continue  # already dying; listing it as an orphan is just noise
        age_h = (time.time() - state.get(inst['id'], time.time())) / 3600
        known = inst['id'] in state
        if force or (known and age_h > ttl_hours):
            victims.append((inst['id'], inst.get('name'), f'{age_h:.1f}h' if known else 'unrecorded'))
        elif not known:  # possibly a live driver from before state-tracking: report only
            print(f'UNRECORDED semdiff instance {inst["id"]} — reap with --force if orphaned')
    for iid, name, age in victims:
        api('/instance-operations/terminate', {'instance_ids': [iid]})
        _record(iid, 'terminate')
        print(f'REAPED {iid} ({name}, {age})')
    if not victims:
        print('no orphaned semdiff instances')
    return victims

def ensure_keypair():
    if not os.path.exists(KEY_PATH):
        subprocess.run(['ssh-keygen', '-t', 'ed25519', '-N', '', '-f', KEY_PATH,
                        '-C', 'semdiff-lambda'], check=True, capture_output=True)
    pub = open(KEY_PATH + '.pub').read().strip()
    names = {k['name']: k.get('public_key', '') for k in api('/ssh-keys')}
    for name, existing in names.items():
        if existing.split()[:2] == pub.split()[:2]:
            return name
    name = f'semdiff-{int(time.time())}'
    api('/ssh-keys', {'name': name, 'public_key': pub})
    return name

def pick_instance():
    types = api('/instance-types')
    for t in PREFERRED:
        info = types.get(t)
        if info and info['regions_with_capacity_available']:
            region = info['regions_with_capacity_available'][0]['name']
            price = info['instance_type']['price_cents_per_hour'] / 100
            return t, region, price
    raise SystemExit('no preferred instance type has capacity right now — retry later')

def ssh(ip, cmd, timeout=3600, get_output=False):
    base = ['ssh', '-i', KEY_PATH, '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null', '-o', 'ConnectTimeout=15',
            f'ubuntu@{ip}', cmd]
    r = subprocess.run(base, timeout=timeout, capture_output=True, text=True)
    if get_output:
        return r.returncode, r.stdout + r.stderr
    def redact(s):
        tok = os.environ.get('HF_TOKEN')
        return s.replace(tok, 'HF_TOKEN***') if tok and s else s
    print(redact(r.stdout[-2000:]) if r.stdout else '',
          redact(r.stderr[-1000:]) if r.returncode else '', flush=True)
    if r.returncode:
        raise RuntimeError(f'remote command failed: {redact(cmd)[:80]}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--reap', action='store_true', help='terminate orphaned semdiff instances')
    ap.add_argument('--force', action='store_true', help='with --reap: kill ALL semdiff instances')
    ap.add_argument('--push', nargs='*', default=[], help='extra files to ship up')
    ap.add_argument('--run', nargs='*', default=None,
                    help='remote scripts (venv python) run in order INSTEAD of the model loop')
    args = ap.parse_args()
    if not os.environ.get('LAMBDA_API_KEY'):
        sys.exit('LAMBDA_API_KEY not set')
    if args.reap:
        reap(force=args.force)
        return
    reap()  # launch-time orphan sweep (recorded + past TTL only; never live drivers)
    models = args.models or (['meta-llama/Llama-3.1-8B-Instruct']
                             if os.environ.get('HF_TOKEN') else
                             ['Qwen/Qwen2.5-7B-Instruct'])
    itype, region, price = pick_instance()
    print(f'instance: {itype} in {region} (${price:.2f}/h) | models: {models}')
    if args.dry_run:
        return
    key_name = ensure_keypair()
    ids = api('/instance-operations/launch',
              {'region_name': region, 'instance_type_name': itype,
               'ssh_key_names': [key_name], 'name': 'semdiff-8b-probe'})['instance_ids']
    iid = ids[0]
    _record(iid, 'launch')
    print(f'LAUNCHED {iid} — manual kill: https://cloud.lambdalabs.com/instances', flush=True)
    try:
        ip = None
        for _ in range(90):  # up to 15 min to boot
            inst = api(f'/instances/{iid}')
            if inst['status'] == 'active' and inst.get('ip'):
                ip = inst['ip']
                break
            time.sleep(10)
        if not ip:
            raise RuntimeError('instance never became active')
        print(f'active at {ip}', flush=True)
        for _ in range(30):  # wait for sshd
            code, _out = ssh(ip, 'true', get_output=True)
            if code == 0:
                break
            time.sleep(10)
        subprocess.run(['scp', '-i', KEY_PATH, '-o', 'StrictHostKeyChecking=no',
                        '-o', 'UserKnownHostsFile=/dev/null', *PUSH, *args.push,
                        f'ubuntu@{ip}:~/'], check=True, capture_output=True)
        # Keep the image's driver-matched torch (upgrading torch breaks CUDA against
        # older drivers); pair it with transformers 4.x (5.x needs newer torch).
        # Preflight before paying for any model download.
        # Clean venv, zero system-site packages: the image's apt Python packages
        # (Pillow 8, numpy 1.x ABI) poisoned every mixed-stack attempt (runs 2-7).
        # One consistent pip set; torch from the cu121 index matches the drivers.
        ssh(ip, 'python3 -m venv ~/venv && ~/venv/bin/pip install -q -U pip '
                '2>&1 | tail -1', timeout=600)
        ssh(ip, '~/venv/bin/pip install -q torch '
                '--index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1',
            timeout=1200)
        ssh(ip, '~/venv/bin/pip install -q "transformers<5" numpy pillow accelerate '
                'scikit-learn 2>&1 | tail -1', timeout=900)
        ssh(ip, '~/venv/bin/python preflight.py', timeout=300)
        hf = f"HF_TOKEN={os.environ['HF_TOKEN']} " if os.environ.get('HF_TOKEN') else ''
        def pull():
            for g in PULL_GLOBS:
                subprocess.run(f'scp -i "{KEY_PATH}" -o StrictHostKeyChecking=no '
                               f'-o UserKnownHostsFile=/dev/null "ubuntu@{ip}:~/{g}" .',
                               shell=True, capture_output=True)
        if args.run:
            for cmd in args.run:
                print(f'--- {cmd} ---', flush=True)
                try:
                    ssh(ip, f'{hf}~/venv/bin/python {cmd}', timeout=14400)
                finally:
                    pull()
            return
        failures = []
        for m in models:
            print(f'--- running {m} ---', flush=True)
            try:
                ssh(ip, f'{hf}~/venv/bin/python hf_backend.py {m}', timeout=5400)
                ssh(ip, '~/venv/bin/python probes.py', timeout=1800)  # latest acts = this model
            except Exception as e:  # one model's crash must not abort its siblings
                failures.append((m, str(e)[:120]))
                print(f'MODEL FAILED (continuing): {m}: {e}', flush=True)
            pull()  # after EVERY model: a later crash must not cost earlier results
            print(f'results pulled after {m}', flush=True)
        if failures:
            raise RuntimeError(f'{len(failures)}/{len(models)} models failed: {failures}')
    finally:
        api('/instance-operations/terminate', {'instance_ids': [iid]})
        _record(iid, 'terminate')
        print(f'TERMINATED {iid} — cross-check anytime: python lambda_driver.py --reap',
              flush=True)

if __name__ == '__main__':
    main()

"""Rebuilds the NHL betting simulator page.

Run by GitHub Actions on a schedule (see .github/workflows/refresh.yml):
  1. fetch the saved matchup inputs from the n8n endpoint (n8n pulls odds, lineups and stats on its own schedule)
  2. simulate every game that is new or whose inputs changed (other games come from the cache in state/)
  3. add line movement, save the pre-game predictions, and build site/index.html
The cache, price history and predictions live on a separate branch called "state" (one commit, force-pushed), so the repo history stays small."""
import os, sys, json, time, gzip, pickle, shutil, subprocess, urllib.request, urllib.parse, zlib
from datetime import datetime, timezone, timedelta
from concurrent.futures import ProcessPoolExecutor
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'engine'))
STATE = os.environ.get('STATE_DIR', os.path.join(HERE, 'state'))
SITE = os.environ.get('SITE_DIR', os.path.join(HERE, 'site'))
ET = ZoneInfo('America/New_York')
ROLLOVER_HOUR = 6          # the slate changes to the next day at 6 AM Eastern, so a late game that ends after midnight stays on its own day's page
PRED_DAYS = 14             # saved pre-game predictions are kept this long (for grading)
WORKERS = int(os.environ.get('SIM_WORKERS', '2'))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def now():
    o = os.environ.get('NOW_OVERRIDE')          # for testing
    return datetime.fromisoformat(o.replace('Z', '+00:00')) if o else datetime.now(timezone.utc)


def slate_date():
    """The day whose games the page shows: the Eastern date, rolled over at 6 AM."""
    if os.environ.get('SLATE_DATE'):
        return datetime.fromisoformat(os.environ['SLATE_DATE']).date()
    return (now().astimezone(ET) - timedelta(hours=ROLLOVER_HOUR)).date()


def game_day(p):
    return start_of(p).astimezone(ET).date()


def no_games_html(day, nxt):
    import tool_build
    when = day.strftime('%A, %B %-d')
    soon = ('The next games are on ' + nxt.strftime('%A, %B %-d') + '.') if nxt else 'Check back on the next game day.'
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>NHL Betting Simulator | RotoWire</title>
<link href="https://fonts.googleapis.com/css2?family=VT323&family=Press+Start+2P&display=swap" rel="stylesheet">
<style>body{margin:0;background:#06152C;color:#EDF1F7;font-family:'VT323','Courier New',monospace;font-size:20px;padding:24px 12px}
.shell{max-width:900px;margin:0 auto;border:4px solid #5C7DB0;box-shadow:8px 8px 0 #D84048;background:#06152C}
.hdr{padding:28px 26px 22px;background:linear-gradient(135deg,#0A2247 0%,#0A1830 60%,#06152C 100%);border-bottom:4px solid #D84048}
.hdr img{height:26px;width:auto;display:block;margin:0 0 14px}
h1{font-family:'Press Start 2P','Courier New',monospace;font-weight:400;font-size:20px;line-height:1.5;margin:0 0 12px;text-shadow:3px 3px 0 #031024}h1 span{color:#E86A72}
.b{padding:36px 26px}.b p{margin:0 0 12px;max-width:60ch;color:#93A4BD}.b strong{font-family:'Press Start 2P','Courier New',monospace;font-weight:400;font-size:12px;line-height:1.7;color:#EDF1F7;display:block;margin-bottom:14px}
</style></head><body><div class="shell"><div class="hdr"><img src="''' + tool_build.lockup_uri() + '''" alt="RotoWire"><h1>NHL Betting <span>Simulator</span></h1></div>
<div class="b"><strong>No NHL games on ''' + when + '''.</strong><p>''' + soon + '''</p><p>The simulator prices every bet for each game the day it is played.</p></div></div></body></html>'''


def start_of(p):
    return datetime.fromisoformat(str(p['fixture']['start']).replace('Z', '+00:00'))


# ---------------------------------------------------------------- state on its own branch
def remote():
    if os.environ.get('STATE_REMOTE'):
        return os.environ['STATE_REMOTE']
    return 'https://x-access-token:%s@github.com/%s.git' % (os.environ['GITHUB_TOKEN'], os.environ['GITHUB_REPOSITORY'])


def git(args, cwd=None):
    subprocess.run(['git'] + args, cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def restore_state():
    shutil.rmtree(STATE, ignore_errors=True)
    if os.environ.get('SKIP_GIT_STATE') == '1':
        os.makedirs(STATE, exist_ok=True); return
    url = remote()
    if subprocess.run(['git', 'ls-remote', '--exit-code', '--heads', url, 'state'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        git(['clone', '--quiet', '--depth', '1', '--branch', 'state', url, STATE])
        shutil.rmtree(os.path.join(STATE, '.git'), ignore_errors=True)
        log('state restored from the state branch')
    else:
        os.makedirs(STATE, exist_ok=True); log('no state branch yet; starting fresh')


def save_state():
    if os.environ.get('SKIP_GIT_STATE') == '1':
        return
    shutil.rmtree(os.path.join(STATE, '.git'), ignore_errors=True)
    git(['init', '--quiet', '-b', 'state'], cwd=STATE)
    git(['config', 'user.name', 'nhl-sim-bot'], cwd=STATE); git(['config', 'user.email', 'actions@users.noreply.github.com'], cwd=STATE)
    git(['add', '-A'], cwd=STATE); git(['commit', '--quiet', '-m', 'state ' + now().strftime('%Y-%m-%dT%H:%MZ')], cwd=STATE)
    git(['push', '--quiet', '--force', remote(), 'HEAD:refs/heads/state'], cwd=STATE)
    shutil.rmtree(os.path.join(STATE, '.git'), ignore_errors=True)
    log('state saved')


def sdir(name):
    d = os.path.join(STATE, name); os.makedirs(d, exist_ok=True); return d


def load_gz(path):
    with gzip.open(path, 'rb') as f:
        return pickle.load(f)


def save_gz(path, obj):
    with gzip.open(path, 'wb', compresslevel=6) as f:
        pickle.dump(obj, f, protocol=4)


# ---------------------------------------------------------------- inputs
def fetch_inputs():
    if os.environ.get('INPUTS_FILE'):                     # local testing
        return json.load(open(os.environ['INPUTS_FILE']))['games']
    url = os.environ['N8N_INPUTS_URL']; token = os.environ['N8N_INPUTS_TOKEN']
    req = urllib.request.Request(url + ('&' if '?' in url else '?') + 'token=' + urllib.parse.quote(token), headers={'User-Agent': 'nhl-sim-refresh/1'})
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.load(r)
    if body.get('error'):
        raise SystemExit('the inputs endpoint refused the token: ' + str(body['error']))
    return body['games']


# ---------------------------------------------------------------- one game (runs in a worker process)
def simulate(payload):
    try:
        import nhl_pipeline as pl, nhl_export as ex
        os.makedirs('/tmp/nhl_cache', exist_ok=True)
        t0 = time.time()
        res, bundle = pl.run_payload(payload, n=int(payload.get('sims') or 200000), cache_dir='/tmp/nhl_cache', verbose=False, return_bundle=True, price_keyed_cache=True)
        fid = res['fixtureId']
        pack = ex.export_game(payload, bundle, seed_idx=zlib.crc32(str(fid).encode()) % 10000)
        return {'fid': fid, 'res': res, 'pack': pack, 'seconds': round(time.time() - t0), 'error': None}
    except Exception as e:
        import traceback
        return {'fid': payload['fixture']['fixture_id'], 'res': None, 'pack': None, 'seconds': 0, 'error': traceback.format_exc()[-1500:]}


def main():
    import price_history as ph, tool_build
    force = os.environ.get('FORCE', '').lower() in ('1', 'true', 'yes')
    restore_state()
    day = slate_date()
    everything = fetch_inputs()
    games = [p for p in everything if game_day(p) == day]
    games.sort(key=start_of)
    log('slate day', day.isoformat(), '|', len(games), 'of', len(everything), 'games on it |', ', '.join(p['fixture']['away']['abbr'] + '@' + p['fixture']['home']['abbr'] for p in games))
    built_path0 = os.path.join(STATE, 'built.json')
    if not games:
        nxt = min([game_day(p) for p in everything if game_day(p) > day] or [None], key=lambda d: d or day)
        prev = json.load(open(built_path0)) if os.path.exists(built_path0) else {}
        if prev.get('games') == {'no-games': day.isoformat()} and not force:
            log('the no-games page is already up'); out('changed', 'false'); return
        shutil.rmtree(SITE, ignore_errors=True); os.makedirs(SITE)
        open(os.path.join(SITE, 'index.html'), 'w', encoding='utf-8').write(no_games_html(day, nxt))
        open(os.path.join(SITE, '.nojekyll'), 'w').write('')
        json.dump({'builtAt': now().isoformat(), 'games': {'no-games': day.isoformat()}, 'failed': []}, open(os.path.join(SITE, 'build.json'), 'w'))
        json.dump({'games': {'no-games': day.isoformat()}, 'builtAt': now().isoformat()}, open(built_path0, 'w'))
        prune([]); save_state(); log('built the no-games page for', day.isoformat()); out('changed', 'true'); return
    simdir, histdir, preddir = sdir('sims'), sdir('hist'), sdir('pred')
    built_path = os.path.join(STATE, 'built.json')
    built = json.load(open(built_path)) if os.path.exists(built_path) else {}
    todo, have = [], {}
    for p in games:
        fid = p['fixture']['fixture_id']; cp = os.path.join(simdir, fid + '.pkl.gz')
        if os.path.exists(cp) and not force:
            c = load_gz(cp)
            if c.get('gen') == p['generatedAt']:
                have[fid] = c; continue
        todo.append(p)
    log('cached:', len(have), '| to simulate:', len(todo), '| workers', WORKERS)
    failures = []
    if todo:
        with ProcessPoolExecutor(max_workers=max(1, min(WORKERS, len(todo)))) as ex_:
            for p, r in zip(todo, ex_.map(simulate, todo)):
                fid = p['fixture']['fixture_id']
                if r['error']:
                    failures.append(fid); log('FAILED', fid, '\n' + r['error']); continue
                pack = r['pack']; hp = os.path.join(histdir, fid + '.json')
                hist = ph.load(hp); n = ph.attach_moves(pack, hist); ph.save(hp, ph.record(hist, pack))
                if start_of(p) > now():                      # the last pull before puck drop is the prediction that gets graded
                    save_gz(os.path.join(preddir, fid + '.pkl.gz'), {'meta': pack['meta'], 'savedAt': time.time()})
                c = {'gen': p['generatedAt'], 'res': r['res'], 'pack': pack}
                save_gz(os.path.join(simdir, fid + '.pkl.gz'), c); have[fid] = c
                log('simulated %-10s %3ds | bets %d | with an earlier price %d' % (pack['meta']['id'], r['seconds'], len(pack['meta']['legs']), n))
    ids = [p['fixture']['fixture_id'] for p in games if p['fixture']['fixture_id'] in have]
    if not ids:
        raise SystemExit('no game could be simulated')
    signature = {fid: have[fid]['gen'] for fid in ids}
    if signature == built.get('games') and not force and not failures:
        log('nothing changed since the last build'); out('changed', 'false'); return
    packs = [{'meta': have[f]['pack']['meta'], 'cols': have[f]['pack']['cols']} for f in ids]
    replays = {have[f]['res']['id']: have[f]['res'] for f in ids}
    html = tool_build.build_html(packs, replays)
    shutil.rmtree(SITE, ignore_errors=True); os.makedirs(SITE)
    open(os.path.join(SITE, 'index.html'), 'w', encoding='utf-8').write(html)
    open(os.path.join(SITE, '.nojekyll'), 'w').write('')
    json.dump({'builtAt': now().isoformat(), 'games': signature, 'failed': failures}, open(os.path.join(SITE, 'build.json'), 'w'))
    json.dump({'games': signature, 'builtAt': now().isoformat()}, open(built_path, 'w'))
    log('built site/index.html: %.1f MB, %d games' % (len(html) / 1e6, len(ids)))
    prune(ids)
    save_state()
    out('changed', 'true')


def prune(keep_ids):
    """Drop cached sims and history for games that have left the page, and old predictions."""
    for name in ('sims', 'hist'):
        d = sdir(name)
        for fn in os.listdir(d):
            if fn.split('.')[0] not in keep_ids:
                os.remove(os.path.join(d, fn))
    d = sdir('pred')
    for fn in os.listdir(d):
        if time.time() - os.path.getmtime(os.path.join(d, fn)) > PRED_DAYS * 86400:
            os.remove(os.path.join(d, fn))


def out(k, v):
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write('%s=%s\n' % (k, v))
    log('output', k, '=', v)


if __name__ == '__main__':
    main()

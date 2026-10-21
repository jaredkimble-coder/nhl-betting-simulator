"""Builds the finished betting-tool page (one self-contained HTML file) from the simulated matchups the service holds."""
import json, zlib, base64, os
import numpy as np
import nhl_grade

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, 'templates', 'tool_template.html')
PARTS = os.path.join(HERE, 'templates', 'sim_parts.json')
LOCKUP = os.path.join(HERE, 'templates', 'rw_lockup.txt')      # the official RotoWire lockup as a base64 data URI, extracted from a reference widget
BLANK = 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=='


def lockup_uri():
    try:
        u = open(LOCKUP).read().strip()
        return u if u.startswith(('data:image/png;base64,', 'data:image/svg+xml;base64,')) else BLANK
    except OSError:
        return BLANK


CORE = ('hg', 'ag', 'margin', 'first', 'last', 'sog_h', 'sog_a', 'ot', 'reg_h', 'reg_a', 'n10', 'g1h', 'g1a')      # the page reads these directly


class _Seen(dict):
    def __init__(self):
        super().__init__(); self.used = set()

    def __getitem__(self, k):
        self.used.add(k); return np.zeros(1, dtype=np.int64)

    def __contains__(self, k):
        return True


def used_columns(meta):
    """The exported columns the page's bets read, in their original order (columns for players with no bets are not sent)."""
    seen = _Seen(); nhl_grade.evaluate(meta, seen)
    want = seen.used | set(CORE)
    return [c for c in meta['cols'] if c in want]


def build_html(packs, replays, template_path=TEMPLATE, parts_path=PARTS, record=None):
    """packs: list of {'meta': ..., 'cols': {name: uint8 array}}; replays: {game id: replay json}. Returns the page as a string."""
    games, raws = [], []
    for pk in packs:
        m = json.loads(json.dumps(pk['meta']))                       # copy, so building never changes what the service stores
        keep = [l for l in m['legs'] if not (l['kind'] in ('p1tot', 'ten', 'saves') and float(l['line']) % 1 == 0) and float(l['check']) >= 0.0005]
        for k, l in enumerate(keep):
            l['id'] = k
        m['legs'] = keep
        seen, tidy = set(), []
        for w in m['warnings']:
            if w.endswith(' matched by last name'):
                continue                                              # the sim already matches these players by last name
            if w not in seen:
                seen.add(w); tidy.append(w)
        m['warnings'] = tidy
        m['cols'] = used_columns(m)
        games.append(m); raws.append(np.concatenate([pk['cols'][k] for k in m['cols']]).tobytes())
    idx = sorted(range(len(games)), key=lambda k: games[k]['start'])
    games = [games[k] for k in idx]; raws = [raws[k] for k in idx]
    z = zlib.compress(b''.join(raws), 9)
    inner = {'games': games, 'sim': [replays[g['id']] for g in games if g['id'] in replays]}
    if record:
        inner['record'] = record
    data = {'z': base64.b64encode(zlib.compress(json.dumps(inner, ensure_ascii=False).encode('utf-8'), 9)).decode(), 'blob': base64.b64encode(z).decode()}
    parts = json.load(open(parts_path))
    safe = lambda s: json.dumps(s, ensure_ascii=False).replace('</', '<\\/')
    html = open(template_path, encoding='utf-8').read()
    html = html.replace('__SIM_JS__', parts['js']).replace('__SIM_CSS__', safe(parts['css'])).replace('__SIM_HTML__', safe(parts['html']))
    html = html.replace('__RW_LOCKUP__', lockup_uri())
    return html.replace('__DATA__', json.dumps(data, ensure_ascii=False))

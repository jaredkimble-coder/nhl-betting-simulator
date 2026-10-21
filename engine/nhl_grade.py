"""Grades every bet on the page against a real game result.

The same bet logic the page uses on simulated games is applied to one real game: `results_to_cols` turns a result into a one-row version of the
simulation's columns, and `evaluate` decides which bets won. A bet on a player who did not play (or a goalie who did not start) is void."""
import re, unicodedata
import numpy as np


def norm(s):
    s = unicodedata.normalize('NFD', str(s or '')).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z]', '', s)


def _first_last(s):
    parts = unicodedata.normalize('NFD', str(s or '')).encode('ascii', 'ignore').decode().lower().replace('.', '').split()
    return (parts[0][:1] if parts else '') + '|' + norm(parts[-1] if parts else '')


def leg_bits(l, C, meta):
    """Boolean array (one entry per simulated or real game in C) for one bet, or None if it cannot be evaluated."""
    k = l['kind']; g = lambda name: C[name]
    plhome = np.array([q['side'] == 'home' for q in meta['players']])
    if k == 'ml': return (g('margin') > 20) if l['side'] == 'home' else (g('margin') < 20)
    if k in ('spread',): m = g('margin') - 20; return (m + l['hp'] > 0) if l['side'] == 'home' else (m + l['hp'] < 0)
    if k == 'total': v = g('hg') + g('ag'); return v > l['line'] if l['over'] else v < l['line']
    if k == 'tt': v = g('hg') if l['team'] == 'home' else g('ag'); return v > l['line'] if l['over'] else v < l['line']
    if k == 'atg': return g(f"g{l['p']}") >= 1
    if k == 'pgoal': return g(f"g{l['p']}") > l['line']
    if k == 'pts': return g(f"g{l['p']}") + g(f"a{l['p']}") > l['line']
    if k == 'ast': return g(f"a{l['p']}") > l['line']
    if k == 'fgs': return g('first') == l['p'] + 2
    if k == 'lgs': return g('last') == l['p'] + 2
    if k == 'saves': v = g(meta['goalies'][l['gi']]['col']); return v > l['line'] if l['over'] else v < l['line']
    if k == 'sog': return g(f"s{l['p']}") > l['line']
    if k == 'reg3': return {'home': g('reg_h') > g('reg_a'), 'away': g('reg_h') < g('reg_a'), 'draw': g('reg_h') == g('reg_a')}[l['side']]
    if k == 'ten': v = g('n10'); return v > l['line'] if l['over'] else v < l['line']
    if k == 'p1puck': d = g('g1h') - g('g1a'); return (d + l['hp'] > 0) if l['side'] == 'home' else (d + l['hp'] < 0)
    if k == 'p1tot': v = g('g1h') + g('g1a'); return v > l['line'] if l['over'] else v < l['line']
    if k == 'ppp': return g(f"pp{l['p']}") > l['line']
    if k == 'pg': return g({1: f"f{l['p']}", 2: f"q2_{l['p']}", 3: f"q3_{l['p']}"}[l['per']]) >= 1
    if k == 'p1pts': return g(f"fp{l['p']}") >= 1
    if k == 'p1ast': return g(f"fa{l['p']}") >= 1
    if k in ('tsog', 'tpp', 'tsh', 'tpen'):
        v = g({'tsog': 'sog', 'tpp': 'ppg', 'tsh': 'shg', 'tpen': 'pen'}[k] + ('_h' if l['team'] == 'home' else '_a')); return v > l['line'] if l['over'] else v < l['line']
    if k == 'sogtot': v = g('sog_h') + g('sog_a'); return v > l['line'] if l['over'] else v < l['line']
    if k == 'pentot': v = g('pen_h') + g('pen_a'); return v > l['line'] if l['over'] else v < l['line']
    if k == 'sogmore': return {'home': g('sog_h') > g('sog_a'), 'away': g('sog_h') < g('sog_a'), 'even': g('sog_h') == g('sog_a')}[l['side']]
    if k == 'btts': b = (g('hg') > 0) & (g('ag') > 0); return b if l['yes'] else ~b
    if k == 'firstteam':
        if '_fs' in C: fs = C['_fs']
        else:
            fs = np.zeros(len(g('hg')), dtype=np.int64); mf = g('first') >= 2; fs[mf] = np.where(plhome[g('first')[mf] - 2], 1, 2)
        return {'home': fs == 1, 'away': fs == 2, 'none': fs == 0}[l['side']]
    if k == 'pres':
        h, a = {1: (g('g1h'), g('g1a')), 2: (g('p2h'), g('p2a')), 3: (g('p3h'), g('p3a'))}[l['per']]; return {'home': h > a, 'away': h < a, 'draw': h == a}[l['side']]
    if k == 'ptot': v = g('p2h') + g('p2a') if l['per'] == 2 else g('p3h') + g('p3a'); return v > l['line'] if l['over'] else v < l['line']
    if k == 'ptt':
        h, a = {1: (g('g1h'), g('g1a')), 2: (g('p2h'), g('p2a')), 3: (g('p3h'), g('p3a'))}[l['per']]; v = h if l['team'] == 'home' else a; return v > l['line'] if l['over'] else v < l['line']
    if k == 'pspread':
        h, a = {2: (g('p2h'), g('p2a')), 3: (g('p3h'), g('p3a'))}[l['per']]; return (h - a + l['hp'] > 0) if l['side'] == 'home' else (h - a + l['hp'] < 0)
    if k == 'pbtts':
        h, a = {0: (g('reg_h'), g('reg_a')), 1: (g('g1h'), g('g1a')), 2: (g('p2h'), g('p2a')), 3: (g('p3h'), g('p3a'))}[l['per']]; b = (h > 0) & (a > 0); return b if l['yes'] else ~b
    if k == 'wmargin':
        m = g('margin') - 20; m = m if l['side'] == 'home' else -m; return (m == l['k']) if l['k'] < 3 else (m >= 3)
    if k == 'oddeven':
        h, a = {0: (g('hg'), g('ag')), 1: (g('g1h'), g('g1a')), 2: (g('p2h'), g('p2a')), 3: (g('p3h'), g('p3a'))}[l.get('per') or 0]; o = ((h + a) % 2) == 1; return o if l['odd'] else ~o
    if k == 'anypp': b = (g('ppg_h') + g('ppg_a')) > 0; return b if l['yes'] else ~b
    if k == 'otg': b = g('ot') == 1; return b if l['yes'] else ~b
    if k == 'shutout': return ((g('ag') == 0) & (g('hg') > 0)) if l['side'] == 'home' else ((g('hg') == 0) & (g('ag') > 0))
    return None


def evaluate(meta, C):
    """{bet index in meta['legs']: bool array} for every bet this module understands."""
    out = {}
    for i, l in enumerate(meta['legs']):
        try:
            b = leg_bits(l, C, meta)
        except Exception:
            b = None
        if b is not None:
            out[i] = np.asarray(b)
    return out


def results_to_cols(meta, res):
    """One real game, as a one-row version of the simulation's columns. Returns (C, played) where `played` lists what is known to have played."""
    players = meta['players']; n = len(players)
    full = {norm(q['n']): i for i, q in enumerate(players)}; short = {}
    for i, q in enumerate(players):
        short.setdefault(_first_last(q['n']), i)

    def find(name):
        return full.get(norm(name), short.get(_first_last(name)))
    one = lambda v: np.array([int(v)], dtype=np.int64)
    goals = sorted(res.get('goals', []), key=lambda e: e['sec'])
    cnt = lambda pred: sum(1 for e in goals if pred(e))
    hg, ag = cnt(lambda e: e['team'] == 'home'), cnt(lambda e: e['team'] == 'away')
    per = lambda p, t: cnt(lambda e: e['period'] == p and e['team'] == t)
    reg_h, reg_a = cnt(lambda e: e['team'] == 'home' and e['period'] <= 3), cnt(lambda e: e['team'] == 'away' and e['period'] <= 3)
    fin = res['final']
    C = {'hg': one(hg), 'ag': one(ag), 'reg_h': one(reg_h), 'reg_a': one(reg_a), 'margin': one(max(0, min(255, fin['home'] - fin['away'] + 20))),
         'n10': one(cnt(lambda e: e['sec'] < 600)), 'g1h': one(per(1, 'home')), 'g1a': one(per(1, 'away')), 'p2h': one(per(2, 'home')), 'p2a': one(per(2, 'away')),
         'p3h': one(per(3, 'home')), 'p3a': one(per(3, 'away')),
         'ppg_h': one(cnt(lambda e: e['team'] == 'home' and e.get('strength') == 'pp')), 'ppg_a': one(cnt(lambda e: e['team'] == 'away' and e.get('strength') == 'pp')),
         'shg_h': one(cnt(lambda e: e['team'] == 'home' and e.get('strength') == 'sh')), 'shg_a': one(cnt(lambda e: e['team'] == 'away' and e.get('strength') == 'sh')),
         'pen_h': one(res.get('penalties', {}).get('home', 0)), 'pen_a': one(res.get('penalties', {}).get('away', 0)),
         'ot': one(1 if reg_h == reg_a else 0), 'pull': one(0), 'sog_h': one(res.get('sog', {}).get('home', 0)), 'sog_a': one(res.get('sog', {}).get('away', 0))}
    for i in range(n):
        for pre in ('g', 'a', 's', 'pp', 'f', 'q2_', 'q3_', 'fa', 'fp'):
            C[f'{pre}{i}'] = one(0)
    played = set()
    for name, st in res.get('players', {}).items():
        i = find(name)
        if i is not None:
            played.add(i); C[f's{i}'] = one(st.get('sog', 0))
    for e in goals:
        si = find(e['scorer']); pp = e.get('strength') == 'pp'
        if si is not None:
            played.add(si); C[f'g{si}'] += 1
            if pp: C[f'pp{si}'] += 1
            if e['period'] == 1: C[f'f{si}'] += 1; C[f'fp{si}'] += 1
            if e['period'] == 2: C[f'q2_{si}'] += 1
            if e['period'] == 3: C[f'q3_{si}'] += 1
        for a in e.get('assists', []):
            ai = find(a)
            if ai is not None:
                played.add(ai); C[f'a{ai}'] += 1
                if pp: C[f'pp{ai}'] += 1
                if e['period'] == 1: C[f'fa{ai}'] += 1; C[f'fp{ai}'] += 1
    for key, pick in (('first', goals[:1]), ('last', goals[-1:])):
        C[key] = one(0)
        if pick:
            i = find(pick[0]['scorer']); C[key] = one(i + 2 if i is not None else 1)
    fs = 0
    if goals: fs = 1 if goals[0]['team'] == 'home' else 2
    C['_fs'] = one(fs)
    # goalies: saves by the matching goalie of the simulated game; a goalie who did not play leaves their bets void
    gplayed = {}
    for gi, gm in enumerate(meta['goalies']):
        C[gm['col']] = one(0)
        for name, st in res.get('goalies', {}).items():
            if norm(name) == norm(gm['n']) or _first_last(name) == _first_last(gm['n']):
                C[gm['col']] = one(st.get('saves', 0)); gplayed[gi] = st.get('shotsAgainst', 0) > 0 or st.get('saves', 0) > 0
    return C, played, gplayed


def grade(meta, res):
    """Grades every bet. Returns a list of {id, key, group, kind, p, sbP, sbOdds, win} with win True/False, or None when the bet is void."""
    C, played, gplayed = results_to_cols(meta, res)
    bits = evaluate(meta, C); out = []
    for i, l in enumerate(meta['legs']):
        if i not in bits:
            continue
        if (l['kind'] in ('p1tot', 'ten', 'saves') and float(l['line']) % 1 == 0) or float(l['check']) < 0.0005:
            continue      # the same bets the page leaves out: whole-number lines (a tie is a push) and bets on players who are not in the lineup
        if l.get('p') is not None and l['kind'] in ('atg', 'pgoal', 'pts', 'ast', 'sog', 'ppp', 'pg', 'p1pts', 'p1ast') and l['p'] not in played:
            win = None      # the player did not play: the bet is void
        elif l['kind'] == 'saves' and not gplayed.get(l['gi'], False):
            win = None      # the goalie did not start: void
        else:
            win = bool(bits[i][0])
        out.append({'id': i, 'key': l['group'] + '|' + l['label'], 'group': l['group'], 'kind': l['kind'], 'label': l['label'], 'p': l['check'],
                    'sbP': l['sb']['p'] if l.get('sb') else None, 'sbOdds': l['sb']['odds'] if l.get('sb') else None, 'win': win})
    return out

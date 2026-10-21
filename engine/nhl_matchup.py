"""Matchup analysis for one game. MoneyPuck gives last season's team and player numbers (shots, expected goals, special teams);
ShotPropz gives this season's shots and goals each team allows to each position."""
import re, unicodedata
import numpy as np

POS = ('C', 'LW', 'RW', 'D')
MP_POS = {'C': 'C', 'L': 'LW', 'R': 'RW', 'D': 'D'}
POS_WORD = {'C': 'centers', 'LW': 'left wingers', 'RW': 'right wingers', 'D': 'defensemen'}


def _norm(s):
    return re.sub(r'[^a-z]', '', unicodedata.normalize('NFD', str(s or '')).encode('ascii', 'ignore').decode().lower())


def _team(t):
    """One team's per-game numbers from MoneyPuck's season totals, or None if anything is missing."""
    if not t:
        return None
    gp = t.get('gp') or 0; a, e, pp, pk = t.get('all'), t.get('5on5'), t.get('5on4'), t.get('4on5')
    if not (gp and a and e and pp and pk and pp.get('toi') and pk.get('toi') and (e['xgF'] + e['xgA'])):
        return None
    pen = {'penTaken': a['penF'] / gp, 'penDrawn': a['penA'] / gp} if ('penF' in a and 'penA' in a) else {}
    return {**pen, 'gp': int(gp), 'sogF': a['sogF'] / gp, 'sogA': a['sogA'] / gp, 'xgF': a['xgF'] / gp, 'xgA': a['xgA'] / gp, 'ev5Share': e['xgF'] / (e['xgF'] + e['xgA']),
            'ppXgF60': pp['xgF'] / (pp['toi'] / 3600.0), 'pkXgA60': pk['xgA'] / (pk['toi'] / 3600.0)}


def _by_pos(tables, defender, val):
    """How many shots (or goals) a defending team allows per game to each position, against the league."""
    out = {}
    for pos in POS:
        tbl = tables.get(pos) or {}; t = tbl.get(defender)
        if not t or val not in t:
            continue
        vals = [v[val] for v in tbl.values() if val in v]; lg = float(np.mean(vals))
        out[pos] = {'v': round(t[val], 2), 'lg': round(lg, 2), 'pct': round((t[val] / lg - 1) * 100, 1) if lg else None, 'rank': 1 + sum(1 for v in vals if v > t[val]), 'of': len(vals), 'gp': int(t.get('gp', 0))}
    return out


def build_matchup(p, meta, cols, sim=None):
    """p: the n8n payload; meta: the exported game meta; cols: the exported sample columns. Returns the matchup dict, or None when there is no data."""
    st = p.get('stats') or {}; mp = st.get('moneyPuck') or {}; sp = st.get('shotPropz') or {}
    ab = {'home': p['fixture']['home']['abbr'], 'away': p['fixture']['away']['abbr']}
    T = {s: _team((mp.get('teams') or {}).get(s)) for s in ('home', 'away')}
    lg = mp.get('league') or {}
    if not (T['home'] and T['away'] and lg.get('sogPerTeamGame')):
        return None
    out = {'season': mp.get('season'), 'abbr': ab, 'league': {k: round(lg[k], 3) for k in ('sogPerTeamGame', 'xgPerTeamGame', 'ppXgF60', 'pkXgA60', 'penPerTeamGame', 'ppgPerTeamGame') if lg.get(k)},
           'teams': {s: {k: round(v, 3) for k, v in T[s].items()} for s in T}}
    # shots each team should take against this opponent, from MoneyPuck alone, next to the sim's
    L = lg['sogPerTeamGame']; opp = {'home': 'away', 'away': 'home'}
    mps = {s: L * (T[s]['sogF'] / L) * (T[opp[s]]['sogA'] / L) for s in T}
    sim_shots = {'home': float(np.mean(cols['sog_h'])), 'away': float(np.mean(cols['sog_a']))}
    out['shots'] = {s: {'mp': round(mps[s], 1), 'sim': round(sim_shots[s], 1)} for s in T}
    # each attack against the other team's defense, by position (ShotPropz, this season)
    sa, ga = sp.get('sogAgainst') or {}, sp.get('goalsAgainst') or {}
    out['attack'] = {s: {'vs': ab[opp[s]], 'shots': _by_pos(sa, ab[opp[s]], 'sa'), 'goals': _by_pos(ga, ab[opp[s]], 'ga')} for s in T}
    # the players with the most expected goals per game, with the sim's chance to score and the position matchup
    plr = {}
    for q in meta['players']:
        plr.setdefault(_norm(q['n']), []).append(q)
    atg = {l['p']: l for l in meta['legs'] if l['kind'] == 'atg'}
    out['players'] = {}
    for s in T:
        rows = []
        for r in mp.get('skaters', []):
            if r.get('t') != ab[s] or (r.get('gp') or 0) < 20 or not r.get('xg'):
                continue
            cand = [q for q in plr.get(_norm(r['n']), []) if q['side'] == s]
            if not cand:
                continue
            idx = meta['players'].index(cand[0]); leg = atg.get(idx); pos = MP_POS.get(r.get('p'))
            if leg is None or leg['check'] < 0.005:      # the sim has him out of the lineup, so he is not shown
                continue
            sh = out['attack'][s]['shots'].get(pos) if pos else None      # this player's position against the opponent's defense
            rows.append({'n': r['n'], 'pos': pos, 'xgPg': round(r['xg'] / r['gp'], 3), 'gPg': round(r['g'] / r['gp'], 3), 'sogPg': round(r['sog'] / r['gp'], 2),
                         'sim': round(leg['check'], 4) if leg else None, 'odds': leg['sb']['odds'] if leg and leg.get('sb') else None, 'book': leg['sb']['book'] if leg and leg.get('sb') else None,
                         'tilt': sh['pct'] if sh else None})
        out['players'][s] = sorted(rows, key=lambda x: -x['xgPg'])[:4]
    out['notes'] = _notes(out, ab)
    out['use'] = _use(out, ab, sim, meta)
    return out


def _notes(m, ab):
    notes = []; T, lg = m['teams'], m['league']; a, h = ab['away'], ab['home']; s = m['shots']
    yr = f"{int(m['season'])}-{str(int(m['season']) + 1)[2:]}"
    notes.append(f"{a} should take about {s['away']['mp']:.1f} shots on goal and {h} about {s['home']['mp']:.1f} (MoneyPuck's {yr} shots for and against; league average {lg['sogPerTeamGame']:.1f}). The sim, which follows the books' goalie saves lines when they are posted, has {s['away']['sim']:.1f} and {s['home']['sim']:.1f}.")
    notes.append(f"At 5v5 in {yr}, {a} had {T['away']['ev5Share'] * 100:.0f}% of the expected goals and {h} had {T['home']['ev5Share'] * 100:.0f}% (50% is even).")
    if lg.get('penPerTeamGame') and 'penTaken' in T['away'] and 'penTaken' in T['home']:
        notes.append(f"Penalties (MoneyPuck, {yr}): {a} takes {T['away']['penTaken']:.1f} a game and draws {T['away']['penDrawn']:.1f}, {h} takes {T['home']['penTaken']:.1f} and draws {T['home']['penDrawn']:.1f} (league {lg['penPerTeamGame']:.1f}).")
    if lg.get('ppXgF60'):
        d = max((('away', T['away']['ppXgF60']), ('home', T['home']['ppXgF60'])), key=lambda x: x[1])
        if d[1] >= lg['ppXgF60'] * 1.12:
            notes.append(f"{ab[d[0]]} has the stronger power play ({d[1]:.1f} expected goals per 60 minutes, league {lg['ppXgF60']:.1f}).")
    cells = [(abs(v['pct']), side, pos, v) for side in ('away', 'home') for pos, v in m['attack'][side]['shots'].items() if v['gp'] >= 5 and v['pct'] is not None]
    if cells:
        _, side, pos, v = max(cells, key=lambda c: c[0])
        if abs(v['pct']) >= 10:
            d = m['attack'][side]['vs']
            notes.append(f"Biggest position matchup: {d} has allowed {abs(v['pct']):.0f}% {'more' if v['pct'] > 0 else 'fewer'} shots than the league average to {POS_WORD[pos]}, which is who {ab[side]} sends (rank {v['rank']} of {v['of']}, where 1 allows the most; ShotPropz, {v['gp']} games this season).")
    return notes


def _use(m, ab, sim, meta):
    """What the matchup changed in the sim, in plain words."""
    use = []; a, h = ab['away'], ab['home']; lg = m['league']
    if sim is None:
        return use
    t = sim.get('pen_target') or {}
    if t and lg.get('penPerTeamGame'):
        use.append(f"Penalties: the sim expects {a} to take {t['away']:.1f} and {h} {t['home']:.1f} (league {lg['penPerTeamGame']:.1f}), from how often each team takes penalties and the other draws them.")
    st = (sim.get('matchup_use') or {}).get('st') or {}
    pp = st.get('ppg') or {}
    if pp.get('home') and pp.get('away'):
        srcs = {pp['home']['src'], pp['away']['src']}
        txt = f"Power-play goals: {a} {pp['away']['v']:.2f} and {h} {pp['home']['v']:.2f} per game (league {lg.get('ppgPerTeamGame', 0.55):.2f})."
        if any('books' in x for x in srcs):
            txt += " The books' lines imply about 0.8 a team, well above what teams score, so the sim scales them to the league rate and averages them with the MoneyPuck matchup."
        else:
            txt += " No book posted a line, so this is the league rate adjusted for the matchup."
        use.append(txt)
    sh = (sim.get('matchup_use') or {}).get('shots') or {}
    mpside = [s for s, v in sh.items() if v.get('source') == 'MoneyPuck']
    if mpside:
        use.append("Team shots: " + " and ".join(ab[s] for s in mpside) + " had no goalie saves line, so their shots were set from MoneyPuck's shots for and against.")
    n = sum(1 for v in (sim.get('shot_flags') or {}).values() if v == 'stats + ShotPropz matchup')
    if n:
        use.append(f"{n} players had no shots price, so their shots use their own rate, tilted by how the opponent defends their position (ShotPropz) and scaled to the team's shots.")
    return use

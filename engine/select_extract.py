import sys, json, math, collections, pickle, time
sys.path.insert(0, '/home/claude')
import numpy as np
import nhl_sim as ns, nhl_props as npp, nhl_v3 as v3

TEAMS = {
    'Pittsburgh Penguins': dict(abbr='PIT', short='Penguins', fill='#1C1C1C', ring='#FCB514', glow='#FCB514'),
    'Colorado Avalanche': dict(abbr='COL', short='Avalanche', fill='#6F263D', ring='#8FB8DE', glow='#C8567A'),
    'Boston Bruins': dict(abbr='BOS', short='Bruins', fill='#FFB81C', ring='#1C1C1C', glow='#FFB81C'),
    'Anaheim Ducks': dict(abbr='ANA', short='Ducks', fill='#1C1C1C', ring='#F47A38', glow='#F47A38'),
    'Buffalo Sabres': dict(abbr='BUF', short='Sabres', fill='#002654', ring='#FCB514', glow='#FCB514'),
    'Calgary Flames': dict(abbr='CGY', short='Flames', fill='#C8102E', ring='#F1BE48', glow='#F1BE48'),
    'Carolina Hurricanes': dict(abbr='CAR', short='Hurricanes', fill='#CC0000', ring='#1C1C1C', glow='#E8453C'),
    'Chicago Blackhawks': dict(abbr='CHI', short='Blackhawks', fill='#CF0A2C', ring='#1C1C1C', glow='#E8453C'),
    'Columbus Blue Jackets': dict(abbr='CBJ', short='Blue Jackets', fill='#002654', ring='#CE1126', glow='#E8453C'),
    'Dallas Stars': dict(abbr='DAL', short='Stars', fill='#006847', ring='#C8D0D4', glow='#3FBF8B'),
    'Detroit Red Wings': dict(abbr='DET', short='Red Wings', fill='#CE1126', ring='#FFFFFF', glow='#E8453C'),
    'Edmonton Oilers': dict(abbr='EDM', short='Oilers', fill='#041E42', ring='#FF4C00', glow='#FF6A2A'),
    'Florida Panthers': dict(abbr='FLA', short='Panthers', fill='#041E42', ring='#C8102E', glow='#E8453C'),
    'Los Angeles Kings': dict(abbr='LAK', short='Kings', fill='#111111', ring='#A2AAAD', glow='#A2AAAD'),
    'Minnesota Wild': dict(abbr='MIN', short='Wild', fill='#154734', ring='#A6192E', glow='#E8453C'),
    'Montreal Canadiens': dict(abbr='MTL', short='Canadiens', fill='#AF1E2D', ring='#192168', glow='#E8453C'),
    'Nashville Predators': dict(abbr='NSH', short='Predators', fill='#FFB81C', ring='#041E42', glow='#FFB81C'),
    'New Jersey Devils': dict(abbr='NJD', short='Devils', fill='#CE1126', ring='#1C1C1C', glow='#E8453C'),
    'New York Islanders': dict(abbr='NYI', short='Islanders', fill='#00539B', ring='#F47D30', glow='#F47D30'),
    'New York Rangers': dict(abbr='NYR', short='Rangers', fill='#0038A8', ring='#CE1126', glow='#6C8FE8'),
    'Ottawa Senators': dict(abbr='OTT', short='Senators', fill='#C52032', ring='#C2912C', glow='#C2912C'),
    'Philadelphia Flyers': dict(abbr='PHI', short='Flyers', fill='#F74902', ring='#1C1C1C', glow='#FF7A3D'),
    'San Jose Sharks': dict(abbr='SJS', short='Sharks', fill='#006D75', ring='#1C1C1C', glow='#00A9B5'),
    'Seattle Kraken': dict(abbr='SEA', short='Kraken', fill='#001628', ring='#99D9D9', glow='#99D9D9'),
    'St. Louis Blues': dict(abbr='STL', short='Blues', fill='#002F87', ring='#FCB514', glow='#FCB514'),
    'Tampa Bay Lightning': dict(abbr='TBL', short='Lightning', fill='#002868', ring='#FFFFFF', glow='#6C8FE8'),
    'Toronto Maple Leafs': dict(abbr='TOR', short='Maple Leafs', fill='#00205B', ring='#FFFFFF', glow='#6C8FE8'),
    'Utah Mammoth': dict(abbr='UTA', short='Mammoth', fill='#6CACE4', ring='#010101', glow='#6CACE4'),
    'Vancouver Canucks': dict(abbr='VAN', short='Canucks', fill='#00205B', ring='#00843D', glow='#2FBF71'),
    'Vegas Golden Knights': dict(abbr='VGK', short='Golden Knights', fill='#B4975A', ring='#333F42', glow='#B4975A'),
    'Washington Capitals': dict(abbr='WSH', short='Capitals', fill='#C8102E', ring='#041E42', glow='#E8453C'),
    'Winnipeg Jets': dict(abbr='WPG', short='Jets', fill='#041E42', ring='#A6A6A6', glow='#7BAFD4'),
}
PEN_TYPES = ['tripping', 'hooking', 'slashing', 'interference', 'holding', 'high-sticking', 'cross-checking', 'roughing', 'delay of game']


def leg_bool(sim, key):
    """Per-sim boolean for one selection (same definitions as nhl_props.sim_prob)."""
    mid, subj, line, side = key
    if mid == 'moneyline':
        return sim['home_win'] if side == 'home' else ~sim['home_win']
    if mid == 'moneyline_3-way':
        rh, ra = sim['reg_h'], sim['reg_a']
        return {'home': rh > ra, 'draw': rh == ra, 'away': rh < ra}[side]
    if mid == 'puck_line':
        m = sim['margin'] + line
        return (m > 0) if side == 'home' else (m < 0)
    if mid == '1st_period_puck_line':
        m = sim['g1h'] - sim['g1a'] + line
        return (m > 0) if side == 'home' else (m < 0)
    if mid == 'total_goals':
        t = sim['hg'] + sim['ag']; return (t > line) if side == 'over' else (t < line)
    if mid == '1st_period_total_goals':
        t = sim['g1h'] + sim['g1a']; return (t > line) if side == 'over' else (t < line)
    if mid == '1st_10_minutes_total_goals':
        return (sim['n10'] > line) if side == 'over' else (sim['n10'] < line)
    if mid == 'team_total':
        t = sim['hg'] if subj == 'home' else sim['ag']; return (t > line) if side == 'over' else (t < line)
    if mid == 'anytime_goal_scorer':
        return sim['goals'][:, sim['idx'][subj]] >= 1
    if mid == 'first_goal_scorer':
        return sim['first_col'] == sim['idx'][subj]
    if mid in ('player_goals', 'player_assists', 'player_points', 'player_shots_on_goal'):
        i = sim['idx'][subj]
        arr = {'player_goals': sim['goals'][:, i], 'player_assists': sim['assists'][:, i], 'player_points': sim['goals'][:, i] + sim['assists'][:, i],
               'player_shots_on_goal': sim['shots'][:, i]}[mid]
        return (arr > line) if side == 'over' else (arr < line)
    if mid == 'player_saves':
        arr = sim['saves'][subj]; return (arr > line) if side == 'over' else (arr < line)
    return None


def pick_most_likely(sim, grid, extra=None):
    """The simulated game whose outcomes are, on average, the most likely ones across all 14 market groups.
    Each group gets equal weight; inside a group each leg is weighted equally. Score = sum of log P(outcome that happened)."""
    n = len(sim['hg'])
    legs = collections.defaultdict(list)
    for r in grid:
        mid, subj, line, side = r['_key']
        if side in ('under', 'away') and mid != 'moneyline_3-way':
            continue                                   # the other side of a two-way market adds no information
        if mid == 'moneyline' and side == 'away':
            continue
        p = r['Sim probability']
        if not (0.0004 < p < 0.9996):
            continue
        legs[r['Group']].append((r['_key'], float(np.clip(p, 0.001, 0.999))))
    score = np.zeros(n)
    agree = {}
    ngroups = len(legs) + (1 if extra else 0)
    for grp, items in legs.items():
        w = 1.0 / len(items) / ngroups
        ag = 0.0
        for key, p in items:
            b = leg_bool(sim, key)
            if b is None:
                continue
            score += w * np.where(b, math.log(p), math.log(1 - p))
            ag += w * float(np.mean(np.where(b, p >= 0.5, p < 0.5)))
        agree[grp] = ag / (1.0 / ngroups)
    if extra:
        for grp, arrs in extra.items():
            w = 1.0 / len(arrs) / ngroups; ag = 0.0
            for b in arrs:
                p = float(np.clip(b.mean(), 0.001, 0.999))
                score += w * np.where(b, math.log(p), math.log(1 - p)); ag += w * float(np.mean(np.where(b, p >= 0.5, p < 0.5)))
            agree[grp] = ag / (1.0 / ngroups)
    # Realism guard: the single 'most likely' combination of many over-bets leans toward games where unlikely things (assists) do not happen,
    # which produced replays where most goals were unassisted. Only consider games whose assist pattern looks like real hockey:
    # at most 20% of goals unassisted and 1.6 to 2.0 assists per goal (the sim's own average is about 1.65).
    try:
        gs = np.asarray(sim['goal_sim']); a1 = np.asarray(sim['goal_a1']); a2 = np.asarray(sim['goal_a2'])
        ng = np.bincount(gs, minlength=n).astype(float)
        un = np.bincount(gs[a1 < 0], minlength=n).astype(float)
        asst = (np.bincount(gs[a1 >= 0], minlength=n) + np.bincount(gs[a2 >= 0], minlength=n)).astype(float)
        ok = (ng == 0) | ((un <= 0.2 * ng) & (asst >= 1.6 * ng) & (asst <= 2.0 * ng))
        if ok.sum() >= 50:
            score = np.where(ok, score, -np.inf)
    except Exception:
        pass
    order = np.argsort(-score)
    return int(order[0]), score, agree, [int(x) for x in order[:5]]


def extract(sim, j, fit, players, roster, goalies, game, seed=7):
    rng = np.random.default_rng(seed + j)
    pids = sim['pids']; idx = sim['idx']; lu = sim['lu']
    ridx = {}
    R = []
    for side in ('away', 'home'):
        tid = fit['aid'] if side == 'away' else fit['hid']
        for pid in lu[side]['pids']:
            info = players[pid]
            no = next((r['no'] for r in roster[tid] if r['id'] == pid), None)
            ln = next((k + 1 for k, l in enumerate(lu[side]['lines']) if pid in l), None)
            pr = next((k + 1 for k, l in enumerate(lu[side]['pairs']) if pid in l), None)
            pp = next((k + 1 for k, l in enumerate(lu[side]['pp']) if pid in l), None)
            ridx[idx[pid]] = len(R)
            is_d = pid in set(lu[side]['D']); is_c = pid in lu[side]['centers'] if 'centers' in lu[side] else info['pos'] == 'C'
            R.append({'n': info['name'], 'no': no, 'pos': 'D' if is_d else ('C' if is_c else 'W'), 'side': side, 'line': ln, 'pair': pr, 'pp': pp})
    # goalies
    gk = {}
    for gpid, gi in goalies.items():
        side = 'home' if gi['team'] == fit['hid'] else 'away'
        no = next((r['no'] for r in roster[gi['team']] if r['id'] == gpid), None)
        gk[side] = {'pid': gpid, 'n': players[gpid]['name'], 'no': no, 'M': gi['M']}

    T = 3900
    # ---- penalties of sim j
    pens = []
    for side in ('home', 'away'):
        st = sim['pen_start'][side][j]; en = sim['pen_end'][side][j]
        for k in range(len(st)):
            if np.isnan(st[k]):
                continue
            e = en[k] if not np.isnan(en[k]) else min(st[k] + 120.0, 3480.0)
            e = max(e, st[k] + 2.0)           # a power-play goal in the same instant can end a penalty 'before' it starts; it still lasts a moment
            pens.append({'side': side, 'start': float(st[k]), 'end': float(min(e, st[k] + 120.0))})
    pens.sort(key=lambda p: p['start'])

    def active(side, t):
        return sum(1 for p in pens if p['side'] == side and p['start'] <= t < p['end'])
    sk = {s: np.array([5 - min(active(s, t + 0.5), 2) for t in range(T)]) for s in ('home', 'away')}
    for s in sk:
        sk[s][3600:] = 3

    # ---- goals of sim j
    m = np.where(sim['goal_sim'] == j)[0]
    m = m[np.argsort(sim['goal_t'][m])]
    goals_raw = []
    for gi in m:
        side = 'home' if sim['goal_team'][gi] == 1 else 'away'
        goals_raw.append({'t': float(sim['goal_t'][gi] * 60.0), 'side': side, 'cat': int(sim['goal_cat'][gi]), 'cols': [int(c) for c in sim['goal_cols'][gi] if c >= 0],
                          'scorer': int(sim['goal_scorer'][gi]), 'a1': int(sim['goal_a1'][gi]), 'a2': int(sim['goal_a2'][gi])})

    # ---- on-ice schedule per team, one entry per second
    base = {}
    for side in ('home', 'away'):
        L = lu[side]; n_l, n_p = len(L['lines']), len(L['pairs'])
        fw = v3.EV_F_W[:n_l] / v3.EV_F_W[:n_l].sum(); dw = v3.EV_D_W[:n_p] / v3.EV_D_W[:n_p].sum()
        fl = np.zeros(T, dtype=int); dp = np.zeros(T, dtype=int)
        t, prev = 0, -1
        while t < T:
            k = int(rng.choice(n_l, p=fw)); k = k if k != prev else int(rng.choice(n_l, p=fw)); prev = k
            ln = int(rng.uniform(34, 56)); fl[t:t + ln] = k; t += ln
        t, prev = 0, -1
        while t < T:
            k = int(rng.choice(n_p, p=dw)); k = k if k != prev else int(rng.choice(n_p, p=dw)); prev = k
            ln = int(rng.uniform(46, 72)); dp[t:t + ln] = k; t += ln
        base[side] = (fl, dp)
    def base_cols(side, t):
        L = lu[side]; fl, dp = base[side]
        return [idx[p] for p in L['lines'][fl[t]]] + [idx[p] for p in L['pairs'][dp[t]]]

    # penalty takers: on the ice at the whistle, and not someone who scores or assists inside the penalty window
    goal_involved = collections.defaultdict(set)
    for p in pens:
        for g in goals_raw:
            if g['side'] != p['side'] and False:
                pass
    for p in pens:
        side = p['side']
        busy = set()
        for g in goals_raw:
            if g['side'] == side and p['start'] - 12 <= g['t'] <= p['end'] + 6:
                busy |= {g['scorer'], g['a1'], g['a2']} | set(g['cols'])
        pool = [c for c in base_cols(side, int(p['start'])) if c in ridx and c not in busy]
        if not pool:
            pool = [c for c in base_cols(side, int(p['start'])) if c in ridx]
        w = np.array([1.6 if R[ridx[c]]['pos'] == 'D' else 1.0 for c in pool])
        p['off'] = int(rng.choice(pool, p=w / w.sum()))
    boxed = {s_: [(p['start'], p['end'], p['off']) for p in pens if p['side'] == s_] for s_ in ('home', 'away')}
    def out_at(side, t):
        return {c for (a_, b_, c) in boxed[side] if a_ <= t < b_}

    def fill(side, cols, gone, want_d=None):
        """Replace anyone in the penalty box with the next available skater of the same kind."""
        L = lu[side]; res = []
        for c in cols:
            if c in gone:
                is_d = R[ridx[c]]['pos'] == 'D'
                pool = [idx[p] for p in (L['D'] if is_d else L['F']) if idx[p] not in gone and idx[p] not in cols and idx[p] not in res]
                res.append(pool[0] if pool else c)
            else:
                res.append(c)
        return res

    onice = {}
    for side in ('home', 'away'):
        L = lu[side]; n_p = len(L['pairs']); opp = 'away' if side == 'home' else 'home'
        oc = [None] * T
        for t in range(T):
            gone = out_at(side, t)
            oc[t] = [c for c in base_cols(side, t) if c not in gone]
        t = 0
        while t < 3600:
            a, b = sk[side][t], sk[opp][t]
            if a == 5 and b < 5:
                u = 0 if rng.random() < 0.65 else 1; ln = int(rng.uniform(40, 75))
                for tt in range(t, min(t + ln, 3600)):
                    if sk[side][tt] == 5 and sk[opp][tt] < 5:
                        oc[tt] = fill(side, [idx[p] for p in L['pp'][(u + (tt - t) // 60) % 2]], out_at(side, tt))
                t += ln
            elif a < 5 and b == 5:
                u = 0 if rng.random() < 0.55 else 1; ln = int(rng.uniform(40, 70))
                for tt in range(t, min(t + ln, 3600)):
                    if sk[side][tt] < 5 and sk[opp][tt] == 5:
                        pk = fill(side, [idx[p] for p in L['pk'][(u + (tt - t) // 55) % 2]], out_at(side, tt))
                        oc[tt] = pk[:sk[side][tt]] if sk[side][tt] < 4 else pk
                t += ln
            elif a == 4 and b == 4:
                u = int(rng.integers(0, 2)); ln = int(rng.uniform(40, 70))
                for tt in range(t, min(t + ln, 3600)):
                    if sk[side][tt] == 4 and sk[opp][tt] == 4:
                        oc[tt] = fill(side, [idx[p] for p in L['lines'][u][:2]] + [idx[p] for p in L['pairs'][u % n_p]], out_at(side, tt))
                t += ln
            else:
                t += 1
        onice[side] = oc
    for g in goals_raw:                           # the scoring unit is on the ice for the goal
        s_ = g['side']; t0 = int(g['t'])
        for tt in range(max(0, t0 - 12), min(T, t0 + 6)):
            onice[s_][tt] = [c for c in g['cols'] if c not in out_at(s_, tt)] if False else g['cols']

    units = []; unit_id = {}
    def uid(side, t):
        t = int(min(max(t, 0), T - 1))
        tup = tuple(ridx[c] for c in onice[side][t] if c in ridx)
        if tup not in unit_id:
            unit_id[tup] = len(units); units.append(list(tup))
        return unit_id[tup]

    events = []
    d58 = int(sim['d58'][j]); leader = 'home' if d58 > 0 else ('away' if d58 < 0 else None); trailing = None if leader is None else ('away' if leader == 'home' else 'home')
    late_goals = [g for g in goals_raw if 3480 <= g['t'] < 3600]
    pull_t = None
    if leader and (abs(d58) == 1 or (abs(d58) == 2 and any(g['side'] == leader for g in late_goals))):
        first_late = min([g['t'] for g in late_goals], default=3600.0)
        pull_t = max(3482.0, min(3492.0, first_late - 6))
        events.append({'type': 'pull', 't': round(pull_t, 1), 'team': trailing, 'on': {'home': uid('home', pull_t), 'away': uid('away', pull_t)}})
    strength_label = {0: 'ev', 1: 'pp', 2: 'sh', 3: 'ev4', 4: 'pp', 5: 'sh', 6: 'ot', 7: 'late'}
    for g in goals_raw:
        lab = strength_label[g['cat']]
        if g['cat'] == 7 and pull_t is not None:
            lab = 'en' if g['side'] == leader else 'tie'
        elif g['cat'] == 7:
            lab = 'ev'
        a = [ridx[c] for c in (g['a1'], g['a2']) if c >= 0 and c in ridx]
        events.append({'type': 'goal', 't': round(g['t'], 1), 'team': g['side'], 'scorer': ridx[g['scorer']], 'assists': a, 'strength': lab,
                       'on': {'home': uid('home', g['t']), 'away': uid('away', g['t'])}})
    # penalty events: offender chosen above; a power-play goal ends the penalty at the goal
    goal_ts = {s_: [g['t'] for g in goals_raw if g['side'] == s_ and g['cat'] in (1, 4)] for s_ in ('home', 'away')}
    for p in pens:
        side = p['side']; opp = 'away' if side == 'home' else 'home'
        reason = 'expired'; end = p['end']
        for t in goal_ts[opp]:
            if p['start'] <= t <= p['start'] + 121 and (t >= p['end'] - 12) and p['end'] < p['start'] + 119:
                end = max(end if end > t else t, t); reason = 'goal'; end = t; break
        p['end'] = end
        events.append({'type': 'penalty', 't': round(p['start'], 1), 'team': side, 'player': ridx[p['off']], 'kind': PEN_TYPES[int(rng.integers(0, len(PEN_TYPES)))],
                       'end': round(end, 1), 'endReason': reason, 'on': {'home': uid('home', p['start']), 'away': uid('away', p['start'])}})
    # saved shots: shooters weighted by their saved-shot counts, placed in seconds when they are on the ice
    taken = [e['t'] for e in events]
    for side in ('away', 'home'):
        dside = 'home' if side == 'away' else 'away'
        gpid = gk[dside]['pid'] if dside in gk else None
        n_sv = int(sim['saves'][gpid][j]) if gpid else 26
        pl = [idx[p] for p in lu[side]['pids']]
        w = np.array([max(int(sim['shots'][j, c]) - int(sim['goals'][j, c]), 0) for c in pl], dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(pl))
        counts = rng.multinomial(n_sv, w / w.sum())
        for c, k in zip(pl, counts):
            secs = [t for t in range(15, 3475) if c in onice[side][t] and (pull_t is None or not (dside == trailing and t >= pull_t))]
            if not secs:
                continue
            wt = np.array([2.4 if sk[side][t] > sk[dside][t] else (0.5 if sk[side][t] < sk[dside][t] else 1.0) for t in secs])
            for _ in range(int(k)):
                for attempt in range(40):
                    t = int(rng.choice(secs, p=wt / wt.sum())) + float(rng.random())
                    if all(abs(t - x) >= 6 for x in taken):
                        break
                taken.append(t)
                events.append({'type': 'save', 't': round(t, 1), 'team': side, 'shooter': ridx[c], 'on': {'home': uid('home', t), 'away': uid('away', t)}})
    for k in (1, 2, 3):
        events.append({'type': 'period_end', 't': 1200.0 * k, 'k': k})
        if k < 3:
            events.append({'type': 'faceoff', 't': 1200.0 * k, 'k': k, 'on': {'home': uid('home', 1200 * k), 'away': uid('away', 1200 * k)}})
    events.append({'type': 'faceoff', 't': 0.0, 'k': 0, 'on': {'home': uid('home', 0), 'away': uid('away', 0)}})
    events.sort(key=lambda e: (e['t'], {'period_end': 0, 'faceoff': 1}.get(e['type'], 2)))
    events.append({'type': 'final', 't': 3600.0})
    return {'roster': R, 'units': units, 'events': events, 'goalies': gk, 'd58': d58}

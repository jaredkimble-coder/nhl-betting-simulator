#!/usr/bin/env python3
"""
nhl_v3.py - game simulation with penalties, special teams, and line deployment.

What changed from nhl_sim.py / nhl_props.py
  * Time-stepped game (10-second slots). Minor penalties start at random times; the other team gets a
    power play until 2:00 passes or it scores. Strength states 5v5, 5v4, 4v5, 5v3, 4v4 change scoring rates.
  * Every team has forward lines (3 F), defense pairs (2 D), power-play units and penalty-kill units.
    Lines are inferred from market production (points props); power-play units from the power-play-points props.
  * Every goal is scored by someone on the ice. The scoring unit is drawn from ice-time weights for that
    strength state, then the scorer and assisters are drawn from that unit only.
  * Special-teams scoring rates are fitted to the posted team power-play-goal and shorthanded-goal lines.

Everything downstream (shots, saves, pricing grid) uses the same array layout as nhl_props.py.
"""
import collections
import math
import numpy as np
from scipy.optimize import minimize
import nhl_sim as ns
import nhl_props as npp
from nhl_sim import imp, devig_pair

EV_F_W = np.array([0.36, 0.29, 0.19, 0.12, 0.04])     # share of 5v5 time by forward line
EV_D_W = np.array([0.40, 0.33, 0.17, 0.10])           # share of 5v5 time by defense pair
LATE_F_W = np.array([0.50, 0.30, 0.15, 0.04, 0.01])
LATE_D_W = np.array([0.60, 0.30, 0.09, 0.01])
PP_W = np.array([0.65, 0.35])
PK_W = np.array([0.55, 0.45])
STATS = None          # nhl_stats.Stats, set by the pipeline when statistics are available
MIX = 0.5             # share of even-strength goals scored by a mixed unit (line juggling), tuned against real games
AST_PROBS = (0.07, 0.22, 0.71)   # goals with 0 / 1 / 2 assists (real games: about 2.65 points per goal)
MU_PEN = 3.0            # penalties that create a power play, taken per team per game (league-typical)
CAL_VERSION = 4         # raise this when the model changes, so older saved calibrations are redone
LEAGUE_PEN = 3.49       # every penalty taken per team per game in 2025-26, including offsetting ones that create no power play
PPG_BOOK_LEVEL = 0.79    # power-play goals per team the posted 'over 0.5' prices imply on average (32 teams, DraftKings), against a real league rate of 0.55


def pen_mu(side):
    """How many power-play-creating penalties this team should take in this game: the league rate scaled by the matchup (MoneyPuck) when we have it."""
    r = STATS.mp_pen_rate(side) if (STATS is not None and hasattr(STATS, 'mp_pen_rate')) else None
    return MU_PEN * (r if r else 1.0)
PP_SECONDS = 120.0
K_PEN = 5
PEN_TYPES = ['tripping', 'hooking', 'slashing', 'interference', 'holding', 'high-sticking', 'cross-checking', 'roughing', 'delay of game']


# ------------------------------------------------------------------ lineups
import json as _json, unicodedata as _ud, re as _re
try:
    DFO = _json.load(open('/home/claude/dfo_lines.json'))
except Exception:
    DFO = {}


def _norm(s):
    return ''.join(c for c in _ud.normalize('NFKD', s) if not _ud.combining(c)).upper().replace('.', '').replace("'", '').replace('-', ' ').strip()


def build_lineups_dfo(team, tid, roster, pids_market, players=None):
    """Real lines, pairs, power-play and penalty-kill units from Daily Faceoff (fetched Oct 16 2026)."""
    d = DFO[team]
    ids = {_norm(r['n']): r['id'] for r in roster[tid]}
    pos_of = {r['id']: r['pos'] for r in roster[tid]}
    numbers = d.get('numbers', {}); unmatched = []
    def g(n):
        k = _norm(n)
        if k in ids:
            return ids[k]
        cands = [i for kk, i in ids.items() if kk.split()[-1] == k.split()[-1]]
        if len(cands) == 1:                                   # same last name, only one on the roster: treat as the same person
            unmatched.append(n + ' matched by last name'); return cands[0]
        sid = 'dfo:' + k.replace(' ', '_')                    # not on the roster feed yet (call-up or trade): add a placeholder
        ids[k] = sid; pos_of[sid] = None
        roster[tid].append({'id': sid, 'n': n, 'no': numbers.get(n), 'pos': None})
        if players is not None:
            players[sid] = {'name': n, 'team': tid, 'pos': None}
        unmatched.append(n + ' is not on the roster feed (placeholder added)')
        return sid
    out_names = {_norm(_re.sub(r'\(.*?\)', '', n)) for n in d.get('out', [])}
    used = {g(n) for ln in d['lines'] for n in ln if n}
    usedD = {g(n) for pr in d['pairs'] for n in pr if n}
    spare = [r['id'] for r in roster[tid] if r['pos'] not in ('D', 'DS', 'G') and r['id'] not in used and _norm(r['n']) not in out_names]
    spare.sort(key=lambda i: 0 if pos_of[i] == 'C' else 1)
    spareD = [r['id'] for r in roster[tid] if r['pos'] in ('D', 'DS') and r['id'] not in usedD and _norm(r['n']) not in out_names]
    def take_f(want_c=False):
        nonlocal spare
        pick = next((i for i in spare if (pos_of[i] == 'C') == want_c), spare[0] if spare else None)
        spare = [i for i in spare if i != pick]
        unmatched.append('Daily Faceoff left a forward slot empty; filled with a spare dressed skater')
        return pick
    def take_d():
        nonlocal spareD
        pick = spareD[0] if spareD else None
        spareD = spareD[1:]
        unmatched.append('Daily Faceoff left a defense slot empty; filled with a spare defenseman')
        return pick
    lines = []
    for lw, c, rw in d['lines']:                     # [C, LW, RW]
        cid = g(c) if c else take_f(True)
        lines.append([cid, g(lw) if lw else take_f(), g(rw) if rw else take_f()])
    pairs = [[g(a) if a else take_d(), g(b) if b else take_d()] for a, b in d['pairs']]
    Dset = {p for pr in pairs for p in pr}
    dressed_ids = {p for ln in lines for p in ln} | Dset
    def unit(names):
        ids_ = [g(n) for n in names if n]
        for n_, i_ in zip([n for n in names if n], ids_):
            if i_ not in dressed_ids:      # listed on a power-play or penalty-kill unit but not in the lines or pairs: he is not in the game's player list
                unmatched.append(f'Daily Faceoff lists {n_} on a special-teams unit but not in the lineup; left out of that unit')
        ids_ = [i for i in ids_ if i in dressed_ids]
        return [i for i in ids_ if i not in Dset] + [i for i in ids_ if i in Dset]
    pp = [unit(u) for u in d['pp']]; pk = [unit(u) for u in d['pk']]
    F = [p for ln in lines for p in ln]; D = [p for pr in pairs for p in pr]
    dressed = F + D
    extra = [p for p in dressed if p not in pids_market]
    byname = {r['id']: r['n'] for r in roster[tid]}
    posgrp = {}
    for ln in lines:
        posgrp[ln[0]] = 'C'; posgrp[ln[1]] = 'LW'; posgrp[ln[2]] = 'RW'
    for p_ in D:
        posgrp[p_] = 'D'
    weights = {}
    if STATS is not None:
        def share(groups, getter, default):
            vals = []
            for grp in groups:
                xs = [getter(byname.get(p_, '')) for p_ in grp]; xs = [x for x in xs if x]
                vals.append(float(np.mean(xs)) if len(xs) >= max(1, len(grp) - 1) else None)
            if sum(v is not None for v in vals) < len(vals) - 0:
                return None
            a = np.array(vals); return a / a.sum()
        wF = share(lines, STATS.ev, None); wD = share(pairs, STATS.ev, None)
        wP = share(pp, lambda n: (STATS.pp(n) or 0) + 1e-6, None); wK = share(pk, lambda n: (STATS.sh(n) or 0) + 1e-6, None)
        weights = {'wF': wF, 'wD': wD, 'wPP': wP, 'wPK': wK}
    return {'tid': tid, 'pids': list(pids_market) + extra, 'extra': extra, 'F': F, 'D': D, 'lines': lines, 'pairs': pairs, 'pp': pp, 'pk': pk, 'posgrp': posgrp, 'names': byname, **weights,
            'pos': pos_of, 'prod': {}, 'centers': {ln[0] for ln in lines}, 'source': 'Daily Faceoff ' + d.get('updated', ''), 'unmatched': unmatched}


def build_lineups(game, fit, players, roster, pg, pa):
    """Per side: lines, pairs, power-play and penalty-kill units, from market production. Returns dict side->lineup."""
    pts = ns.player_marginals(game, 'player_points', 0.5)
    ppp = ns.player_marginals(game, 'player_power_play_points', 0.5)
    pos_of = {}
    for tid, lst in roster.items():
        for r in lst:
            pos_of[r['id']] = r['pos']
    out = {}
    for side, tid in (('home', fit['hid']), ('away', fit['aid'])):
        pids = [p for p in pg if players.get(p, {}).get('team') == tid]
        if fit[side] in DFO:
            out[side] = build_lineups_dfo(fit[side], tid, roster, pids, players)
            continue
        prod = {p: pts.get(p, pg.get(p, 0) * 1.6) for p in pids}
        F = sorted([p for p in pids if pos_of.get(p) not in ('D', 'DS', 'G')], key=lambda p: -prod[p])
        D = sorted([p for p in pids if pos_of.get(p) in ('D', 'DS')], key=lambda p: -prod[p])
        extraF = [r['id'] for r in roster[tid] if r['pos'] not in ('D', 'DS', 'G') and r['id'] not in pids]
        extraD = [r['id'] for r in roster[tid] if r['pos'] in ('D', 'DS') and r['id'] not in pids]
        extra = []
        # depth forwards complete the bottom line; depth defensemen complete the bottom pair
        while len(F) % 3 and extraF:
            p = extraF.pop(0); F.append(p); extra.append(p); prod[p] = 0.0
        while len(F) < 12 and extraF:
            p = extraF.pop(0); F.append(p); extra.append(p); prod[p] = 0.0
        while len(D) % 2 and extraD:
            p = extraD.pop(0); D.append(p); extra.append(p); prod[p] = 0.0
        while len(D) < 6 and extraD:
            p = extraD.pop(0); D.append(p); extra.append(p); prod[p] = 0.0
        lines = []
        for i in range(0, len(F), 3):
            ch = F[i:i + 3]
            c = next((p for p in ch if pos_of.get(p) == 'C'), ch[0])
            lines.append([c] + [p for p in ch if p != c])
        if lines and len(lines[-1]) < 3:
            ln = lines[-1]
            for p in reversed(F[:-1]):
                if len(ln) >= 3:
                    break
                if p not in ln:
                    ln.append(p)
        lines = [ln for ln in lines if len(ln) == 3][:5]
        pairs = [D[i:i + 2] for i in range(0, len(D), 2)]
        if pairs and len(pairs[-1]) < 2:
            for p in reversed(D[:-1]):
                if len(pairs[-1]) >= 2:
                    break
                if p not in pairs[-1]:
                    pairs[-1].append(p)
        pairs = [pr for pr in pairs if len(pr) == 2][:4]
        # power play: top 3 F and top 2 D by PP-points market, else by production
        key = lambda p: -ppp.get(p, prod[p] * 0.45)
        pf = sorted(F, key=key); pd_ = sorted(D, key=key)
        pp1 = pf[:3] + pd_[:2]
        pp2 = pf[3:6] + pd_[2:4]
        # penalty kill: depth forwards (ranks 3-8) and both pairs
        pk1 = [F[3 % len(F)], F[5 % len(F)], D[0], D[2 % len(D)]]
        pk2 = [F[4 % len(F)], F[6 % len(F)], D[1 % len(D)], D[3 % len(D)]]
        out[side] = {'tid': tid, 'pids': pids + extra, 'extra': extra, 'F': F, 'D': D, 'lines': lines, 'pairs': pairs, 'pp': [pp1, pp2], 'pk': [pk1, pk2], 'pos': pos_of,
                     'prod': prod}
    return out


def unit_tables(lu, idx):
    """Category -> (cols (U,5) padded with -1, weights (U,)). cols are global column indices."""
    def pad(l):
        l = [idx[p] for p in l][:5]
        return l + [-1] * (5 - len(l))
    t = {}
    nl, npair = len(lu['lines']), len(lu['pairs'])
    fw = EV_F_W[:nl] / EV_F_W[:nl].sum(); dw = EV_D_W[:npair] / EV_D_W[:npair].sum()
    if lu.get('wF') is not None and len(lu['wF']) == nl: fw = np.asarray(lu['wF'])
    if lu.get('wD') is not None and len(lu['wD']) == npair: dw = np.asarray(lu['wD'])
    ev5, w5 = [], []
    lw = LATE_F_W[:nl] / LATE_F_W[:nl].sum(); ldw = LATE_D_W[:npair] / LATE_D_W[:npair].sum()
    late, wl = [], []
    for i, ln in enumerate(lu['lines']):
        for j, pr in enumerate(lu['pairs']):
            ev5.append(pad(ln + pr)); w5.append(fw[i] * dw[j])
            late.append(pad(ln + pr)); wl.append(lw[i] * ldw[j])
    t['EV5'] = (np.array(ev5), np.array(w5)); t['LATE'] = (np.array(late), np.array(wl))
    ppw = np.asarray(lu['wPP']) if lu.get('wPP') is not None and len(lu['wPP']) == len(lu['pp']) else PP_W
    pkw = np.asarray(lu['wPK']) if lu.get('wPK') is not None and len(lu['wPK']) == len(lu['pk']) else PK_W
    t['PP5'] = (np.array([pad(u) for u in lu['pp']]), ppw)
    t['PK4'] = (np.array([pad(u) for u in lu['pk']]), pkw)
    t['PP4'] = (np.array([pad(u[:2] + u[3:5]) for u in lu['pp']]), ppw)
    t['PK3'] = (np.array([pad((u[:1] + u[2:4]) if len(u) >= 4 else u[:3]) for u in lu['pk']]), pkw)      # a unit can be short when Daily Faceoff lists a player who is not dressed
    t['MIXF'] = (np.array([idx[p] for ln in lu['lines'] for p in ln]), np.repeat(fw, 3) / 3.0)
    t['MIXD'] = (np.array([idx[p] for pr in lu['pairs'] for p in pr]), np.repeat(dw, 2) / 2.0)
    ev4, w4 = [], []
    for i, ln in enumerate(lu['lines'][:2]):
        for j, pr in enumerate(lu['pairs'][:2]):
            ev4.append(pad(ln[:2] + pr)); w4.append(fw[i] * dw[j])
    t['EV4'] = (np.array(ev4), np.array(w4) / np.sum(w4))
    ot, wo = [], []
    F = lu['lines'][0] + lu['lines'][1]
    for a in range(min(4, len(F))):
        for b in range(a + 1, min(4, len(F))):
            for j, pr in enumerate(lu['pairs'][:2]):
                ot.append(pad([F[a], F[b], pr[0]])); wo.append((1 - 0.25 * (a + b)) * (0.6 if j == 0 else 0.4))
    wo = np.maximum(np.array(wo), 0.05)
    t['OT3'] = (np.array(ot), wo / wo.sum())
    return t


# ------------------------------------------------------------------ special-teams fit from the market
def special_teams_targets(game, fit, players):
    """Expected power-play goals and shorthanded goals per team from the posted team PPG / SHG lines."""
    out = {'ppg': {}, 'shg': {}}
    shrink = 0.93      # one-sided 'over' prices: remove roughly 7% book margin
    for mid, key in (('team_total_power_play_goals', 'ppg'), ('team_total_short_handed_goals', 'shg')):
        acc = collections.defaultdict(list)
        for b, rows in game['odds'].items():
            for r in rows:
                if r['market_id'] == mid and r['team_id'] and r['points'] == 0.5 and r['selection_line'] == 'over':
                    side = 'home' if r['team_id'] == fit['hid'] else 'away'
                    acc[side].append(imp(r['price']) * shrink)
        for side, v in acc.items():
            p1 = float(np.clip(np.median(v), 0.02, 0.95))
            out[key][side] = -math.log(1 - p1)       # Poisson mean implied by P(at least one)
    out['src'] = {k: {s: 'books' for s in out[k]} for k in ('ppg', 'shg')}
    if STATS is not None and hasattr(STATS, 'mp_st_defaults'):
        lgp = (STATS.mp_lg or {}).get('ppgPerTeamGame')
        for side in ('home', 'away'):
            d = STATS.mp_st_defaults(side)          # the league rate, scaled by this matchup (penalties each team takes and draws, power-play and shorthanded scoring)
            if side in out['ppg']:
                # the posted 'over 0.5 power-play goals' prices carry a large margin: they imply well over the real league rate. Scale them to the league level (they keep their
                # team-to-team differences), then average with the matchup estimate.
                books = out['ppg'][side] * ((lgp / PPG_BOOK_LEVEL) if lgp else 1.0)
                out['ppg'][side] = 0.5 * books + 0.5 * d[0] if d else books
                out['src']['ppg'][side] = 'books + MoneyPuck' if d else 'books'
            elif d:
                out['ppg'][side] = d[0]; out['src']['ppg'][side] = 'MoneyPuck'
            if d:
                if side in out['shg']:
                    out['shg'][side] = 0.5 * out['shg'][side] + 0.5 * d[1]; out['src']['shg'][side] = 'books + MoneyPuck'
                else:
                    out['shg'][side] = d[1]; out['src']['shg'][side] = 'MoneyPuck'
    return out


def strength_table(kpp, kpk):
    T = np.ones((3, 3))   # index [sX-3, sY-3]
    T[2, 2] = 1.0
    T[2, 1] = kpp; T[2, 0] = kpp * 1.8
    T[1, 2] = kpk; T[0, 2] = kpk * 0.25
    T[1, 1] = 1.05; T[1, 0] = kpp * 0.9; T[0, 1] = kpk * 0.5; T[0, 0] = 1.0
    return T


# ------------------------------------------------------------------ the time-stepped game
def run_timeline(P, n, slot, rng, record):
    S = int(58 * 60 / slot)
    x = np.ones(n) if P['pace_k'] is None else rng.gamma(P['pace_k'], 1.0 / P['pace_k'], size=n)
    sides = ('home', 'away')
    pen_rem = {s: np.zeros((n, K_PEN)) for s in sides}
    pen_cnt = {s: np.zeros(n, dtype=np.int64) for s in sides}
    pen_start = {s: np.full((n, K_PEN), np.nan) for s in sides} if record else None
    pen_end = {s: np.full((n, K_PEN), np.nan) for s in sides} if record else None
    tab = {s: strength_table(P['kpp'][s], P['kpk'][s]) for s in sides}
    pi = {s: P['mu_pen'][s] * slot / (58 * 60.0) for s in sides}
    cnt = {s: np.zeros(n, dtype=np.int64) for s in sides}
    ppg = {s: np.zeros(n, dtype=np.int64) for s in sides}
    shg = {s: np.zeros(n, dtype=np.int64) for s in sides}
    rec = {'sim': [], 't': [], 'team': [], 'sx': [], 'sy': []} if record else None
    for si in range(S):
        t0 = si * slot
        seg = 0 if t0 < 600 else (1 if t0 < 1200 else (2 if t0 < 2400 else 3))
        m = P['mult'][seg]
        for s in sides:
            st = (rng.random(n) < pi[s]) & (pen_cnt[s] < K_PEN)
            ix = np.nonzero(st)[0]
            if len(ix):
                k = pen_cnt[s][ix]
                pen_rem[s][ix, k] = PP_SECONDS
                if record:
                    pen_start[s][ix, k] = t0 + rng.random(len(ix)) * slot
                pen_cnt[s][ix] += 1
        sk = {s: 5 - np.minimum((pen_rem[s] > 0).sum(axis=1), 2) for s in sides}
        scored = {}
        for s in sides:
            o = 'away' if s == 'home' else 'home'
            f = tab[s][sk[s] - 3, sk[o] - 3]
            rate = P['rho'][s] * m * x * f
            scored[s] = rng.random(n) < (1 - np.exp(-rate * slot))
        for s in sides:
            o = 'away' if s == 'home' else 'home'
            rows = np.nonzero(scored[s])[0]
            if not len(rows):
                continue
            cnt[s][rows] += 1
            if record:
                rec['sim'].append(rows); rec['t'].append(t0 + rng.random(len(rows)) * slot)
                rec['team'].append(np.full(len(rows), 0 if s == 'home' else 1)); rec['sx'].append(sk[s][rows]); rec['sy'].append(sk[o][rows])
            pp_rows = rows[(sk[s][rows] > sk[o][rows]) & ((pen_rem[o][rows] > 0).any(axis=1))]
            sh_rows = rows[sk[s][rows] < sk[o][rows]]
            ppg[s][pp_rows] += 1; shg[s][sh_rows] += 1
            if len(pp_rows):
                masked = np.where(pen_rem[o][pp_rows] > 0, pen_rem[o][pp_rows], np.inf)
                k = masked.argmin(axis=1)
                if record:
                    pen_end[o][pp_rows, k] = t0 + slot / 2
                pen_rem[o][pp_rows, k] = 0.0
        for s in sides:
            before = pen_rem[s] > 0
            pen_rem[s] = np.maximum(pen_rem[s] - slot, 0.0)
            if record:
                ended = before & (pen_rem[s] == 0) & np.isnan(pen_end[s])
                pen_end[s][ended] = t0 + slot
    out = {'base_h': cnt['home'], 'base_a': cnt['away'], 'x': x, 'ppg': ppg, 'shg': shg}
    if record:
        out.update({'rec': {k: (np.concatenate(v) if v else np.array([])) for k, v in rec.items()}, 'pen_start': pen_start, 'pen_end': pen_end, 'pen_cnt': pen_cnt})
    return out


def finalize(base_h, base_a, P, rng, n):
    """Late-game goalie-pull goals after 58:00, then overtime / shootout. Returns arrays and late-goal info."""
    q, r = P['q'], P['r']; r2 = 0.5 * r
    d = base_h - base_a
    u = rng.random(n)
    x = P.get('x_late', np.ones(n))
    a_h = np.minimum(P['rho']['home'] * 120 * x, 0.5); a_a = np.minimum(P['rho']['away'] * 120 * x, 0.5)
    lh = np.zeros(n, dtype=bool); la = np.zeros(n, dtype=bool)
    lh |= (d == 0) & (u < a_h); la |= (d == 0) & (u >= a_h) & (u < a_h + a_a)
    la |= (d == 1) & (u < q); lh |= (d == 1) & (u >= q) & (u < q + r)
    lh |= (d == -1) & (u < q); la |= (d == -1) & (u >= q) & (u < q + r)
    lh |= (d == 2) & (u < r2); la |= (d == -2) & (u < r2)
    reg_h = base_h + lh; reg_a = base_a + la
    tie = reg_h == reg_a
    ot_goal = tie & (rng.random(n) < npp.P_OT_GOAL)
    ph = P['rho']['home'] / (P['rho']['home'] + P['rho']['away'])
    home_ot = ot_goal & (rng.random(n) < ph); away_ot = ot_goal & ~home_ot
    so_home = tie & ~ot_goal & (rng.random(n) < 0.5)
    hg = reg_h + home_ot; ag = reg_a + away_ot
    margin = hg - ag
    margin = np.where(margin == 0, np.where(so_home, 1, -1), margin)
    return {'late_h': lh, 'late_a': la, 'reg_h': reg_h, 'reg_a': reg_a, 'tie': tie, 'home_ot': home_ot, 'away_ot': away_ot,
            'so_home': so_home, 'hg': hg, 'ag': ag, 'margin': margin, 'd58': d}


# ------------------------------------------------------------------ calibration of the game-level parameters
def calibrate(game, fit, players, prof, st_targets, seed=5, n=45000, slot=20.0, verbose=False):
    tg = fit['targets']
    mult = np.array(prof['mult'])
    base_rho = {'home': fit['lam_h'] / (58 * 60.0), 'away': fit['lam_a'] / (58 * 60.0)}

    def params(theta):
        sh, sa, q, r = theta
        rho = {'home': base_rho['home'] * sh, 'away': base_rho['away'] * sa}
        kpp, kpk = {}, {}
        mu = {'home': pen_mu('home'), 'away': pen_mu('away')}
        for s in ('home', 'away'):
            o = 'away' if s == 'home' else 'home'
            m_pp = st_targets['ppg'].get(s, 0.75); m_sh = st_targets['shg'].get(s, 0.12)
            p_pp = float(np.clip(m_pp / mu[o], 0.02, 0.9))          # power plays this team gets are the other team's penalties
            kpp[s] = float(np.clip(1.10 * -math.log(1 - p_pp) / (rho[s] * PP_SECONDS), 1.1, 6.0))
            kpk[s] = float(np.clip(1.15 * m_sh / (mu[s] * 105.0 * rho[s]), 0.1, 1.2))
        return {'rho': rho, 'kpp': kpp, 'kpk': kpk, 'mu_pen': mu, 'mult': mult, 'pace_k': fit.get('pace_k'), 'q': q, 'r': r}

    def loss(theta):
        if not (0.4 < theta[0] < 2.5 and 0.4 < theta[1] < 2.5 and 0 <= theta[2] <= 0.6 and 0 <= theta[3] <= 0.95):
            return 1e3
        rng = np.random.default_rng(seed)
        P = params(theta)
        tl = run_timeline(P, n, slot, rng, False)
        fin = finalize(tl['base_h'], tl['base_a'], P, rng, n)
        e = 4 * (float((fin['margin'] > 0).mean()) - tg['ml']) ** 2
        if tg['draw'] is not None:
            e += 4 * (float(fin['tie'].mean()) - tg['draw']) ** 2
        tot = fin['hg'] + fin['ag']
        for line, p in tg['totals'].items():
            gt = (tot > line).mean(); lt = (tot < line).mean()
            e += (gt / (gt + lt) - p) ** 2
        for hp, p in tg['puck'].items():
            m_ = fin['margin'] + hp
            up = (m_ > 0).mean(); dn = (m_ < 0).mean()
            e += (up / (up + dn) - p) ** 2
        return float(e)

    best = minimize(loss, [1.0, 1.0, fit['q_tie'], fit['r_en']], method='Nelder-Mead',
                    options={'xatol': 0.004, 'fatol': 2e-6, 'maxfev': 110, 'initial_simplex': None})
    th = best.x
    P = params(th)
    P['theta'] = th.tolist(); P['loss'] = float(best.fun); P['nfev'] = int(best.nfev); P['version'] = CAL_VERSION
    return P


# ------------------------------------------------------------------ players: goals and assists from on-ice units
CAT = {'EV5': 0, 'PP5': 1, 'PK4': 2, 'EV4': 3, 'PP4': 4, 'PK3': 5, 'OT3': 6, 'LATE': 7}


def cat_of(sx, sy):
    c = np.full(len(sx), CAT['EV5'])
    c[(sx == 5) & (sy < 5)] = CAT['PP5']
    c[(sx == 4) & (sy == 5)] = CAT['PK4']
    c[(sx == 3) & (sy >= 4)] = CAT['PK3']
    c[(sx == 4) & (sy == 4)] = CAT['EV4']
    c[(sx == 3) & (sy == 3)] = CAT['EV4']
    c[(sx == 4) & (sy == 3)] = CAT['PP4']
    return c


def simulate_v3(game, fit, players, roster, prof, P, n=200_000, slot=10.0, seed=11, ast_probs=None):
    rng = np.random.default_rng(seed)
    ast_probs = ast_probs or AST_PROBS
    pg = dict(ns.goal_marginals(game))
    pa = dict(ns.player_marginals(game, 'player_assists', 0.5))
    lu = build_lineups(game, fit, players, roster, pg, pa)
    import select_extract as se_ga
    gdiag = []
    all_pids = lu['home']['pids'] + lu['away']['pids']
    G_F = [0.30, 0.22, 0.15, 0.10, 0.05]; A_F = [0.40, 0.32, 0.22, 0.15, 0.08]
    G_D = [0.10, 0.06, 0.04, 0.02]; A_D = [0.34, 0.26, 0.17, 0.08]
    for side_ in ('home', 'away'):
        L_ = lu[side_]
        for p_ in L_['pids']:
            if p_ in pg and p_ in pa:
                continue
            if p_ in L_['D']:
                k = next((i for i, pr in enumerate(L_['pairs']) if p_ in pr), len(G_D) - 1); slot_g, slot_a = G_D[min(k, len(G_D) - 1)], A_D[min(k, len(A_D) - 1)]
            else:
                k = next((i for i, ln in enumerate(L_['lines']) if p_ in ln), len(G_F) - 1); slot_g, slot_a = G_F[min(k, len(G_F) - 1)] * (1.15 if any(p_ in u for u in L_['pp'][:1]) else 1.0), A_F[min(k, len(A_F) - 1)]
            if STATS is not None:      # the player's own season rates, shrunk toward the slot prior by games played
                nm_ = L_['names'].get(p_) or players[p_]['name']
                rg, _ = STATS.rate(nm_, 'g', -math.log(1 - slot_g)); ra, _ = STATS.rate(nm_, 'a', -math.log(1 - slot_a) / 0.9)
                gm_ = 1.0
                if GA_MATCHUP and p_ not in pg:      # no goal price: tilt the player's goal rate by how the opponent defends his position (ShotPropz goals against)
                    grp_ = L_['posgrp'].get(p_); opp_ = se_ga.TEAMS.get(fit['away' if side_ == 'home' else 'home'], {}).get('abbr')
                    if grp_ and opp_:
                        gm_ = STATS.goal_matchup(grp_, opp_) if hasattr(STATS, 'goal_matchup') else 1.0
                    gdiag.append({'name': nm_, 'pos': grp_, 'vs': opp_, 'mult': round(gm_, 3)})
                slot_g, slot_a = 1 - math.exp(-rg * gm_), 1 - math.exp(-0.9 * ra)
            if p_ not in pg: pg[p_] = slot_g
            if p_ not in pa: pa[p_] = slot_a
    all_pids = lu['home']['pids'] + lu['away']['pids']
    idx = {p: i for i, p in enumerate(all_pids)}
    Pn = len(all_pids)
    tabs = {s: unit_tables(lu[s], idx) for s in ('home', 'away')}

    tl = run_timeline(P, n, slot, rng, True)
    Pf = dict(P); Pf['x_late'] = tl['x']
    fin = finalize(tl['base_h'], tl['base_a'], Pf, rng, n)
    rec = tl['rec']

    # ---- goal records: regular, late (goalie pull), overtime
    g_sim = [rec['sim'].astype(np.int64)]; g_t = [rec['t']]; g_side = [rec['team'].astype(np.int64)]
    g_cat = [cat_of(rec['sx'].astype(np.int64), rec['sy'].astype(np.int64))]
    for side_code, key in ((0, 'late_h'), (1, 'late_a')):
        rows = np.nonzero(fin[key])[0]
        g_sim.append(rows); g_t.append(3480 + 120 * rng.random(len(rows))); g_side.append(np.full(len(rows), side_code)); g_cat.append(np.full(len(rows), CAT['LATE']))
    for side_code, key in ((0, 'home_ot'), (1, 'away_ot')):
        rows = np.nonzero(fin[key])[0]
        g_sim.append(rows); g_t.append(3600 + 300 * rng.random(len(rows))); g_side.append(np.full(len(rows), side_code)); g_cat.append(np.full(len(rows), CAT['OT3']))
    g_sim = np.concatenate(g_sim); g_t = np.concatenate(g_t); g_side = np.concatenate(g_side); g_cat = np.concatenate(g_cat)
    G = len(g_sim)

    # ---- on-ice unit for each goal, drawn from the ice-time weights for that strength state
    cols = np.full((G, 5), -1, dtype=np.int64)
    unit_id = np.zeros(G, dtype=np.int64)
    for s_code, s in ((0, 'home'), (1, 'away')):
        for cname, c in CAT.items():
            m = (g_side == s_code) & (g_cat == c)
            k = int(m.sum())
            if not k:
                continue
            u_cols, u_w = tabs[s][cname]
            ui = np.minimum(np.searchsorted(np.cumsum(u_w / u_w.sum()), rng.random(k)), len(u_w) - 1)
            cols[m] = u_cols[ui]; unit_id[m] = ui
            if cname == 'EV5' and MIX > 0:                      # line juggling: some 5v5 goals come from a mixed group of forwards and defensemen
                pos_ = np.nonzero(m)[0]; pos_ = pos_[rng.random(k) < MIX]
                if len(pos_):
                    fc, fw_ = tabs[s]['MIXF']; dc, dw_ = tabs[s]['MIXD']
                    gf = np.log(fw_)[None, :] - np.log(-np.log(rng.random((len(pos_), len(fw_)))))
                    gd = np.log(dw_)[None, :] - np.log(-np.log(rng.random((len(pos_), len(dw_)))))
                    cols[pos_, :3] = fc[np.argsort(-gf, axis=1)[:, :3]]; cols[pos_, 3:5] = dc[np.argsort(-gd, axis=1)[:, :2]]

    # ---- scorer weights fitted (IPF) so P(goal) matches the market for every listed player
    target_g = np.array([pg[p] for p in all_pids])
    w = np.array([-math.log(1 - min(t, 0.9)) for t in target_g])
    u_sc = rng.random(G)
    valid = cols >= 0
    safe = np.where(valid, cols, 0)

    def draw_scorer(w):
        W = np.where(valid, w[safe], 0.0)
        cum = np.cumsum(W, axis=1)
        pick = (u_sc * cum[:, -1])[:, None]
        k = np.minimum((cum < pick).sum(axis=1), 4)
        return safe[np.arange(G), k]

    for _ in range(7):
        sc = draw_scorer(w)
        cnt = np.bincount(g_sim * Pn + sc, minlength=n * Pn).reshape(n, Pn)
        m_i = np.maximum((cnt >= 1).mean(axis=0), 1e-4)
        w = np.clip(w * (target_g / m_i) ** 0.9, 0.01, 50)
    scorer = draw_scorer(w)
    goals = np.bincount(g_sim * Pn + scorer, minlength=n * Pn).reshape(n, Pn).astype(np.int16)

    # ---- assists from the same unit
    target_a = np.array([pa.get(p, 0.05) for p in all_pids])
    al = np.array([-math.log(1 - min(t, 0.9)) for t in target_a])
    n_ast = np.searchsorted(np.cumsum(ast_probs), rng.random(G))
    gum = -np.log(-np.log(rng.random((G, 5))))
    on_scorer = (cols == scorer[:, None])

    def draw_assists(al):
        keys = np.log(np.maximum(al[safe], 1e-9)) + gum
        keys = np.where(valid & ~on_scorer, keys, -np.inf)
        order = np.argsort(-keys, axis=1)[:, :2]
        out = []
        for slot_i in (0, 1):
            ok = (n_ast > slot_i) & np.isfinite(keys[np.arange(G), order[:, slot_i]])
            out.append(np.where(ok, safe[np.arange(G), order[:, slot_i]], -1))
        return out

    for _ in range(7):
        a1, a2 = draw_assists(al)
        ac = np.zeros(n * Pn)
        for a_ in (a1, a2):
            ok = a_ >= 0
            ac += np.bincount(g_sim[ok] * Pn + a_[ok], minlength=n * Pn)
        m_a = np.maximum((ac.reshape(n, Pn) >= 1).mean(axis=0), 1e-4)
        al = np.clip(al * (target_a / m_a) ** 0.9, 0.01, 50)
    a1, a2 = draw_assists(al)
    assists = np.zeros(n * Pn)
    for a_ in (a1, a2):
        ok = a_ >= 0
        assists += np.bincount(g_sim[ok] * Pn + a_[ok], minlength=n * Pn)
    assists = assists.reshape(n, Pn).astype(np.int16)

    # ---- first goal scorer, period counts, first ten
    order = np.lexsort((g_t, g_sim))
    sim_s, sc_s = g_sim[order], scorer[order]
    first_idx = np.searchsorted(sim_s, np.arange(n))
    has = first_idx < len(sim_s)
    has[has] = sim_s[first_idx[has]] == np.arange(n)[has]
    first_col = np.full(n, -2, dtype=np.int32); first_col[has] = sc_s[first_idx[has]]
    is_h = g_side == 0
    g1h = np.bincount(g_sim[(g_t < 1200) & is_h], minlength=n); g1a = np.bincount(g_sim[(g_t < 1200) & ~is_h], minlength=n)
    n10 = np.bincount(g_sim[g_t < 600], minlength=n)

    teams = {s: {'tid': lu[s]['tid'], 'pids': lu[s]['pids'], 'lam': fit['lam_h' if s == 'home' else 'lam_a']} for s in ('home', 'away')}
    sim = {'idx': idx, 'pids': all_pids, 'goals': goals, 'assists': assists, 'teams': teams, 'pg': pg, 'pa': pa,
           'hg': fin['hg'], 'ag': fin['ag'], 'reg_h': fin['reg_h'], 'reg_a': fin['reg_a'], 'tie': fin['tie'], 'so_home': fin['so_home'],
           'first_col': first_col, 'n10': n10, 'g1h': g1h, 'g1a': g1a, 'fit': fit, 'prof': prof, 'margin': fin['margin'], 'home_win': fin['margin'] > 0,
           'lu': lu, 'P': P, 'w_goal': w, 'w_ast': al,
           'goal_sim': g_sim, 'goal_t': g_t / 60.0, 'goal_team': (1 - g_side), 'goal_cat': g_cat, 'goal_unit': unit_id, 'goal_cols': cols,
           'goal_scorer': scorer, 'goal_a1': a1, 'goal_a2': a2,
           'pen_start': tl['pen_start'], 'pen_end': tl['pen_end'], 'pen_cnt': tl['pen_cnt'], 'ppg': tl['ppg'], 'shg': tl['shg'], 'd58': fin['d58'],
           'late_h': fin['late_h'], 'late_a': fin['late_a'], 'x_pace': tl['x'], 'goal_matchup_diag': gdiag}
    return sim


GA_MATCHUP = True
SHOT_ANCHOR = True; SHOT_ANCHOR_MIN = 4; SHOT_R_DEFAULT = 0.93
SHOT_SAVES_ANCHOR = True; SHOT_COVER = 1.0; SHOT_MP_ANCHOR = True


def add_shots_and_saves(game, fit, players, sim):
    """Shots = goals + saved shots fitted to SOG lines; goalie saves thinned to the saves lines (same as nhl_props)."""
    n = len(sim['hg'])
    rng = np.random.default_rng(31)
    saved_means, flags = npp.fit_saved_means(game, players, sim, sim['pids'])
    goalies = npp.fit_goalies(game, players)
    for tid_, tname_ in ((fit['hid'], fit['home']), (fit['aid'], fit['away'])):
        if any(g_['team'] == tid_ for g_ in goalies.values()):
            continue
        names_ = DFO.get(tname_, {}).get('goalies', [])
        pid_ = next((p_ for p_, info_ in players.items() if info_['team'] == tid_ and names_ and _norm(info_['name']) == _norm(names_[0])), None)
        if pid_ is None:
            pid_ = next((p_ for p_, info_ in players.items() if info_['team'] == tid_ and info_['pos'] == 'G'), None)
        if pid_ is not None:
            goalies[pid_] = {'team': tid_, 'M': 26.5, 'name': players[pid_]['name'], 'n_lines': 0, 'default': True}
    if STATS is not None:      # players with no shots price: their own shots per game, tilted by how the opponent defends their position (ShotPropz)
        import select_extract as se_
        SLOT_S_F = [2.6, 2.1, 1.6, 1.2, 1.0]; SLOT_S_D = [2.0, 1.6, 1.2, 1.0]
        # anchor to the market: on each team, the priced players show how far the market's shot expectation sits from their season rates
        # (new season, lineup and matchup effects). Unpriced players get the same ratio, so a team's shots are not inflated by season rates alone.
        anchor = {}
        for side_ in ('home', 'away'):
            L_ = sim['lu'][side_]; opp_abbr = se_.TEAMS.get(fit['away' if side_ == 'home' else 'home'], {}).get('abbr')
            num = den = msum = 0.0; k_ = 0
            for pid in L_['pids']:
                if flags.get(pid) is not None or pid not in sim['idx'] or L_['posgrp'].get(pid) is None:
                    continue
                grp = L_['posgrp'][pid]
                if grp == 'D':
                    kk = next((i for i, pr in enumerate(L_['pairs']) if pid in pr), len(SLOT_S_D) - 1); base = SLOT_S_D[min(kk, len(SLOT_S_D) - 1)]
                else:
                    kk = next((i for i, ln in enumerate(L_['lines']) if pid in ln), len(SLOT_S_F) - 1); base = SLOT_S_F[min(kk, len(SLOT_S_F) - 1)]
                spg, _ = STATS.rate(L_['names'].get(pid) or players[pid]['name'], 's', base)
                num += float(sim['goals'][:, sim['idx'][pid]].mean()) + saved_means[pid]; den += spg; msum += STATS.matchup(grp, opp_abbr); k_ += 1
            r_ = min(1.25, max(0.6, num / den)) if (k_ >= SHOT_ANCHOR_MIN and den > 0) else SHOT_R_DEFAULT
            anchor[side_] = {'r': r_, 'mu': (msum / k_) if k_ else 1.0, 'n': k_}
        sim['shot_anchor'] = anchor
        for side_ in ('home', 'away'):
            L_ = sim['lu'][side_]; opp_abbr = se_.TEAMS.get(fit['away' if side_ == 'home' else 'home'], {}).get('abbr')
            for pid in L_['pids']:
                if flags.get(pid) != 'no SOG line; default' or pid not in sim['idx']:
                    continue
                grp = L_['posgrp'].get(pid)
                if grp is None:
                    continue
                if grp == 'D':
                    k = next((i for i, pr in enumerate(L_['pairs']) if pid in pr), len(SLOT_S_D) - 1); base = SLOT_S_D[min(k, len(SLOT_S_D) - 1)]
                else:
                    k = next((i for i, ln in enumerate(L_['lines']) if pid in ln), len(SLOT_S_F) - 1); base = SLOT_S_F[min(k, len(SLOT_S_F) - 1)]
                spg, _ = STATS.rate(L_['names'].get(pid) or players[pid]['name'], 's', base)
                mu_ = STATS.matchup(grp, opp_abbr)
                saved_means[pid] = max(0.1, spg * (mu_ / anchor[side_]['mu'] if SHOT_ANCHOR else mu_) * (anchor[side_]['r'] if SHOT_ANCHOR else 1.0) - float(sim['goals'][:, sim['idx'][pid]].mean()))
                flags[pid] = 'stats + ShotPropz matchup'
    teams = sim['teams']
    # a posted saves line says how many shots the goalie faces (saves + goals allowed), which is the sharpest check on the other team's total shots.
    # Players with their own shots price keep it; the unpriced players are scaled so the team's saved shots add up to that goalie's saves mean.
    sim['shot_saves_anchor'] = {}
    if SHOT_SAVES_ANCHOR:
        for s_ in ('home', 'away'):
            og_ = [g_ for g_ in goalies.values() if g_['team'] == (fit['aid'] if s_ == 'home' else fit['hid']) and not g_.get('default')]
            mp_ = STATS.mp_team_shots(s_) if (SHOT_MP_ANCHOR and STATS is not None and hasattr(STATS, 'mp_team_shots')) else None
            if og_:
                target_saved, src_ = SHOT_COVER * og_[0]['M'], 'saves line'
            elif mp_ is not None:       # no saves line: MoneyPuck's shots for and against (saves are shots minus goals)
                target_saved, src_ = mp_ - float(sim['hg' if s_ == 'home' else 'ag'].mean()), 'MoneyPuck'
            else:
                continue
            unp = [p_ for p_ in teams[s_]['pids'] if flags.get(p_) == 'stats + ShotPropz matchup']
            fixed = sum(saved_means[p_] for p_ in teams[s_]['pids'] if p_ not in unp); cur = sum(saved_means[p_] for p_ in unp)
            if cur <= 0:
                continue
            f_ = min(1.3, max(0.35, (target_saved - fixed) / cur))
            for p_ in unp:
                saved_means[p_] = max(0.1, saved_means[p_] * f_)
            sim['shot_saves_anchor'][s_] = {'target': round(float(target_saved), 2), 'source': src_, 'factor': f_, 'unpriced': len(unp)}
    shots = sim['goals'].astype(np.int32).copy()
    saved_side = {'home': np.zeros(n, dtype=np.int64), 'away': np.zeros(n, dtype=np.int64)}
    sums_raw = {s: sum(saved_means[p] for p in teams[s]['pids']) for s in ('home', 'away')}
    thin = {}
    for s in ('home', 'away'):
        og = [g for g in goalies.values() if g['team'] == (fit['aid'] if s == 'home' else fit['hid'])]
        M = og[0]['M'] if og else None
        thin[s] = 1.0 if (M is None or sums_raw[s] <= 0.92 * M) else (0.92 * M) / sums_raw[s]
    for pid in sim['pids']:
        side = 'home' if players[pid]['team'] == fit['hid'] else 'away'
        sv = rng.poisson(saved_means[pid], size=n)
        shots[:, sim['idx'][pid]] += sv; saved_side[side] += sv
    saves, gl = {}, {}
    for gpid, g in goalies.items():
        faces = 'away' if g['team'] == fit['hid'] else 'home'
        part = rng.binomial(saved_side[faces], thin[faces])
        other = max(g['M'] - sums_raw[faces] * thin[faces], 0.5)
        gl[gpid] = {'M': g['M'], 'sog_implied_saved_mean': sums_raw[faces], 'thinning': thin[faces], 'other_mean': other}
        saves[gpid] = part + rng.poisson(other, size=n)
    sim['shot_flags'] = flags
    sim['shots'] = shots; sim['saves'] = saves; sim['goalie_diag'] = gl; sim['saved_means'] = saved_means
    return goalies


def add_extra_penalties(sim, seed=77):
    """Penalty counts on the page include every penalty, as real counts do. The game itself only has the ones that create a power play (they set the power-play time,
    which matches the league), so the offsetting ones that do not are added here, per team, so the average count matches the matchup's expected penalties."""
    rng = np.random.default_rng(seed)
    lg = (STATS.mp_lg.get('penPerTeamGame') if (STATS is not None and hasattr(STATS, 'mp_lg')) else None) or LEAGUE_PEN
    sim['pen_cnt_pp'] = {s: sim['pen_cnt'][s].copy() for s in ('home', 'away')}
    sim['pen_target'] = {}
    for s in ('home', 'away'):
        r = STATS.mp_pen_rate(s) if (STATS is not None and hasattr(STATS, 'mp_pen_rate')) else None
        target = lg * (r if r else 1.0)
        extra = max(0.0, target - float(sim['pen_cnt_pp'][s].mean()))
        sim['pen_cnt'][s] = sim['pen_cnt_pp'][s] + rng.poisson(extra, size=len(sim['pen_cnt_pp'][s]))
        sim['pen_target'][s] = round(float(target), 3)


def run_game_v3(game, players, roster, n=200_000, verbose=True, cache_path=None):
    import time
    t0 = time.time()
    fit = npp.fit_game_v2(game, players)
    prof = npp.fit_time_profile(game, fit)
    st = special_teams_targets(game, fit, players)
    import os, pickle
    cache = cache_path if cache_path and os.path.exists(cache_path) else None
    P = pickle.load(open(cache, 'rb')) if cache else None
    if P is not None and P.get('version') != CAL_VERSION:      # saved before the model changed
        P = None
    if P is None:
        P = calibrate(game, fit, players, prof, st)
        if cache_path:
            pickle.dump(P, open(cache_path, 'wb'))
    if verbose:
        print('  calibrated', {k: (round(v, 3) if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()}) for k, v in P.items() if k in ('rho', 'kpp', 'kpk', 'q', 'r', 'loss', 'nfev')}, round(time.time() - t0), 's')
    sim = simulate_v3(game, fit, players, roster, prof, P, n=n)
    add_extra_penalties(sim)
    sim['matchup_use'] = {'pen': {s: {'mu': round(float(P['mu_pen'][s]), 3)} for s in ('home', 'away')}, 'st': {k: {s: {'v': round(float(st[k][s]), 3), 'src': st.get('src', {}).get(k, {}).get(s, 'default')} for s in st[k]} for k in ('ppg', 'shg')}}
    goalies = add_shots_and_saves(game, fit, players, sim)
    sim['matchup_use']['shots'] = sim.get('shot_saves_anchor')
    if verbose:
        print('  simulated', round(time.time() - t0), 's')
    return fit, prof, P, sim, goalies

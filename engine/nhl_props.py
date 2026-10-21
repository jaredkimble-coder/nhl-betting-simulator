#!/usr/bin/env python3
"""
nhl_props.py - prices every market group on the requested list from one simulation.

Extends nhl_sim.py (game fit, player goal/assist marginals) with:
  * goal timing           -> 1st goalscorer, goal in first ten, 1st period lines
  * shots layer           -> shots on goal = goals + saved shots, fitted to player SOG lines
  * goalie saves          -> saves = saved shots by the opposing team, fitted to saves lines
  * a generic pricer      -> any line of any market, whether or not a book posts it

Market groups covered:
  Moneyline | All Spreads | All Totals | All Team Totals | All Goalscorer Props | All Points Props |
  All Assist Props | All 1st Goalscorer | All Saves | Shots On Goal Props | 60 Min Line |
  Goal In First Ten | 1st Period Puck Line | 1st Period Total
"""
import collections
import json
import math
import sys
import numpy as np
import nhl_sim as ns
from nhl_sim import imp, devig_pair

RNG = np.random.default_rng(21)
EDGES = np.array([0.0, 10.0, 20.0, 40.0, 60.0])

GROUPS = {
    'moneyline': 'Moneyline',
    'puck_line': 'All Spreads',
    'total_goals': 'All Totals',
    'team_total': 'All Team Totals',
    'anytime_goal_scorer': 'All Goalscorer Props',
    'player_goals': 'All Goalscorer Props',
    'player_points': 'All Points Props',
    'player_assists': 'All Assist Props',
    'first_goal_scorer': 'All 1st Goalscorer',
    'player_saves': 'All Saves',
    'player_shots_on_goal': 'Shots On Goal Props',
    'moneyline_3-way': '60 Min Line',
    '1st_10_minutes_total_goals': 'Goal In First Ten',
    '1st_period_puck_line': '1st Period Puck Line',
    '1st_period_total_goals': '1st Period Total',
}
GROUP_ORDER = ['Moneyline', 'All Spreads', 'All Totals', 'All Team Totals', 'All Goalscorer Props', 'All Points Props',
               'All Assist Props', 'All 1st Goalscorer', 'All Saves', 'Shots On Goal Props', '60 Min Line',
               'Goal In First Ten', '1st Period Puck Line', '1st Period Total']
BOOK_LABEL = {'kalshi': 'Kalshi', 'polymarket': 'Polymarket', 'polymarket_usa_': 'Polymarket', 'draftkings': 'DraftKings',
              'fanduel': 'FanDuel', 'betmgm': 'BetMGM', 'caesars': 'Caesars'}
BOOK_COLS = ['Kalshi', 'Polymarket', 'DraftKings', 'FanDuel', 'BetMGM', 'Caesars']


def american(p):
    if p is None or not (0 < p < 1):
        return None
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def pois_pmf(mean, kmax=60):
    k = np.arange(kmax + 1)
    lf = np.array([math.lgamma(i + 1) for i in k])
    return np.exp(-mean + k * math.log(max(mean, 1e-12)) - lf)


# ---------------------------------------------------------------- game fit with end-of-game mechanics
from scipy.optimize import minimize
from scipy.stats import gamma as _gamma

KM = 16
PACE_K_GRID = [None, 60.0, 35.0, 22.0, 14.0, 9.0]
P_OT_GOAL = 0.6
F58_DEFAULT = 58.0 / 60.0


def pace_x(k, m=120):
    """Deterministic quantile grid of a mean-1 gamma pace factor (None = no shared pace factor)."""
    if k is None:
        return np.ones(1)
    x = _gamma.ppf((np.arange(m) + 0.5) / m, a=k, scale=1.0 / k)
    return x / x.mean()


def pois_rows(lams, kmax):
    ks = np.arange(kmax + 1)
    lf = np.array([math.lgamma(i + 1) for i in ks])
    L = np.maximum(lams, 1e-12)[:, None]
    return np.exp(-L + ks[None, :] * np.log(L) - lf[None, :])


def joint_final(lh, la, k, q, r, f58=F58_DEFAULT, kmax=KM):
    """Regulation score distribution: Poisson play for ~58 minutes, then goalie-pull mechanics.
    q = P(trailing-by-1 team ties), r = P(leader scores an empty-netter when up 1), r2 = 0.5*r when up 2."""
    x = pace_x(k)
    r2 = 0.5 * r
    ph = pois_rows(lh * x * f58, kmax)
    pa = pois_rows(la * x * f58, kmax)
    J = ph[:, :, None] * pa[:, None, :]
    K = kmax + 1
    D = np.arange(K)[:, None] - np.arange(K)[None, :]
    a_h = np.minimum(lh * x * (2 / 60.0), 0.5)[:, None, None]
    a_a = np.minimum(la * x * (2 / 60.0), 0.5)[:, None, None]
    Mh = a_h * (D == 0) + r * (D == 1) + r2 * (D == 2) + q * (D == -1)
    Ma = a_a * (D == 0) + r * (D == -1) + r2 * (D == -2) + q * (D == 1)
    new = np.zeros((len(x), K + 1, K + 1))
    new[:, :K, :K] += J * (1 - Mh - Ma)
    new[:, 1:, :K] += J * Mh
    new[:, :K, 1:] += J * Ma
    return new.mean(axis=0)


def outcome_dists(F, lh, la, p_ot=P_OT_GOAL):
    """From the regulation joint F -> distributions of draw, final margin (shootout = +/-1), total goals, team goals."""
    K1 = F.shape[0]
    OFF = K1 + 2
    margin = np.zeros(2 * OFF + 1)
    total = np.zeros(2 * K1 + 3)
    hg = np.zeros(K1 + 2)
    ag = np.zeros(K1 + 2)
    ph_ot = lh / (lh + la)
    draw = 0.0
    for i in range(K1):
        for j in range(K1):
            p = F[i, j]
            if p <= 0:
                continue
            if i != j:
                margin[OFF + i - j] += p; total[i + j] += p; hg[i] += p; ag[j] += p
            else:
                draw += p
                oh, oa, sh, sa = p * p_ot * ph_ot, p * p_ot * (1 - ph_ot), p * (1 - p_ot) * 0.5, p * (1 - p_ot) * 0.5
                margin[OFF + 1] += oh + sh; margin[OFF - 1] += oa + sa
                total[2 * i + 1] += oh + oa; total[2 * i] += sh + sa
                hg[i + 1] += oh; hg[i] += oa + sh + sa; ag[i + 1] += oa; ag[i] += oh + sh + sa
    return {'draw': draw, 'margin': margin, 'OFF': OFF, 'total': total, 'hg': hg, 'ag': ag,
            'p_home_win': float(margin[OFF + 1:].sum())}


def prob_over(pmf, line):
    ks = np.arange(len(pmf))
    gt = pmf[ks > line].sum(); lt = pmf[ks < line].sum()
    return float(gt / (gt + lt)) if gt + lt > 0 else None


def market_targets(game, hid, aid, players):
    mk = collect_market(game, {'hid': hid, 'aid': aid}, players)
    t = {'ml': [], 'draw': [], 'totals': collections.defaultdict(list), 'puck': collections.defaultdict(list)}
    for (mid, subj, line, side), books in mk.items():
        ps = [p for p, kind in books.values() if kind == 'devig']
        if not ps:
            continue
        if mid == 'moneyline' and side == 'home':
            t['ml'] += ps
        elif mid == 'moneyline_3-way' and side == 'draw':
            t['draw'] += ps
        elif mid == 'total_goals' and side == 'over' and line is not None and (line * 2) % 2 == 1:
            t['totals'][line] += ps
        elif mid == 'puck_line' and side == 'home' and line is not None and (line * 2) % 2 == 1:
            t['puck'][line] += ps
    return {'ml': float(np.mean(t['ml'])), 'draw': float(np.mean(t['draw'])) if t['draw'] else None,
            'totals': {l: float(np.mean(v)) for l, v in t['totals'].items()},
            'puck': {l: float(np.mean(v)) for l, v in t['puck'].items()}}


def fit_game_v2(game, players):
    base = ns.fit_game(game)
    hid, aid = base['hid'], base['aid']
    tg = market_targets(game, hid, aid, players)

    def model(theta, k):
        lh, la, q, r = theta
        F = joint_final(lh, la, k, q, r)
        return outcome_dists(F, lh, la)

    def loss(theta, k):
        d = model(theta, k)
        e = 4 * (d['p_home_win'] - tg['ml']) ** 2
        if tg['draw'] is not None:
            e += 4 * (d['draw'] - tg['draw']) ** 2
        for line, p in tg['totals'].items():
            mp = prob_over(d['total'], line)
            if mp is not None:
                e += (mp - p) ** 2
        for hp, p in tg['puck'].items():
            m = d['margin']; OFF = d['OFF']
            ks = np.arange(len(m)) - OFF
            up = m[(ks + hp) > 0].sum(); dn = m[(ks + hp) < 0].sum()
            e += (up / (up + dn) - p) ** 2
        return float(e)

    best = None
    for k in PACE_K_GRID:
        x0 = [base['lam_h'], base['lam_a'], 0.12, 0.35]
        res = minimize(loss, x0, args=(k,), method='L-BFGS-B', bounds=[(1.0, 6.0), (1.0, 6.0), (0.0, 0.6), (0.0, 0.95)])
        if best is None or res.fun < best[0]:
            best = (float(res.fun), k, res.x)
    err, k, th = best
    d = model(th, k)
    diag = {'loss': err, 'pace_k': k, 'p_home_win_model': d['p_home_win'], 'p_home_win_mkt': tg['ml'],
            'draw_model': d['draw'], 'draw_mkt': tg['draw'],
            'mean_total_model': float((np.arange(len(d['total'])) * d['total']).sum())}
    fit = dict(base)
    fit.update({'lam_h': float(th[0]), 'lam_a': float(th[1]), 'q_tie': float(th[2]), 'r_en': float(th[3]),
                'pace_k': k, 'mu': float(th[0] + th[1]), 'fit_diag': diag, 'targets': tg})
    return fit


# ---------------------------------------------------------------- time profile
def two_way_over(game, mid, lines=None):
    """line -> median devigged P(over) across books (two-sided only)."""
    acc = collections.defaultdict(list)
    for b, rows in game['odds'].items():
        sides = collections.defaultdict(dict)
        for r in rows:
            if r['market_id'] == mid and r['points'] is not None and r['selection_line'] in ('over', 'under') and not r['player_id'] and not r['team_id']:
                sides[r['points']][r['selection_line']] = r['price']
        for line, s_ in sides.items():
            if 'over' in s_ and 'under' in s_:
                acc[line].append(devig_pair(s_['over'], s_['under']))
    return {l: float(np.median(v)) for l, v in acc.items()}


EDGES58 = np.array([0.0, 10.0, 20.0, 40.0, 58.0])


def fit_time_profile(game, fit):
    """Scoring intensity by segment [0-10, 10-20, 20-40, 40-60] fitted to first-ten and 1st-period totals,
    under the same shared pace factor as the game fit."""
    lam = fit['lam_h'] + fit['lam_a']
    x = pace_x(fit.get('pace_k'))
    ten = two_way_over(game, '1st_10_minutes_total_goals').get(0.5)
    p1 = {l: p for l, p in two_way_over(game, '1st_period_total_goals').items() if (l * 2) % 2 == 1}
    m1 = 1.0
    notes = []
    if ten:
        lo, hi = 0.3, 2.0
        for _ in range(50):
            mid_ = (lo + hi) / 2
            if float(np.mean(1 - np.exp(-lam * x * mid_ / 6.0))) < ten:
                lo = mid_
            else:
                hi = mid_
        m1 = float(np.clip((lo + hi) / 2, 0.4, 1.8))
        notes.append(f'first-ten fitted to market P(goal)={ten:.3f}')
    m2 = 1.0
    if p1:
        best = (1e9, 1.0)
        for cand in np.linspace(0.3, 1.9, 161):
            mu = lam * (10 * m1 + 10 * cand) / 60.0
            err = 0.0
            for l, p in p1.items():
                pm = pois_rows(mu * x, 25)
                err += (float(pm[:, int(l + 0.5):].sum(axis=1).mean()) - p) ** 2
            if err < best[0]:
                best = (err, float(cand))
        m2 = best[1]
        notes.append('1st-period total ladder fitted')
    m3 = (60 - 10 * m1 - 10 * m2) / 40.0
    # the 2nd and 3rd periods share the rest of the game's scoring (total goals stay as the game fit has them), but the books price them differently:
    # fit how that remainder splits between the two periods to their own totals ladders
    mp2 = mp3 = m3
    q2 = {l: p for l, p in two_way_over(game, '2nd_period_total_goals').items() if (l * 2) % 2 == 1}
    q3 = {l: p for l, p in two_way_over(game, '3rd_period_total_goals').items() if (l * 2) % 2 == 1}
    if q2 or q3:
        best = (1e9, 0.5)
        for sh in np.linspace(0.30, 0.70, 81):
            a2, a3 = 2 * m3 * sh, 2 * m3 * (1 - sh); err = 0.0
            for lad, mp in ((q2, a2), (q3, a3)):
                pm = pois_rows(lam * (20 * mp) / 60.0 * x, 25)
                for l, p in lad.items():
                    err += (float(pm[:, int(l + 0.5):].sum(axis=1).mean()) - p) ** 2
            if err < best[0]:
                best = (err, float(sh))
        mp2, mp3 = 2 * m3 * best[1], 2 * m3 * (1 - best[1])
        notes.append(f'2nd/3rd period totals fitted (2nd period takes {best[1]:.0%} of the remaining scoring)')
    mult = np.array([m1, m2, mp2, mp3])
    w58 = mult * np.diff(EDGES58)
    return {'mult': mult.tolist(), 'weights58': (w58 / w58.sum()).tolist(), 'f58': float(w58.sum() / 60.0), 'notes': notes}


def sample_times(k, weights):
    seg = np.minimum(np.searchsorted(np.cumsum(weights), RNG.random(k)), 3)
    return EDGES58[seg] + RNG.random(k) * (EDGES58[seg + 1] - EDGES58[seg])


# ---------------------------------------------------------------- shots / saves fits
SOG_TWO_SIDED_W = 6.0


def fit_saved_means(game, players, sim0, pids):
    """Mean saved shots per listed player so that shots = goals + saved matches the SOG lines."""
    lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    mk = {l: ns.player_marginals(game, 'player_shots_on_goal', l) for l in lines}
    means, flags = {}, {}
    grid = np.arange(0.02, 8.0, 0.02)
    pm_grid = [pois_pmf(s, 25) for s in grid]
    # lines a book posts on both sides are properly de-vigged; one-sided ladders carry the book's margin, so the fit leans on the two-sided lines
    two_sided = {}
    for l in lines:
        sides = collections.defaultdict(lambda: collections.defaultdict(set))
        for b, rows in game['odds'].items():
            for r in rows:
                if r['market_id'] == 'player_shots_on_goal' and r['player_id'] and r['points'] == l and r['selection_line'] in ('over', 'under'):
                    sides[(b, r['player_id'])]['s'].add(r['selection_line'])
        two_sided[l] = {pid_ for (b, pid_), v in sides.items() if len(v['s']) == 2}
    for pid in pids:
        i = sim0['idx'][pid]
        gcount = np.bincount(np.minimum(sim0['goals'][:, i], 4), minlength=5) / len(sim0['goals'])
        pts = [(l, mk[l][pid], (SOG_TWO_SIDED_W if pid in two_sided[l] else 1.0)) for l in lines if pid in mk[l]]
        if not pts:
            means[pid] = 1.5; flags[pid] = 'no SOG line; default'
            continue
        best = (1e9, 1.5)
        for s, pm in zip(grid, pm_grid):
            cdf_tail = np.concatenate([[1.0], 1 - np.cumsum(pm)])  # P(Saved >= m) = cdf_tail[m]
            err = 0.0
            for l, p, w_ in pts:
                k = int(l + 0.5)
                prob = sum(gcount[g] * (1.0 if k - g <= 0 else cdf_tail[k - g]) for g in range(5))
                err += w_ * (prob - p) ** 2
            if err < best[0]:
                best = (err, float(s))
        means[pid] = best[1]
    return means, flags


def fit_goalies(game, players):
    """goalie pid -> (team id, fitted mean saves) using all posted saves lines."""
    lines = sorted({r['points'] for rows in game['odds'].values() for r in rows if r['market_id'] == 'player_saves' and r['points'] is not None})
    mk = {l: ns.player_marginals(game, 'player_saves', l) for l in lines}
    pids = {p for m in mk.values() for p in m}
    out = {}
    grid = np.arange(8.0, 45.0, 0.1)
    for pid in pids:
        pts = [(l, mk[l][pid]) for l in lines if pid in mk[l]]
        best = (1e9, 27.0)
        for M in grid:
            pm = pois_pmf(M, 80)
            err = sum((pm[int(l + 0.5):].sum() - p) ** 2 for l, p in pts)
            if err < best[0]:
                best = (err, float(M))
        out[pid] = {'team': players.get(pid, {}).get('team'), 'M': best[1], 'name': players.get(pid, {}).get('name', pid), 'n_lines': len(pts)}
    return out


# ---------------------------------------------------------------- simulation
def simulate_full(game, fit, players, prof, aw_scale, n, saved_means=None, goalies=None,
                  ast_probs=(0.07, 0.27, 0.66), p_ot_goal=P_OT_GOAL):
    pg = ns.goal_marginals(game)
    pa = ns.player_marginals(game, 'player_assists', 0.5)
    teams = {}
    for side, tid, lam in (('home', fit['hid'], fit['lam_h']), ('away', fit['aid'], fit['lam_a'])):
        roster = [pid for pid in pg if players.get(pid, {}).get('team') == tid]
        teams[side] = {'tid': tid, 'lam': lam, 'pids': roster}
    all_pids = teams['home']['pids'] + teams['away']['pids']
    idx = {pid: i for i, pid in enumerate(all_pids)}
    P = len(all_pids)
    goals = np.zeros((n, P), dtype=np.int16)
    assists = np.zeros((n, P), dtype=np.int16)

    # --- regulation: pace factor, ~58 minutes of normal play, then goalie-pull mechanics
    k = fit.get('pace_k')
    x = np.ones(n) if k is None else RNG.gamma(k, 1.0 / k, size=n)
    f58 = prof['f58']
    q, r = fit['q_tie'], fit['r_en']
    r2 = 0.5 * r
    base = {'home': RNG.poisson(teams['home']['lam'] * x * f58), 'away': RNG.poisson(teams['away']['lam'] * x * f58)}
    d = base['home'] - base['away']
    u = RNG.random(n)
    a_h = np.minimum(teams['home']['lam'] * x * (2 / 60.0), 0.5)
    a_a = np.minimum(teams['away']['lam'] * x * (2 / 60.0), 0.5)
    late_h = np.zeros(n, dtype=bool); late_a = np.zeros(n, dtype=bool)
    # even: a goal by either side; home leads by 1: away ties (q) or home empty-netter (r); etc.
    late_h |= (d == 0) & (u < a_h)
    late_a |= (d == 0) & (u >= a_h) & (u < a_h + a_a)
    late_a |= (d == 1) & (u < q)
    late_h |= (d == 1) & (u >= q) & (u < q + r)
    late_h |= (d == -1) & (u < q)
    late_a |= (d == -1) & (u >= q) & (u < q + r)
    late_h |= (d == 2) & (u < r2)
    late_a |= (d == -2) & (u < r2)
    late = {'home': late_h.astype(np.int64), 'away': late_a.astype(np.int64)}
    reg = {s_: base[s_] + late[s_] for s_ in ('home', 'away')}

    for s_ in ('home', 'away'):
        t = teams[s_]
        rates = np.array([-math.log(1 - min(pg[p], 0.95)) for p in t['pids']])
        other = max(t['lam'] - rates.sum(), 0.04 * t['lam'])
        t['gw'] = np.append(rates, other)
        ar = np.array([-math.log(1 - min(pa.get(p, 0.05), 0.95)) for p in t['pids']])
        exp_ast = float(np.dot(ast_probs, [0, 1, 2])) * t['lam']
        other_a = max(exp_ast - ar.sum(), 0.10 * exp_ast)
        if aw_scale:
            ar = ar * np.array([aw_scale.get(p, 1.0) for p in t['pids']])
        t['aw'] = np.append(ar, other_a)

    tie = reg['home'] == reg['away']
    ot_goal = tie & (RNG.random(n) < p_ot_goal)
    home_ot = ot_goal & (RNG.random(n) < fit['lam_h'] / (fit['lam_h'] + fit['lam_a']))
    away_ot = ot_goal & ~home_ot
    so_home = tie & ~ot_goal & (RNG.random(n) < 0.5)
    tg = {'home': reg['home'] + home_ot, 'away': reg['away'] + away_ot}
    ot_flag = {'home': home_ot, 'away': away_ot}
    ast_cdf = np.cumsum(ast_probs)

    rec = {'sim': [], 't': [], 'col': [], 'team': [], 'a1': [], 'a2': []}
    n10 = {}; n20 = {}
    for si, side in enumerate(('away', 'home')):
        t = teams[side]
        Gt = tg[side]
        sim_idx = np.repeat(np.arange(n), Gt)
        total = len(sim_idx)
        starts = np.cumsum(Gt) - Gt
        rank = np.arange(total) - np.repeat(starts, Gt)
        nb = np.repeat(base[side], Gt)
        nl = np.repeat(late[side], Gt)
        times = np.where(rank < nb, sample_times(total, prof['weights58']),
                         np.where(rank < nb + nl, 58 + 2 * RNG.random(total), 60 + 5 * RNG.random(total)))
        n10[side] = np.bincount(sim_idx[times < 10], minlength=n)
        n20[side] = np.bincount(sim_idx[times < 20], minlength=n)
        npl = len(t['pids'])
        w = t['gw'] / t['gw'].sum()
        scorer = np.searchsorted(np.cumsum(w), RNG.random(total))
        listed = scorer < npl
        gcols = np.array([idx[p] for p in t['pids']]) if npl else np.array([], dtype=int)
        np.add.at(goals, (sim_idx[listed], gcols[scorer[listed]]), 1)
        col_global = np.where(listed, gcols[np.minimum(scorer, max(npl - 1, 0))] if npl else -1, -1)
        rec['sim'].append(sim_idx); rec['t'].append(times); rec['col'].append(col_global); rec['team'].append(np.full(total, si))
        n_ast = np.searchsorted(ast_cdf, RNG.random(total))
        aw = np.log(np.maximum(t['aw'], 1e-9))
        gumbel = -np.log(-np.log(RNG.random((total, npl + 1))))
        keys = aw[None, :] + gumbel
        mask = np.zeros_like(keys, dtype=bool)
        rows_ix = np.arange(total)[listed]
        mask[rows_ix, scorer[listed]] = True
        keys = np.where(mask, -np.inf, keys)
        order = np.argsort(-keys, axis=1)[:, :2]
        for slot in (0, 1):
            take = (n_ast > slot) & (order[:, slot] < npl)
            np.add.at(assists, (sim_idx[take], gcols[order[take, slot]]), 1)
        for slot, name_ in ((0, 'a1'), (1, 'a2')):
            who = np.where(n_ast > slot, np.where(order[:, slot] < npl, gcols[np.minimum(order[:, slot], max(npl - 1, 0))] if npl else -3, -3), -1)
            rec[name_].append(who)

    sim_all = np.concatenate(rec['sim']); t_all = np.concatenate(rec['t']); col_all = np.concatenate(rec['col'])
    order = np.lexsort((t_all, sim_all))
    sim_s, col_s = sim_all[order], col_all[order]
    first_idx = np.searchsorted(sim_s, np.arange(n))
    has = (first_idx < len(sim_s))
    has[has] = sim_s[first_idx[has]] == np.arange(n)[has]
    first_col = np.full(n, -2, dtype=np.int32)
    first_col[has] = col_s[first_idx[has]]

    out = {'idx': idx, 'pids': all_pids, 'goals': goals, 'assists': assists, 'teams': teams, 'pg': pg, 'pa': pa,
           'hg': tg['home'], 'ag': tg['away'], 'reg_h': reg['home'], 'reg_a': reg['away'], 'tie': tie,
           'so_home': so_home, 'first_col': first_col, 'n10': n10['home'] + n10['away'],
           'g1h': n20['home'], 'g1a': n20['away'], 'fit': fit, 'prof': prof,
           'goal_sim': sim_all, 'goal_t': t_all, 'goal_col': col_all, 'goal_team': np.concatenate(rec['team']),
           'goal_a1': np.concatenate(rec['a1']), 'goal_a2': np.concatenate(rec['a2']),
           'base_h': base['home'], 'base_a': base['away'], 'late_h': late['home'], 'late_a': late['away']}
    margin = tg['home'] - tg['away']
    margin = np.where(margin == 0, np.where(so_home, 1, -1), margin)
    out['margin'] = margin
    out['home_win'] = margin > 0
    if saved_means is not None:
        sums_raw = {s_: sum(saved_means[p] for p in teams[s_]['pids']) for s_ in ('home', 'away')}
        thin = {}
        for s_ in ('home', 'away'):
            opp_goalie = [g for g in (goalies or {}).values() if g['team'] == (fit['aid'] if s_ == 'home' else fit['hid'])]
            M = opp_goalie[0]['M'] if opp_goalie else None
            # shots stay faithful to the player SOG lines; goalie saves are thinned to the saves lines
            thin[s_] = 1.0 if (M is None or sums_raw[s_] <= 0.92 * M) else (0.92 * M) / sums_raw[s_]
        shots = goals.astype(np.int32).copy()
        saved_side = {'home': np.zeros(n, dtype=np.int64), 'away': np.zeros(n, dtype=np.int64)}
        for pid in all_pids:
            side = 'home' if players[pid]['team'] == fit['hid'] else 'away'
            sv = RNG.poisson(saved_means[pid], size=n)
            shots[:, idx[pid]] += sv
            saved_side[side] += sv
        saves = {}; gl_diag = {}
        for gpid, g in (goalies or {}).items():
            faces = 'away' if g['team'] == fit['hid'] else 'home'
            listed_part = RNG.binomial(saved_side[faces], thin[faces])
            other_mean = max(g['M'] - sums_raw[faces] * thin[faces], 0.5)
            gl_diag[gpid] = {'M': g['M'], 'sog_implied_saved_mean': sums_raw[faces], 'thinning': thin[faces], 'other_mean': other_mean}
            saves[gpid] = listed_part + RNG.poisson(other_mean, size=n)
        out['shots'] = shots; out['saves'] = saves; out['goalie_diag'] = gl_diag; out['saved_means'] = saved_means
    return out


# ---------------------------------------------------------------- pricer
def p_over(arr, line, over=True):
    """P(arr > line) or P(arr < line), push-adjusted for integer lines."""
    gt = (arr > line).mean(); lt = (arr < line).mean(); eq = 1 - gt - lt
    if eq > 1e-12:
        return float(gt / (gt + lt)) if over else float(lt / (gt + lt))
    return float(gt if over else lt)


def sim_prob(sim, key):
    """key = (market_id, subject, line, side). Returns the simulated probability of that selection."""
    mid, subj, line, side = key
    if mid == 'moneyline':
        h = sim['home_win'].mean()
        return float(h if side == 'home' else 1 - h)
    if mid == 'moneyline_3-way':
        rh, ra = sim['reg_h'], sim['reg_a']
        return float({'home': (rh > ra), 'draw': (rh == ra), 'away': (rh < ra)}[side].mean())
    if mid == 'puck_line':
        m = sim['margin'] + line  # line = home points
        return float((m > 0).mean() / ((m != 0).mean())) if side == 'home' else float((m < 0).mean() / ((m != 0).mean()))
    if mid == '1st_period_puck_line':
        m = sim['g1h'] - sim['g1a'] + line
        nz = (m != 0).mean()
        if nz == 0: return None
        return float((m > 0).mean() / nz) if side == 'home' else float((m < 0).mean() / nz)
    if mid == 'total_goals':
        return p_over(sim['hg'] + sim['ag'], line, side == 'over')
    if mid == '1st_period_total_goals':
        return p_over(sim['g1h'] + sim['g1a'], line, side == 'over')
    if mid == '1st_10_minutes_total_goals':
        return p_over(sim['n10'], line, side == 'over')
    if mid == 'team_total':
        return p_over(sim['hg'] if subj == 'home' else sim['ag'], line, side == 'over')
    if mid == 'anytime_goal_scorer':
        return float((sim['goals'][:, sim['idx'][subj]] >= 1).mean()) if subj in sim['idx'] else None
    if mid == 'first_goal_scorer':
        return float((sim['first_col'] == sim['idx'][subj]).mean()) if subj in sim['idx'] else None
    if mid in ('player_goals', 'player_assists', 'player_points', 'player_shots_on_goal'):
        if subj not in sim['idx']:
            return None
        i = sim['idx'][subj]
        arr = {'player_goals': sim['goals'][:, i], 'player_assists': sim['assists'][:, i],
               'player_points': sim['goals'][:, i] + sim['assists'][:, i], 'player_shots_on_goal': sim['shots'][:, i]}[mid]
        return p_over(arr, line, side == 'over')
    if mid == 'player_saves':
        arr = sim['saves'].get(subj)
        return None if arr is None else p_over(arr, line, side == 'over')
    return None


# ---------------------------------------------------------------- market side
def collect_market(game, fit, players):
    """key -> {book label: (devigged or raw prob, kind)} with keys shaped like sim_prob keys."""
    hid, aid = fit['hid'], fit['aid']
    out = collections.defaultdict(dict)
    for b, rows in game['odds'].items():
        label = BOOK_LABEL[b]
        groups = collections.defaultdict(dict)
        for r in rows:
            mid = r['market_id']
            if mid not in GROUPS:
                continue
            tid, pid, pts, sl = r['team_id'], r['player_id'], r['points'], r['selection_line']
            if mid == 'moneyline' and tid:
                groups[(mid, None, None)]['home' if tid == hid else 'away'] = r['price']
            elif mid == 'moneyline_3-way':
                side = 'home' if tid == hid else ('away' if tid == aid else 'draw')
                groups[(mid, None, None)][side] = r['price']
            elif mid in ('puck_line', '1st_period_puck_line') and tid and pts is not None:
                hp = pts if tid == hid else -pts
                groups[(mid, None, hp)]['home' if tid == hid else 'away'] = r['price']
            elif mid in ('total_goals', '1st_period_total_goals', '1st_10_minutes_total_goals') and pts is not None and sl in ('over', 'under'):
                groups[(mid, None, pts)][sl] = r['price']
            elif mid == 'team_total' and tid and pts is not None and sl in ('over', 'under'):
                groups[(mid, 'home' if tid == hid else 'away', pts)][sl] = r['price']
            elif mid in ('player_goals', 'player_assists', 'player_points', 'player_shots_on_goal', 'player_saves') and pid and pts is not None and sl in ('over', 'under'):
                groups[(mid, pid, pts)][sl] = r['price']
            elif mid in ('anytime_goal_scorer', 'first_goal_scorer') and pid:
                groups[(mid, pid, None)]['yes'] = r['price']
        for (mid, subj, line), sides in groups.items():
            if mid in ('anytime_goal_scorer', 'first_goal_scorer'):
                out[(mid, subj, None, 'yes')][label] = (imp(sides['yes']), 'raw')
            elif mid == 'moneyline_3-way':
                if len(sides) == 3:
                    tot = sum(imp(v) for v in sides.values())
                    for sd, v in sides.items():
                        out[(mid, None, None, sd)][label] = (imp(v) / tot, 'devig')
            else:
                names = list(sides)
                if len(names) == 2:
                    a, c = names
                    pa_, pc = imp(sides[a]), imp(sides[c])
                    out[(mid, subj, line, a)][label] = (pa_ / (pa_ + pc), 'devig')
                    out[(mid, subj, line, c)][label] = (pc / (pa_ + pc), 'devig')
                else:
                    sd = names[0]
                    out[(mid, subj, line, sd)][label] = (imp(sides[sd]), 'raw')
    return out


STD_LINES = {
    'puck_line': [-2.5, -1.5, 1.5, 2.5],
    'total_goals': [4.5, 5.5, 6.5, 7.5, 8.5],
    'team_total': [1.5, 2.5, 3.5, 4.5],
    '1st_period_puck_line': [-1.5, -0.5, 0.5, 1.5],
    '1st_period_total_goals': [0.5, 1.5, 2.5, 3.5],
    '1st_10_minutes_total_goals': [0.5, 1.5],
}


def standard_keys(sim, fit, players):
    keys = set()
    keys |= {('moneyline', None, None, 'home'), ('moneyline', None, None, 'away')}
    keys |= {('moneyline_3-way', None, None, s) for s in ('home', 'draw', 'away')}
    for l in STD_LINES['puck_line']:
        keys |= {('puck_line', None, l, 'home'), ('puck_line', None, l, 'away')}
    for l in STD_LINES['1st_period_puck_line']:
        keys |= {('1st_period_puck_line', None, l, 'home'), ('1st_period_puck_line', None, l, 'away')}
    for mid in ('total_goals', '1st_period_total_goals', '1st_10_minutes_total_goals'):
        for l in STD_LINES[mid]:
            keys |= {(mid, None, l, 'over'), (mid, None, l, 'under')}
    for l in STD_LINES['team_total']:
        for s in ('home', 'away'):
            keys |= {('team_total', s, l, 'over'), ('team_total', s, l, 'under')}
    for pid in sim['pids']:
        keys |= {('anytime_goal_scorer', pid, None, 'yes'), ('first_goal_scorer', pid, None, 'yes')}
        for l in (0.5, 1.5):
            keys |= {('player_goals', pid, l, 'over'), ('player_goals', pid, l, 'under'), ('player_assists', pid, l, 'over'), ('player_assists', pid, l, 'under')}
        for l in (0.5, 1.5, 2.5):
            keys |= {('player_points', pid, l, 'over'), ('player_points', pid, l, 'under')}
        for l in (1.5, 2.5, 3.5):
            keys |= {('player_shots_on_goal', pid, l, 'over'), ('player_shots_on_goal', pid, l, 'under')}
    for gpid, g in sim['saves'].items():
        arr = g
        mu = float(arr.mean())
        for l in np.arange(math.floor(mu) - 4.5, math.floor(mu) + 6.5, 1.0):
            keys |= {('player_saves', gpid, float(l), 'over'), ('player_saves', gpid, float(l), 'under')}
    return keys


def build_grid(sim, mk, fit, players):
    keys = set(mk) | standard_keys(sim, fit, players)
    rows = []
    for key in keys:
        mid, subj, line, side = key
        sp = sim_prob(sim, key)
        if sp is None:
            continue
        books = mk.get(key, {})
        label = ''
        if mid in ('moneyline', 'moneyline_3-way'):
            label = {'home': fit['home'], 'away': fit['away'], 'draw': 'Draw (tied after 60)'}[side]
        elif mid in ('puck_line', '1st_period_puck_line'):
            label = (fit['home'] if side == 'home' else fit['away'])
        elif mid == 'team_total':
            label = (fit['home'] if subj == 'home' else fit['away'])
        elif mid in ('total_goals', '1st_period_total_goals', '1st_10_minutes_total_goals'):
            label = 'Game'
        else:
            label = players.get(subj, {}).get('name', subj)
        if mid in ('puck_line', '1st_period_puck_line'):
            shown_line = line if side == 'home' else -line
        else:
            shown_line = line
        diffs = [sp - p for p, kind in books.values() if kind == 'devig']
        row = {
            'Group': GROUPS[mid], 'Market': mid, 'Subject': label,
            'Selection': side.title() if mid not in ('anytime_goal_scorer', 'first_goal_scorer') else 'Yes',
            'Line': shown_line, 'Sim probability': round(sp, 4), 'Sim fair price': american(sp),
            'Books posting': len(books),
        }
        for bl in BOOK_COLS:
            v = books.get(bl)
            row[bl] = round(v[0], 4) if v else None
        row['Sim minus market (avg)'] = round(float(np.mean(diffs)), 4) if diffs else None
        row['Market type'] = ('two-sided, devigged' if diffs else ('one-sided, vig included' if books else 'no book posts this'))
        row['_key'] = key
        rows.append(row)
    rows.sort(key=lambda r: (GROUP_ORDER.index(r['Group']), r['Market'], str(r['Subject']), r['Line'] if r['Line'] is not None else -99, r['Selection']))
    return rows


def run_game(game, players, n=300_000):
    fit = fit_game_v2(game, players)
    prof = fit_time_profile(game, fit)
    scale = ns.calibrate_assists(game, fit, players)
    sim0 = simulate_full(game, fit, players, prof, scale, n=80_000)
    saved_means, sflags = fit_saved_means(game, players, sim0, sim0['pids'])
    goalies = fit_goalies(game, players)
    sim = simulate_full(game, fit, players, prof, scale, n=n, saved_means=saved_means, goalies=goalies)
    mk = collect_market(game, fit, players)
    grid = build_grid(sim, mk, fit, players)
    return fit, prof, sim, goalies, grid, sflags


if __name__ == '__main__':
    games, players = ns.load(sys.argv[1], sys.argv[2])
    for g, game in games.items():
        fit, prof, sim, goalies, grid, sflags = run_game(game, players)
        print(g, fit['home'], len(grid), 'rows; time profile', [round(x, 2) for x in prof['mult']], prof['notes'])

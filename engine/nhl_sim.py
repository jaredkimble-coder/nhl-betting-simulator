#!/usr/bin/env python3
"""
NHL market-anchored game simulator + correlated parlay pricer.

Inputs (JSON saved from the n8n OpticOdds workflow):
  odds_raw.json     execution data containing 'Get Odds per Book' (7 books x N games)
  players_raw.json  roster responses from 'Get Team Players'

Method
  1. Game level: devig moneyline + totals across books, fit Poisson team goal rates.
  2. Player level: devig two-sided player goal / assist lines (Kalshi, Polymarket US, DK, FD ...)
     into marginal probabilities; convert to scoring and assisting rates.
  3. Simulate regulation goals per team, allocate each goal to a scorer and up to two
     assisters (Plackett-Luce), add OT goals when tied, record player stats per sim.
  4. Price multi-leg combos as the share of simulated games where every leg hits, and
     compare to the product of single-leg prices (the "independent" price).

Everything is derived from the market inputs; swap in your own ATG model rates by
replacing player_goal_rates() output.
"""
import json
import math
import sys
import collections
import numpy as np

RNG = np.random.default_rng(7)
BOOKS = ['kalshi', 'polymarket', 'polymarket_usa_', 'draftkings', 'fanduel', 'betmgm', 'caesars']
N_SIMS = 400_000


# ---------- odds helpers ----------
def imp(a):
    return 100.0 / (a + 100.0) if a > 0 else -a / (-a + 100.0)


def devig_pair(a, b):
    pa, pb = imp(a), imp(b)
    return pa / (pa + pb)


def poisson_pmf_matrix(lam_h, lam_a, kmax=16):
    k = np.arange(kmax + 1)
    fact = np.array([math.factorial(int(i)) for i in k], dtype=float)
    ph = np.exp(-lam_h) * lam_h ** k / fact
    pa = np.exp(-lam_a) * lam_a ** k / fact
    return np.outer(ph, pa)


def p_home_win(lam_h, lam_a):
    m = poisson_pmf_matrix(lam_h, lam_a)
    win = np.tril(m, -1).sum()  # rows = home goals > away goals
    tie = np.trace(m)
    return win + 0.5 * tie


def p_total_over(mu, line):
    k = np.arange(0, 30)
    fact = np.array([math.factorial(int(i)) for i in k], dtype=float)
    pmf = np.exp(-mu) * mu ** k / fact
    return pmf[k > line].sum()


def bisect(f, lo, hi, target, it=60):
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------- load ----------
def load(odds_path, players_path):
    d = json.load(open(odds_path))
    runs = d['data']['resultData']['runData']['Get Odds per Book'][0]['data']['main'][0]
    games = {}
    for i, it in enumerate(runs):
        fx = it['json']['data'][0]
        g = i // len(BOOKS)
        games.setdefault(g, {'fx': fx, 'odds': {}})['odds'][BOOKS[i % len(BOOKS)]] = fx['odds']
    players = {}
    for resp in json.load(open(players_path)):
        for p in resp.get('data', []):
            players[p['id']] = {'name': p['name'], 'team': p['team']['id'], 'pos': p.get('position')}
    return games, players


# ---------- game level ----------
PACE_GRID = [None, 80.0, 50.0, 30.0, 20.0, 14.0, 10.0, 7.0, 5.0]
_Q = np.linspace(0.0005, 0.9995, 800)


def pace_samples(k):
    """Deterministic multiplicative pace factors (mean 1) from a Gamma(k, 1/k) quantile grid."""
    if k is None:
        return np.ones(1)
    r = np.random.default_rng(11)
    x = r.gamma(k, 1.0 / k, size=4000)
    x.sort()
    return x[np.linspace(0, 3999, 400).astype(int)]


def over_prob(mu, line, x):
    ks = np.arange(0, 30)
    logf = np.array([math.lgamma(i + 1) for i in ks])
    lam = np.outer(x, np.ones(1)) * mu  # (nx, 1)
    pmf = np.exp(-lam + ks[None, :] * np.log(lam) - logf[None, :])
    return float(pmf[:, ks > line].sum(axis=1).mean())


def p_home_win_pace(lam_h, lam_a, x):
    return float(np.mean([p_home_win(lam_h * xi, lam_a * xi) for xi in x[::8]]))


def fit_game(game):
    fx = game['fx']
    home = fx['home_competitors'][0]
    away = fx['away_competitors'][0]
    hid, aid = home['id'], away['id']
    p_h, ladder = [], []
    for b, rows in game['odds'].items():
        ml = {r['team_id']: r['price'] for r in rows if r['market_id'] == 'moneyline' and r['team_id']}
        if hid in ml and aid in ml:
            p_h.append(devig_pair(ml[hid], ml[aid]))
        lines = collections.defaultdict(dict)
        for r in rows:
            if r['market_id'] == 'total_goals' and r['points'] is not None and r['selection_line'] in ('over', 'under'):
                lines[r['points']][r['selection_line']] = r['price']
        for line, sides in lines.items():
            if 'over' in sides and 'under' in sides and (line * 2) % 2 == 1:
                po = devig_pair(sides['over'], sides['under'])
                if 0.05 < po < 0.95:
                    ladder.append((line, po))
    ph = float(np.mean(p_h))
    L = np.array([l for l, _ in ladder])
    PO = np.array([p for _, p in ladder])
    mus = np.linspace(4.5, 8.5, 161)
    best = None
    fits = {}
    for k in PACE_GRID:
        x = pace_samples(k)
        sse_by_mu = [float(np.sum([(over_prob(m, l, x) - p) ** 2 for l, p in ladder])) for m in mus]
        j = int(np.argmin(sse_by_mu))
        fits[str(k)] = round(sse_by_mu[j], 4)
        if best is None or sse_by_mu[j] < best[2]:
            best = (k, float(mus[j]), sse_by_mu[j])
    k, mu, _ = best
    x = pace_samples(k)
    lam_h = bisect(lambda lh: p_home_win_pace(lh, mu - lh, x), 0.3, mu - 0.3, ph, it=30)
    lam_a = mu - lam_h
    # main total line = the .5 line with devigged over closest to 50% at DraftKings/FanDuel/Kalshi
    mb = None
    for line in sorted(set(L)):
        po = float(np.mean(PO[L == line]))
        if mb is None or abs(po - 0.5) < abs(mb[1] - 0.5):
            mb = (float(line), po)
    return {'hid': hid, 'aid': aid, 'home': home['name'], 'away': away['name'],
            'p_home_ml': ph, 'mu': mu, 'lam_h': lam_h, 'lam_a': lam_a, 'pace_k': k,
            'ladder_sse_by_k': fits, 'main_total': mb[0], 'n_ml_books': len(p_h), 'n_ladder_points': len(ladder)}


# ---------- player marginals ----------
def player_marginals(game, market_id, line):
    """pid -> median devigged P(over line) across books with both sides; fallback one-sided with shrink."""
    two = collections.defaultdict(list)
    one = collections.defaultdict(dict)
    for b, rows in game['odds'].items():
        sides = collections.defaultdict(dict)
        for r in rows:
            if r['market_id'] == market_id and r['player_id'] and r['points'] == line and r['selection_line'] in ('over', 'under'):
                sides[r['player_id']][r['selection_line']] = r['price']
        for pid, s in sides.items():
            if 'over' in s and 'under' in s:
                two[pid].append(devig_pair(s['over'], s['under']))
            elif 'over' in s:
                one[pid][b] = imp(s['over'])
    out = {pid: float(np.median(v)) for pid, v in two.items()}
    # shrink factor for one-sided prices: median(two-sided / one-sided) over overlap
    ratios = []
    for pid, bk in one.items():
        if pid in out:
            ratios += [out[pid] / p for p in bk.values()]
    k = float(np.median(ratios)) if ratios else 0.9
    for pid, bk in one.items():
        if pid not in out:
            out[pid] = float(np.median([p * k for p in bk.values()]))
    return out


def goal_marginals(game):
    m = player_marginals(game, 'player_goals', 0.5)
    # anytime goal scorer is one-sided; blend in with shrink
    atg = collections.defaultdict(dict)
    for b, rows in game['odds'].items():
        for r in rows:
            if r['market_id'] == 'anytime_goal_scorer' and r['player_id']:
                atg[r['player_id']][b] = imp(r['price'])
    ratios = [m[p] / v for p, bk in atg.items() if p in m for v in bk.values()]
    k = float(np.median(ratios)) if ratios else 0.9
    for pid, bk in atg.items():
        if pid not in m:
            m[pid] = float(np.median([v * k for v in bk.values()]))
    return m


# ---------- simulation ----------
def simulate(game, fit, players, n=N_SIMS, ast_probs=(0.07, 0.27, 0.66), p_ot_goal=0.6, aw_scale=None):
    pg = goal_marginals(game)
    pa = player_marginals(game, 'player_assists', 0.5)
    teams = {}
    for side, tid, lam in (('home', fit['hid'], fit['lam_h']), ('away', fit['aid'], fit['lam_a'])):
        roster = [pid for pid in pg if players.get(pid, {}).get('team') == tid]
        teams[side] = {'tid': tid, 'lam': lam, 'pids': roster}
    all_pids = teams['home']['pids'] + teams['away']['pids']
    idx = {pid: i for i, pid in enumerate(all_pids)}
    P = len(all_pids)

    goals = np.zeros((n, P), dtype=np.int16)
    assists = np.zeros((n, P), dtype=np.int16)
    reg = {}
    diag = {}
    k = fit.get('pace_k')
    pace = np.ones(n) if k is None else RNG.gamma(k, 1.0 / k, size=n)
    for side in ('home', 'away'):
        t = teams[side]
        reg[side] = RNG.poisson(t['lam'] * pace, size=n)
        rates = np.array([-math.log(1 - min(pg[p], 0.95)) for p in t['pids']])
        other = max(t['lam'] - rates.sum(), 0.04 * t['lam'])
        diag[side] = {'sum_player_rate': float(rates.sum()), 'team_lam': float(t['lam']),
                      'listed_share': float(rates.sum() / t['lam'])}
        t['gw'] = np.append(rates, other)
        arates = np.array([-math.log(1 - min(pa.get(p, 0.05), 0.95)) for p in t['pids']])
        exp_ast = float(np.dot(ast_probs, [0, 1, 2])) * t['lam']
        other_a = max(exp_ast - arates.sum(), 0.10 * exp_ast)
        if aw_scale:
            arates = arates * np.array([aw_scale.get(p, 1.0) for p in t['pids']])
        t['aw'] = np.append(arates, other_a)

    # tie -> overtime
    tie = reg['home'] == reg['away']
    ot_goal = tie & (RNG.random(n) < p_ot_goal)
    home_wins_ot = RNG.random(n) < fit['lam_h'] / (fit['lam_h'] + fit['lam_a'])
    ot_home = ot_goal & home_wins_ot
    ot_away = ot_goal & ~home_wins_ot
    so_home_win = tie & ~ot_goal & (RNG.random(n) < 0.5)

    team_goals = {'home': reg['home'] + ot_home, 'away': reg['away'] + ot_away}
    ast_cdf = np.cumsum(ast_probs)

    for side in ('home', 'away'):
        t = teams[side]
        G = team_goals[side]
        sim_idx = np.repeat(np.arange(n), G)
        total = len(sim_idx)
        if total == 0 or not t['pids']:
            continue
        w = t['gw'] / t['gw'].sum()
        scorer = np.searchsorted(np.cumsum(w), RNG.random(total))
        listed = scorer < len(t['pids'])
        gcols = np.array([idx[p] for p in t['pids']])
        # goals
        s_sim = sim_idx[listed]
        s_col = gcols[scorer[listed]]
        np.add.at(goals, (s_sim, s_col), 1)
        # assists: count per goal, then Plackett-Luce choice among teammates excluding scorer
        n_ast = np.searchsorted(ast_cdf, RNG.random(total))
        aw = np.log(np.maximum(t['aw'], 1e-9))
        npl = len(t['pids'])
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

    out = {
        'players': all_pids, 'idx': idx, 'goals': goals, 'assists': assists,
        'home_goals': team_goals['home'], 'away_goals': team_goals['away'],
        'home_win': (team_goals['home'] > team_goals['away']) | so_home_win,
        'total': team_goals['home'] + team_goals['away'],
        'went_ot': tie, 'ot_goal': ot_goal,
        'diag': diag, 'teams': teams, 'pg': pg, 'pa': pa,
    }
    return out


# ---------- reporting ----------
def name(players, pid):
    return players[pid]['name'] if pid in players else pid


def calibration(sim, players, top=8):
    rows = []
    for pid in sim['players']:
        i = sim['idx'][pid]
        g = (sim['goals'][:, i] >= 1).mean()
        a = (sim['assists'][:, i] >= 1).mean()
        p = ((sim['goals'][:, i] + sim['assists'][:, i]) >= 1).mean()
        rows.append((name(players, pid), sim['pg'].get(pid), g, sim['pa'].get(pid), a, p))
    rows.sort(key=lambda r: -(r[1] or 0))
    return rows[:top]


def fair_cents(p):
    return 100 * p


def combos(sim, fit, players, game):
    """Illustrative same-game combos using each side's top scorers; returns list of dicts."""
    out = []
    def top_by(side, n):
        pids = sim['teams'][side]['pids']
        return sorted(pids, key=lambda p: -sim['pg'].get(p, 0))[:n]
    h1, h2 = top_by('home', 2)
    a1, a2 = top_by('away', 2)
    gi = lambda p: sim['idx'][p]
    G = sim['goals']; A = sim['assists']
    hw = sim['home_win']; aw = ~hw
    line = fit['main_total']
    over = sim['total'] > line
    def pts(p): return G[:, gi(p)] + A[:, gi(p)]
    defs = [
        (f"{fit['home']} win + {name(players, h1)} anytime goal", [hw, G[:, gi(h1)] >= 1]),
        (f"{fit['away']} win + {name(players, a1)} anytime goal", [aw, G[:, gi(a1)] >= 1]),
        (f"{name(players, h1)} + {name(players, h2)} both score (same team)", [G[:, gi(h1)] >= 1, G[:, gi(h2)] >= 1]),
        (f"{name(players, h1)} + {name(players, a1)} both score (opposing teams)", [G[:, gi(h1)] >= 1, G[:, gi(a1)] >= 1]),
        (f"{name(players, h1)} anytime goal + over {line} total goals", [G[:, gi(h1)] >= 1, over]),
        (f"{name(players, h1)} 1+ point + {fit['home']} win", [pts(h1) >= 1, hw]),
        (f"{name(players, h1)} 1+ point + under {line} total goals", [pts(h1) >= 1, ~over]),
    ]
    for label, legs in defs:
        joint = np.logical_and.reduce(legs).mean()
        marg = [l.mean() for l in legs]
        indep = float(np.prod(marg))
        out.append({'combo': label, 'legs': marg, 'joint': float(joint), 'indep': indep,
                    'ratio': float(joint / indep) if indep > 0 else None})
    return out


def calibrate_assists(game, fit, players, iters=5, n=80000):
    """Iterative proportional fitting so simulated P(1+ assist) matches the devigged market."""
    scale = {}
    for _ in range(iters):
        sim = simulate(game, fit, players, n=n, aw_scale=scale)
        for pid in sim['players']:
            tgt = sim['pa'].get(pid)
            cur = (sim['assists'][:, sim['idx'][pid]] >= 1).mean()
            if tgt is not None and cur > 0:
                scale[pid] = scale.get(pid, 1.0) * (tgt / cur) ** 0.9
    return scale


def main(odds_path='odds_raw.json', players_path='players_raw.json'):
    games, players = load(odds_path, players_path)
    result = {}
    for g, game in games.items():
        fit = fit_game(game)
        scale = calibrate_assists(game, fit, players)
        sim = simulate(game, fit, players, aw_scale=scale)
        cal = calibration(sim, players)
        cmb = combos(sim, fit, players, game)
        ml_sim = float(sim['home_win'].mean())
        result[g] = {
            'matchup': f"{fit['away']} @ {fit['home']}",
            'fit': {k: (round(v, 3) if isinstance(v, float) else v) for k, v in fit.items() if k not in ('hid', 'aid')},
            'ml_home_sim': round(ml_sim, 3),
            'avg_total_sim': round(float(sim['total'].mean()), 2),
            'diag': sim['diag'],
            'calibration': [(n_, None if mg is None else round(mg, 3), round(sg, 3), None if ma is None else round(ma, 3), round(sa, 3), round(sp, 3)) for n_, mg, sg, ma, sa, sp in cal],
            'combos': cmb,
        }
    return result


if __name__ == '__main__':
    res = main(*(sys.argv[1:3]))
    print(json.dumps(res, indent=1, default=float))

"""
nhl_pipeline.py - run the whole NHL simulation from one 'nhl-sim-input/v1' payload (the JSON the n8n workflow assembles).

payload = {fixture, odds{book: rows}, roster{home, away}, lines{home, away}, sims}
Returns the replay data for the most likely simulated game, plus warnings about anything that had to fall back.
"""
import sys, json, math, collections, os, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import nhl_sim as ns, nhl_props as npp, nhl_v3 as v3
import select_extract as se


def _starter_first(goalies, starter):
    """Daily Faceoff's starting-goalies page is fresher than its lines page, so a named starter goes first."""
    name = (starter or {}).get('goalie')
    return [name] + [g for g in goalies if g != name] if name else list(goalies)


def payload_to_inputs(p):
    fx = p['fixture']
    game = {'fx': {'id': fx['fixture_id'], 'start_date': fx['start'], 'home_team_display': fx['home']['name'], 'away_team_display': fx['away']['name'],
                   'home_competitors': [{'id': fx['home']['id'], 'name': fx['home']['name']}], 'away_competitors': [{'id': fx['away']['id'], 'name': fx['away']['name']}]},
            'odds': {b: rows for b, rows in p['odds'].items() if rows}}
    players, roster = {}, collections.defaultdict(list)
    for side in ('home', 'away'):
        tid = fx[side]['id']
        for r in p['roster'][side]:
            players[r['id']] = {'name': r['name'], 'team': tid, 'pos': r.get('position')}
            roster[tid].append({'id': r['id'], 'n': r['name'], 'no': r.get('number'), 'pos': r.get('position')})
        L = p['lines'][side]
        if L:
            v3.DFO[fx[side]['name']] = {'updated': L.get('updatedAt', ''), 'lines': L['lines'], 'pairs': L['pairs'], 'pp': L['pp'], 'pk': L['pk'],
                                        'goalies': _starter_first(L.get('goalies', []), (p.get('starters') or {}).get(side)), 'out': [o['name'] for o in L.get('out', [])], 'numbers': L.get('numbers', {})}
    return game, players, roster


def load_stats(path='/home/claude/stats_pull.json'):
    import nhl_stats
    try:
        return nhl_stats.Stats(json.load(open(path)))
    except Exception:
        return None


def run_payload(p, n=200_000, cache_dir='/tmp/nhl_cache', verbose=True, stats=None, mix=None, return_bundle=False, price_keyed_cache=False):
    import nhl_stats
    stats_source = 'none'
    if stats is not None:
        v3.STATS = stats; stats_source = 'passed in'
    elif p.get('stats') and p['stats'].get('iceTime', {}).get('rows'):
        v3.STATS = nhl_stats.Stats(dict(p['stats'])); stats_source = 'payload'          # the statistics the n8n workflow pulled with this matchup
    else:
        v3.STATS = load_stats(); stats_source = 'file' if v3.STATS is not None else 'none'
    if mix is not None:
        v3.MIX = mix
    t0 = time.time(); warnings = []
    game, players, roster = payload_to_inputs(p)
    os.makedirs(cache_dir, exist_ok=True)
    fx = p['fixture']
    for side in ('home', 'away'):
        if not p['lines'][side]:
            warnings.append(f"No Daily Faceoff lineup for {fx[side]['name']}: lines inferred from market production")
    cnt = collections.Counter(r['market_id'] for rows in game['odds'].values() for r in rows)
    if cnt['anytime_goal_scorer'] < 20: warnings.append('Few or no anytime-goalscorer prices posted: player scoring uses line-deployment priors')
    if cnt['player_points'] < 20: warnings.append('Few or no points props posted: assists and points use line-deployment priors')
    if v3.STATS is None: warnings.append('No statistics were supplied: ice-time weights, player rates, and shot matchups use defaults')
    if cnt['player_saves'] == 0: warnings.append("No goalie saves lines posted: using Daily Faceoff's listed goalie and a typical saves mean")
    ck = ''
    if price_keyed_cache:      # a new pull has new prices, so it must not reuse an older calibration of the same game
        import hashlib; ck = '_' + hashlib.md5(json.dumps(p['odds'], sort_keys=True).encode()).hexdigest()[:10]
    fit, prof, P, sim, goalies = v3.run_game_v3(game, players, roster, n=n, cache_path=os.path.join(cache_dir, f"P_{fx['fixture_id']}{ck}.pkl"), verbose=verbose)
    for side in ('home', 'away'):
        for u in sim['lu'][side].get('unmatched', []):
            warnings.append(f"{fx[side]['name']}: {u}")
    for side in ('home', 'away'):
        a_ = (sim.get('shot_saves_anchor') or {}).get(side)
        if a_ and a_.get('source') == 'MoneyPuck':
            warnings.append(f"{fx[side]['name']}: no goalie saves line is posted, so their shots come from MoneyPuck's team shots for and against last season (about {a_['target'] + float(sim['hg' if side == 'home' else 'ag'].mean()):.0f} shots)")
    mk = npp.collect_market(game, fit, players); grid = npp.build_grid(sim, mk, fit, players)
    pc_ = sim.get('pen_cnt_pp') or sim['pen_cnt']       # the replay shows the penalties that create power plays
    extra = {'Penalties and special teams': [pc_['home'] >= 2, pc_['home'] >= 3, pc_['home'] >= 4, pc_['away'] >= 2, pc_['away'] >= 3,
             pc_['away'] >= 4, sim['ppg']['home'] >= 1, sim['ppg']['home'] >= 2, sim['ppg']['away'] >= 1, sim['ppg']['away'] >= 2, sim['shg']['home'] >= 1, sim['shg']['away'] >= 1]}
    j, score, agree, top5 = se.pick_most_likely(sim, grid, extra)
    ex = se.extract(sim, j, fit, players, roster, goalies, game)
    n_ = len(sim['hg']); hg, ag = int(sim['hg'][j]), int(sim['ag'][j]); so = hg == ag
    dh = hg + (1 if so and sim['so_home'][j] else 0); da = ag + (1 if so and not sim['so_home'][j] else 0)
    code = (np.where(sim['hg'] == sim['ag'], sim['hg'] + sim['so_home'], sim['hg']) * 100 + np.where(sim['hg'] == sim['ag'], sim['ag'] + (~sim['so_home']), sim['ag']))
    vals, cnts = np.unique(code, return_counts=True); order = np.argsort(-cnts)
    scorelines = [{'s': f"{vals[k] % 100}\u2013{vals[k] // 100}", 'p': float(cnts[k] / n_), 'winner': 'home' if vals[k] // 100 > vals[k] % 100 else 'away'} for k in order[:6]]
    home, away = fit['home'], fit['away']; ab = {'home': se.TEAMS.get(home, {}).get('abbr', fx['home'].get('abbr') or home[:3].upper()), 'away': se.TEAMS.get(away, {}).get('abbr', fx['away'].get('abbr') or away[:3].upper())}
    col = lambda nm: se.TEAMS.get(nm, dict(abbr=nm[:3].upper(), short=nm.split()[-1], fill='#33414F', ring='#C6D3DD', glow='#9DB0C1'))
    out = {'id': f"{ab['away'].lower()}-{ab['home'].lower()}", 'fixtureId': fx['fixture_id'], 'away': dict(name=away, **col(away)), 'home': dict(name=home, **col(home)),
           'simIndex': j + 1, 'simCount': n_, 'start': fx['start'], 'final': {'away': da, 'home': dh}, 'roster': ex['roster'], 'units': ex['units'], 'events': ex['events'],
           'goalies': {k: {'n': v['n'], 'no': v['no']} for k, v in ex['goalies'].items()}, 'scorelines': scorelines, 'agree': {k: round(v, 3) for k, v in agree.items()},
           'warnings': warnings, 'linesSource': {s: (p['lines'][s] or {}).get('updatedAt') for s in ('home', 'away')},
           'stats': {'winHome': float(sim['home_win'].mean()), 'expAway': float(fit['lam_a']), 'expHome': float(fit['lam_h']), 'avgTotal': float((sim['hg'] + sim['ag']).mean()), 'pastReg': float(sim['tie'].mean()),
                     'penH': float(sim['pen_cnt']['home'].mean()), 'penA': float(sim['pen_cnt']['away'].mean()), 'qTie': float(P['q']), 'rEN': float(P['r']), 'sims': n_}}
    out['statsSource'] = stats_source
    out['props'] = props_rows(sim, fit, players, ex, ab)
    out['seconds'] = round(time.time() - t0)
    if return_bundle:
        return out, {'game': game, 'players': players, 'roster': roster, 'fit': fit, 'sim': sim, 'goalies': goalies, 'grid': grid}
    return out


def props_rows(sim, fit, players, ex, ab):
    lam_h, lam_a = fit['lam_h'], fit['lam_a']
    fav = 'home' if sim['home_win'].mean() >= 0.5 else 'away'
    hp15 = -1.5 if fav == 'home' else 1.5; hp05 = -0.5 if fav == 'home' else 0.5
    def row(group, kind, label, key, **kw):
        pr = npp.sim_prob(sim, key)
        d = {'group': group, 'kind': kind, 'label': label, 'p': None if pr is None else round(float(pr), 4), 'fair': npp.american(pr) if pr else None}; d.update(kw); return d
    rows = [row('Moneyline', 'ml', f"{ab[fav]} to win", ('moneyline', None, None, fav), team=fav),
            row('All Spreads', 'spread', f"{ab[fav]} \u22121.5", ('puck_line', None, hp15, fav), team=fav, pts=-1.5)]
    mt = float(fit.get('main_total', 6.5)); rows.append(row('All Totals', 'total', f"Over {mt:g}", ('total_goals', None, mt, 'over'), line=mt, over=True))
    for s_ in ('home', 'away'):
        L = math.floor(lam_h if s_ == 'home' else lam_a) + 0.5
        rows.append(row('All Team Totals', 'tt', f"{ab[s_]} over {L:g}", ('team_total', s_, L, 'over'), team=s_, line=L))
    top = lambda s_, key, k: sorted(sim['teams'][s_]['pids'], key=lambda p: -key.get(p, 0))[:k]
    pgm = {p: float((sim['goals'][:, sim['idx'][p]] >= 1).mean()) for p in sim['pids']}
    pam = {p: float((sim['assists'][:, sim['idx'][p]] >= 1).mean()) for p in sim['pids']}
    nm = lambda p: players[p]['name']
    for s_ in ('home', 'away'):
        for p in top(s_, pgm, 2): rows.append(row('All Goalscorer Props', 'atg', f"{nm(p)} anytime goal", ('anytime_goal_scorer', p, None, 'yes'), player=nm(p)))
    for s_ in ('home', 'away'):
        p = top(s_, pgm, 1)[0]; rows.append(row('All Points Props', 'pts', f"{nm(p)} over 1.5 points", ('player_points', p, 1.5, 'over'), player=nm(p), line=1.5))
    for s_ in ('home', 'away'):
        p = top(s_, pam, 1)[0]; rows.append(row('All Assist Props', 'ast', f"{nm(p)} over 0.5 assists", ('player_assists', p, 0.5, 'over'), player=nm(p), line=0.5))
    for s_ in ('home', 'away'):
        p = top(s_, pgm, 1)[0]; rows.append(row('All 1st Goalscorer', 'fgs', f"{nm(p)} first goal", ('first_goal_scorer', p, None, 'yes'), player=nm(p)))
    for dside, gi in ex['goalies'].items():
        L = math.floor(gi['M']) + 0.5
        rows.append(row('All Saves', 'saves', f"{gi['n']} over {L:g} saves", ('player_saves', gi['pid'], L, 'over'), goalie=dside, line=L))
    for s_ in ('home', 'away'):
        p = top(s_, pgm, 1)[0]; L = math.floor(float(sim['shots'][:, sim['idx'][p]].mean())) + 0.5
        rows.append(row('Shots On Goal Props', 'sog', f"{nm(p)} over {L:g} shots", ('player_shots_on_goal', p, L, 'over'), player=nm(p), line=L))
    rows.append(row('60 Min Line', 'reg3', "Tied after 60 minutes", ('moneyline_3-way', None, None, 'draw')))
    rows.append(row('Goal In First Ten', 'ten', "Goal in first 10 minutes", ('1st_10_minutes_total_goals', None, 0.5, 'over')))
    rows.append(row('1st Period Puck Line', 'p1puck', f"{ab[fav]} \u22120.5 in the 1st", ('1st_period_puck_line', None, hp05, fav), team=fav, pts=-0.5))
    rows.append(row('1st Period Total', 'p1total', "Over 1.5 in the 1st", ('1st_period_total_goals', None, 1.5, 'over'), line=1.5))
    return rows


if __name__ == '__main__':
    payload = json.load(open(sys.argv[1]))
    items = payload if isinstance(payload, list) else [payload]
    res = run_payload(items[int(sys.argv[2]) if len(sys.argv) > 2 else 0], n=int(sys.argv[3]) if len(sys.argv) > 3 else 200000)
    json.dump(res, open(sys.argv[4] if len(sys.argv) > 4 else 'pipeline_out.json', 'w'), ensure_ascii=False)
    print('done', res['id'], res['final'], 'events', len(res['events']), 'warnings', res['warnings'], res['seconds'], 's')

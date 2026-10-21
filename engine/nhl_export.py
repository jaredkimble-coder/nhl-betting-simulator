"""Builds the data the betting-tool page needs for one simulated matchup (bets, prices, and a 20,000-game sample), from the pipeline's own simulation."""
import sys, json, collections, math, os, re
import numpy as np
import nhl_sim as ns, nhl_props as npp, nhl_v3 as v3, select_extract as se, nhl_pipeline as pl
from nhl_sim import imp

BOOKNAME = {'kalshi': 'Kalshi', 'polymarket': 'Polymarket', 'polymarket_usa_': 'Polymarket', 'draftkings': 'DraftKings', 'fanduel': 'FanDuel', 'betmgm': 'BetMGM', 'caesars': 'Caesars'}

def u8(a, lo=0): return np.clip(np.asarray(a).astype(np.int64) - lo, 0, 255).astype(np.uint8)

def book_asks(game, hid):
    """key -> {book: ask probability (the price you would pay)}."""
    out = collections.defaultdict(dict)
    for b, rows in game['odds'].items():
        lab = BOOKNAME.get(b, b)
        for r in rows:
            mid, tid, pid, pts, sl, price = r['market_id'], r['team_id'], r['player_id'], r['points'], r['selection_line'], r['price']
            k = None
            if mid == 'moneyline' and tid: k = (mid, None, None, 'home' if tid == hid else 'away')
            elif mid in ('puck_line', '1st_period_puck_line') and tid and pts is not None: k = (mid, None, pts if tid == hid else -pts, 'home' if tid == hid else 'away')
            elif mid in ('total_goals', '1st_period_total_goals', '1st_10_minutes_total_goals') and pts is not None and sl in ('over', 'under'): k = (mid, None, pts, sl)
            elif mid == 'team_total' and tid and pts is not None and sl in ('over', 'under'): k = (mid, 'home' if tid == hid else 'away', pts, sl)
            elif mid in ('player_goals', 'player_assists', 'player_points', 'player_saves', 'player_shots_on_goal', 'player_power_play_points') and pid and pts is not None and sl in ('over', 'under'): k = (mid, pid, pts, sl)
            elif mid in ('team_total_power_play_goals', 'team_total_short_handed_goals') and tid and pts is not None and sl in ('over', 'under'): k = (mid, 'home' if tid == hid else 'away', pts, sl)
            elif mid in ('team_total_shots_on_goal',) and tid and pts is not None and sl in ('over', 'under'): k = ('tsog', 'home' if tid == hid else 'away', pts, sl)
            elif mid == 'total_shots_on_goal' and pts is not None and sl in ('over', 'under'): k = ('sogtot', None, pts, sl)
            elif mid == 'will_there_be_overtime' and (r.get('name') or '').lower() in ('yes', 'no'): k = ('otg', None, None, r['name'].lower())
            elif mid == 'first_team_to_score_3-way': k = ('firstteam', None, None, 'home' if tid == hid else ('away' if tid else 'none'))
            elif mid == 'both_teams_to_score' and (r.get('name') or '').lower() in ('yes', 'no'): k = ('btts', None, None, r['name'].lower())
            elif mid == 'both_teams_to_score_reg_time' and (r.get('name') or '').lower() in ('yes', 'no'): k = ('pbtts', 0, None, r['name'].lower())
            # Caesars' full-game odd/even prices (odd -260 / even +175 in every game) are not credible for hockey, so they are not attached
            elif mid == 'team_to_win_to_nil' and tid: k = ('wtn', None, None, 'home' if tid == hid else 'away')
            elif mid == 'last_goal_scorer' and pid: k = ('lgs', pid, None, 'yes')
            elif mid in ('1st_period_player_points', '1st_period_player_assists') and pid and pts == 0.5 and sl == 'over': k = ('p1pts' if 'points' in mid else 'p1ast', pid, 0.5, 'over')
            elif re.match(r'^(1st|2nd|3rd)_period_', mid):
                per = {'1st': 1, '2nd': 2, '3rd': 3}[mid[:3]]; base = mid[11:]
                if base == 'moneyline_3-way': k = ('pres', per, None, 'home' if tid == hid else ('away' if tid else 'draw'))
                elif base == 'total_goals' and per >= 2 and pts is not None and sl in ('over', 'under'): k = ('ptot', per, pts, sl)
                elif base == 'team_total' and tid and pts is not None and sl in ('over', 'under'): k = ('ptt', (per, 'home' if tid == hid else 'away'), pts, sl)
                elif base == 'puck_line' and per >= 2 and tid and pts is not None: k = ('pspread', per, pts if tid == hid else -pts, 'home' if tid == hid else 'away')
                elif base == 'both_teams_to_score' and (r.get('name') or '').lower() in ('yes', 'no'): k = ('pbtts', per, None, r['name'].lower())
                elif base == 'total_goals_odd_even' and (r.get('name') or '').lower() in ('odd', 'even'): k = ('oddeven', per, None, r['name'].lower())
                elif base in ('player_goals', 'anytime_goal_scorer') and pid: k = ('pg', pid, per, 'yes')
            elif mid in ('anytime_goal_scorer', 'first_goal_scorer') and pid: k = (mid, pid, None, 'yes')
            if k is not None:
                v = imp(price)
                if lab not in out[k] or v < out[k][lab][0]: out[k][lab] = (v, int(price))
    return out


def export_game(p, bundle, seed_idx=0, n_export=20000):
    """p: the n8n payload for the matchup; bundle: what nhl_pipeline.run_payload(..., return_bundle=True) returned. Returns {'meta': ..., 'cols': {...}}."""
    game, players, roster, fit, sim, goalies, grid = (bundle[k] for k in ('game', 'players', 'roster', 'fit', 'sim', 'goalies', 'grid'))
    fx = p['fixture']
    asks = book_asks(game, fit['hid'])
    n = len(sim['hg']); rng = np.random.default_rng(500 + seed_idx); sel = np.sort(rng.choice(n, n_export, replace=False))
    pids, idx_ = sim['pids'], sim['idx']
    cols = collections.OrderedDict()
    cols['hg'] = u8(sim['hg'][sel]); cols['ag'] = u8(sim['ag'][sel]); cols['reg_h'] = u8(sim['reg_h'][sel]); cols['reg_a'] = u8(sim['reg_a'][sel]); cols['margin'] = u8(sim['margin'][sel], -20)
    cols['n10'] = u8(sim['n10'][sel]); cols['g1h'] = u8(sim['g1h'][sel]); cols['g1a'] = u8(sim['g1a'][sel]); cols['first'] = u8(sim['first_col'][sel] + 2)
    cols['ppg_h'] = u8(sim['ppg']['home'][sel]); cols['ppg_a'] = u8(sim['ppg']['away'][sel]); cols['pen_h'] = u8(sim['pen_cnt']['home'][sel]); cols['pen_a'] = u8(sim['pen_cnt']['away'][sel])
    d58 = sim['d58'][sel]; cols['pull'] = u8((np.abs(d58) == 1) | ((np.abs(d58) == 2) & (sim['late_h'][sel] | sim['late_a'][sel]))); cols['ot'] = u8(sim['tie'][sel])
    shots = sim['shots'][sel]; goals = sim['goals'][sel]; assists = sim['assists'][sel]
    cols['sog_h'] = u8(sum(shots[:, idx_[q]].astype(np.int64) for q in sim['teams']['home']['pids'])); cols['sog_a'] = u8(sum(shots[:, idx_[q]].astype(np.int64) for q in sim['teams']['away']['pids']))
    gl = []
    for gpid, gi in goalies.items():
        side = 'home' if gi['team'] == fit['hid'] else 'away'
        cols[f'sv{len(gl)}'] = u8(sim['saves'][gpid][sel]); gl.append({'pid': gpid, 'n': players[gpid]['name'], 'side': side, 'col': f'sv{len(gl)}', 'M': round(gi['M'], 1), 'default': bool(gi.get('default'))})
    plist = []
    for i, pid in enumerate(pids):
        cols[f'g{i}'] = u8(goals[:, i]); cols[f'a{i}'] = u8(assists[:, i]); cols[f's{i}'] = u8(shots[:, i])
        side = 'home' if players[pid]['team'] == fit['hid'] else 'away'; lu = sim['lu'][side]
        ln = next((k + 1 for k, l in enumerate(lu['lines']) if pid in l), None); pr = next((k + 1 for k, l in enumerate(lu['pairs']) if pid in l), None); pp = next((k + 1 for k, l in enumerate(lu['pp']) if pid in l), None)
        no = next((r['no'] for r in roster[fit['hid'] if side == 'home' else fit['aid']] if r['id'] == pid), None)
        plist.append({'n': players[pid]['name'], 'no': no, 'pos': 'D' if pid in set(lu['D']) else 'F', 'side': side, 'line': ln, 'pair': pr, 'pp': pp, 'dressed': ln is not None or pr is not None})
    gs, gt, gteam, gcat = sim['goal_sim'], sim['goal_t'], sim['goal_team'], sim['goal_cat']; gsc, ga1, ga2 = sim['goal_scorer'], sim['goal_a1'], sim['goal_a2']; Pn = len(pids)
    isPP = (gcat == v3.CAT['PP5']) | (gcat == v3.CAT['PP4'])
    def per_goals(lo, hi, home): m = (gt >= lo) & (gt < hi) & ((gteam == 1) if home else (gteam == 0)); return np.bincount(gs[m], minlength=n)
    cols['p2h'] = u8(per_goals(20, 40, True)[sel]); cols['p2a'] = u8(per_goals(20, 40, False)[sel]); cols['p3h'] = u8(per_goals(40, 60, True)[sel]); cols['p3a'] = u8(per_goals(40, 60, False)[sel])
    cols['shg_h'] = u8(sim['shg']['home'][sel]); cols['shg_a'] = u8(sim['shg']['away'][sel])
    ppp = np.bincount(gs[isPP] * Pn + gsc[isPP], minlength=n * Pn).astype(np.int64)
    for a_ in (ga1, ga2):
        ok = isPP & (a_ >= 0); ppp += np.bincount(gs[ok] * Pn + a_[ok], minlength=n * Pn)
    ppp = ppp.reshape(n, Pn); m1 = gt < 20; f1 = np.bincount(gs[m1] * Pn + gsc[m1], minlength=n * Pn).reshape(n, Pn)
    def per_player(lo, hi): m = (gt >= lo) & (gt < hi); return np.bincount(gs[m] * Pn + gsc[m], minlength=n * Pn).reshape(n, Pn)
    f2, f3 = per_player(20, 40), per_player(40, 60); m1 = gt < 20
    fa1 = np.zeros(n * Pn, dtype=np.int64)
    for a_ in (ga1, ga2):
        ok = m1 & (a_ >= 0); fa1 += np.bincount(gs[ok] * Pn + a_[ok], minlength=n * Pn)
    fa1 = fa1.reshape(n, Pn)
    for i in range(Pn):
        cols[f'pp{i}'] = u8(ppp[sel, i]); cols[f'f{i}'] = u8(f1[sel, i]); cols[f'q2_{i}'] = u8(f2[sel, i]); cols[f'q3_{i}'] = u8(f3[sel, i]); cols[f'fa{i}'] = u8(fa1[sel, i]); cols[f'fp{i}'] = u8((f1 + fa1)[sel, i])
    order_ = np.lexsort((gt, gs)); sim_s = gs[order_]; sc_s = gsc[order_]; last_idx = np.searchsorted(sim_s, np.arange(n), side='right') - 1
    has_ = last_idx >= 0; has_[has_] = sim_s[last_idx[has_]] == np.arange(n)[has_]; last_col = np.full(n, -2, dtype=np.int32); last_col[has_] = sc_s[last_idx[has_]]
    cols['last'] = u8(last_col[sel] + 2)
    ABV = {'home': se.TEAMS[fit['home']]['abbr'], 'away': se.TEAMS[fit['away']]['abbr']}
    gidx = {g_['pid']: k for k, g_ in enumerate(gl)}
    legs = []
    for r in grid:
        mid, subj, line, side = r['_key']; key = r['_key']; leg = None
        if mid == 'moneyline': leg = dict(kind='ml', side=side, label=f"{ABV[side]} to win")
        elif mid == 'puck_line':
            if abs(line) not in (1.5, 2.5): continue
            shown = line if side == 'home' else -line; leg = dict(kind='spread', hp=line, side=side, label=f"{ABV[side]} {'+' if shown > 0 else '\u2212'}{abs(shown):g}")
        elif mid == 'total_goals':
            if line not in (4.5, 5.5, 6.5, 7.5, 8.5): continue
            leg = dict(kind='total', line=line, over=(side == 'over'), label=f"{side.title()} {line:g} goals")
        elif mid == 'team_total':
            if line not in (1.5, 2.5, 3.5, 4.5): continue
            leg = dict(kind='tt', team=subj, line=line, over=(side == 'over'), label=f"{ABV[subj]} {side} {line:g} goals")
        elif mid == 'anytime_goal_scorer':
            if subj not in idx_: continue
            leg = dict(kind='atg', p=idx_[subj], label=f"{players[subj]['name']} anytime goal")
        elif mid == 'player_goals':
            if line != 1.5 or side != 'over' or subj not in idx_: continue
            leg = dict(kind='pgoal', p=idx_[subj], line=line, label=f"{players[subj]['name']} over 1.5 goals")
        elif mid == 'player_points':
            if side != 'over' or line not in (0.5, 1.5, 2.5) or subj not in idx_: continue
            leg = dict(kind='pts', p=idx_[subj], line=line, label=f"{players[subj]['name']} over {line:g} points")
        elif mid == 'player_assists':
            if side != 'over' or line not in (0.5, 1.5) or subj not in idx_: continue
            leg = dict(kind='ast', p=idx_[subj], line=line, label=f"{players[subj]['name']} over {line:g} assists")
        elif mid == 'first_goal_scorer':
            if subj not in idx_: continue
            leg = dict(kind='fgs', p=idx_[subj], label=f"{players[subj]['name']} first goal")
        elif mid == 'player_saves':
            if subj not in gidx: continue
            leg = dict(kind='saves', gi=gidx[subj], line=line, over=(side == 'over'), label=f"{players[subj]['name']} {side} {line:g} saves")
        elif mid == 'player_shots_on_goal':
            if side != 'over' or line not in (1.5, 2.5, 3.5, 4.5) or subj not in idx_: continue
            leg = dict(kind='sog', p=idx_[subj], line=line, label=f"{players[subj]['name']} over {line:g} shots")
        elif mid == 'moneyline_3-way': leg = dict(kind='reg3', side=side, label={'home': f"{ABV['home']} lead after 60", 'away': f"{ABV['away']} lead after 60", 'draw': 'Tied after 60 minutes'}[side])
        elif mid == '1st_10_minutes_total_goals':
            if side != 'over': continue
            leg = dict(kind='ten', line=line, over=True, label="Goal in first 10 minutes" if line == 0.5 else f"Over {line:g} goals in first 10")
        elif mid == '1st_period_puck_line':
            shown = line if side == 'home' else -line
            if abs(line) not in (0.5, 1.5): continue
            leg = dict(kind='p1puck', hp=line, side=side, label=f"{ABV[side]} {'+' if shown > 0 else '\u2212'}{abs(shown):g} in the 1st")
        elif mid == '1st_period_total_goals': leg = dict(kind='p1tot', line=line, over=(side == 'over'), label=f"{side.title()} {line:g} goals in the 1st")
        if leg is None: continue
        a = asks.get(key, {})
        SB = {n_: v_ for n_, v_ in a.items() if n_ in ('DraftKings', 'FanDuel', 'BetMGM', 'Caesars')}; EX = {n_: v_ for n_, v_ in a.items() if n_ in ('Kalshi', 'Polymarket')}
        sb = min(SB.items(), key=lambda kv: kv[1][0]) if SB else None; ex = min(EX.items(), key=lambda kv: kv[1][0]) if EX else None
        leg.update(group=r['Group'], id=len(legs), sb=({'p': round(sb[1][0], 4), 'odds': sb[1][1], 'book': sb[0]} if sb else None), ex=({'p': round(ex[1][0], 4), 'book': ex[0]} if ex else None), nBooks=len(a), check=round(r['Sim probability'], 4))
        legs.append(leg)
    C = {k: v.astype(np.int64) for k, v in cols.items()}
    plhome = np.array([q['side'] == 'home' for q in plist]); fs = np.zeros(n_export, dtype=np.int64); mf = C['first'] >= 2; fs[mf] = np.where(plhome[C['first'][mf] - 2], 1, 2)
    def team_col(base, team): return C[f'{base}_h' if team == 'home' else f'{base}_a']
    def pbits(l):
        k = l['kind']
        if k == 'ppp': return C[f"pp{l['p']}"] > l['line']
        if k == 'pg': return C[{1: f"f{l['p']}", 2: f"q2_{l['p']}", 3: f"q3_{l['p']}"}[l['per']]] >= 1
        if k == 'p1pts': return C[f"fp{l['p']}"] >= 1
        if k == 'p1ast': return C[f"fa{l['p']}"] >= 1
        if k == 'lgs': return C['last'] == l['p'] + 2
        if k == 'ptt': h, a = {1: (C['g1h'], C['g1a']), 2: (C['p2h'], C['p2a']), 3: (C['p3h'], C['p3a'])}[l['per']]; v = h if l['team'] == 'home' else a; return v > l['line'] if l['over'] else v < l['line']
        if k == 'pspread': h, a = {2: (C['p2h'], C['p2a']), 3: (C['p3h'], C['p3a'])}[l['per']]; return (h - a + l['hp'] > 0) if l['side'] == 'home' else (h - a + l['hp'] < 0)
        if k == 'pbtts':
            h, a = {0: (C['reg_h'], C['reg_a']), 1: (C['g1h'], C['g1a']), 2: (C['p2h'], C['p2a']), 3: (C['p3h'], C['p3a'])}[l['per']]; b = (h > 0) & (a > 0); return b if l['yes'] else ~b
        if k in ('tsog', 'tpp', 'tsh', 'tpen'): v = team_col({'tsog': 'sog', 'tpp': 'ppg', 'tsh': 'shg', 'tpen': 'pen'}[k], l['team']); return v > l['line'] if l['over'] else v < l['line']
        if k == 'sogtot': v = C['sog_h'] + C['sog_a']; return v > l['line'] if l['over'] else v < l['line']
        if k == 'pentot': v = C['pen_h'] + C['pen_a']; return v > l['line'] if l['over'] else v < l['line']
        if k == 'sogmore': return {'home': C['sog_h'] > C['sog_a'], 'away': C['sog_h'] < C['sog_a'], 'even': C['sog_h'] == C['sog_a']}[l['side']]
        if k == 'btts': b = (C['hg'] > 0) & (C['ag'] > 0); return b if l['yes'] else ~b
        if k == 'firstteam': return {'home': fs == 1, 'away': fs == 2, 'none': fs == 0}[l['side']]
        if k == 'pres':
            h, a = {1: (C['g1h'], C['g1a']), 2: (C['p2h'], C['p2a']), 3: (C['p3h'], C['p3a'])}[l['per']]; return {'home': h > a, 'away': h < a, 'draw': h == a}[l['side']]
        if k == 'ptot': v = C['p2h'] + C['p2a'] if l['per'] == 2 else C['p3h'] + C['p3a']; return v > l['line'] if l['over'] else v < l['line']
        if k == 'wmargin':
            m = C['margin'] - 20; m = m if l['side'] == 'home' else -m; return (m == l['k']) if l['k'] < 3 else (m >= 3)
        if k == 'oddeven':
            h, a = {0: (C['hg'], C['ag']), 1: (C['g1h'], C['g1a']), 2: (C['p2h'], C['p2a']), 3: (C['p3h'], C['p3a'])}[l.get('per') or 0]; o = ((h + a) % 2) == 1; return o if l['odd'] else ~o
        if k == 'anypp': b = (C['ppg_h'] + C['ppg_a']) > 0; return b if l['yes'] else ~b
        if k == 'otg': b = C['ot'] == 1; return b if l['yes'] else ~b
        if k == 'shutout': return ((C['ag'] == 0) & (C['hg'] > 0)) if l['side'] == 'home' else ((C['hg'] == 0) & (C['ag'] > 0))
        raise ValueError(k)
    seen_new = set(); nameof = lambda q: plist[q]['n']
    def add_leg(leg, group, key=None):
        ident = (leg['kind'], leg.get('p'), leg.get('team'), leg.get('side'), leg.get('line'), leg.get('over'), leg.get('yes'), leg.get('per') or 0, leg.get('k'), leg.get('odd'), leg.get('hp'))
        if ident in seen_new: return
        pr = float(pbits(leg).mean())
        if pr < 0.01 or pr > 0.995: return
        seen_new.add(ident); a = asks.get(key, {}) if key else {}
        SB = {n_: v_ for n_, v_ in a.items() if n_ in ('DraftKings', 'FanDuel', 'BetMGM', 'Caesars')}; EX = {n_: v_ for n_, v_ in a.items() if n_ in ('Kalshi', 'Polymarket')}
        sb = min(SB.items(), key=lambda kv: kv[1][0]) if SB else None; ex = min(EX.items(), key=lambda kv: kv[1][0]) if EX else None
        leg.update(group=group, id=len(legs), sb=({'p': round(sb[1][0], 4), 'odds': sb[1][1], 'book': sb[0]} if sb else None), ex=({'p': round(ex[1][0], 4), 'book': ex[0]} if ex else None), nBooks=len(a), check=round(pr, 4))
        legs.append(leg)
    # priced by the books: power-play points and team power-play / short-handed goals
    for key in sorted(asks.keys(), key=str):
        mid, subj, line, side = key
        if mid == 'player_power_play_points' and subj in idx_ and side == 'over' and line in (0.5, 1.5):
            add_leg(dict(kind='ppp', p=idx_[subj], line=line, label=f"{players[subj]['name']} over {line:g} power-play points"), 'All Power Play Props', key)
        elif mid == 'team_total_power_play_goals' and line in (0.5, 1.5, 2.5):
            add_leg(dict(kind='tpp', team=subj, line=line, over=(side == 'over'), label=f"{ABV[subj]} {side} {line:g} power-play goals"), 'All Power Play Props', key)
        elif mid == 'team_total_short_handed_goals' and line in (0.5, 1.5):
            add_leg(dict(kind='tsh', team=subj, line=line, over=(side == 'over'), label=f"{ABV[subj]} {side} {line:g} short-handed goals"), 'All Power Play Props', key)
    ORD = {1: '1st', 2: '2nd', 3: '3rd'}
    for key in sorted(asks.keys(), key=str):
        tag, subj, line, side = key
        if tag == 'tsog' and line is not None: add_leg(dict(kind='tsog', team=subj, line=line, over=(side == 'over'), label=f"{ABV[subj]} {side} {line:g} shots on goal"), 'Team Shots Props', key)
        elif tag == 'sogtot': add_leg(dict(kind='sogtot', line=line, over=(side == 'over'), label=f"{side.title()} {line:g} total shots on goal"), 'Team Shots Props', key)
        elif tag == 'otg': add_leg(dict(kind='otg', yes=(side == 'yes'), label='Goes to overtime' if side == 'yes' else 'Decided in 60 minutes'), 'Game Props', key)
        elif tag == 'firstteam': add_leg(dict(kind='firstteam', side=side, label={'home': f"{ABV['home']} scores first", 'away': f"{ABV['away']} scores first", 'none': 'No goals'}[side]), 'Game Props', key)
        elif tag == 'btts': add_leg(dict(kind='btts', yes=(side == 'yes'), label='Both teams score' if side == 'yes' else 'A team is shut out'), 'Game Props', key)
        elif tag == 'pbtts':
            ps_ = {0: 'in regulation', 1: 'in the 1st', 2: 'in the 2nd', 3: 'in the 3rd'}[subj]; add_leg(dict(kind='pbtts', per=subj, yes=(side == 'yes'), label=f"Both teams score {ps_}" if side == 'yes' else f"A team is blanked {ps_}"), 'Game Props' if subj == 0 else 'Period Props', key)
        elif tag == 'oddeven':
            ps_ = {0: '', 1: ' in the 1st', 2: ' in the 2nd', 3: ' in the 3rd'}[subj]; add_leg(dict(kind='oddeven', per=subj, odd=(side == 'odd'), label=f"{'Odd' if side == 'odd' else 'Even'} total goals{ps_}"), 'Game Props' if subj == 0 else 'Period Props', key)
        elif tag == 'wtn': add_leg(dict(kind='shutout', side=side, label=f"{ABV[side]} win to nil"), 'Game Props', key)
        elif tag == 'lgs' and subj in idx_: add_leg(dict(kind='lgs', p=idx_[subj], label=f"{nameof(idx_[subj])} last goal"), 'Game Props', key)
        elif tag == 'p1pts' and subj in idx_: add_leg(dict(kind='p1pts', p=idx_[subj], line=0.5, label=f"{nameof(idx_[subj])} over 0.5 points in the 1st"), 'Period Props', key)
        elif tag == 'p1ast' and subj in idx_: add_leg(dict(kind='p1ast', p=idx_[subj], line=0.5, label=f"{nameof(idx_[subj])} over 0.5 assists in the 1st"), 'Period Props', key)
        elif tag == 'pg' and subj in idx_: add_leg(dict(kind='pg', p=idx_[subj], per=line, label=f"{nameof(idx_[subj])} goal in the {ORD[line]} period"), 'Period Props', key)
        elif tag == 'pres': add_leg(dict(kind='pres', per=subj, side=side, label={'home': f"{ABV['home']} lead after {subj}" if subj == 1 else f"{ABV['home']} win the {ORD[subj]}", 'away': f"{ABV['away']} lead after {subj}" if subj == 1 else f"{ABV['away']} win the {ORD[subj]}", 'draw': 'Tied after 1' if subj == 1 else f"{ORD[subj]} period tied"}[side]), 'Period Props', key)
        elif tag == 'ptot' and line % 1 == 0.5: add_leg(dict(kind='ptot', per=subj, line=line, over=(side == 'over'), label=f"{side.title()} {line:g} goals in the {ORD[subj]}"), 'Period Props', key)
        elif tag == 'ptt' and line % 1 == 0.5: add_leg(dict(kind='ptt', per=subj[0], team=subj[1], line=line, over=(side == 'over'), label=f"{ABV[subj[1]]} {side} {line:g} goals in the {ORD[subj[0]]}"), 'Period Props', key)
        elif tag == 'pspread' and abs(line) in (0.5, 1.5):
            shown = line if side == 'home' else -line; add_leg(dict(kind='pspread', per=subj, hp=line, side=side, label=f"{ABV[side]} {'+' if shown > 0 else chr(0x2212)}{abs(shown):g} in the {ORD[subj]}"), 'Period Props', key)
    # from the stored games, no price yet: power plays
    for q in range(Pn):
        if plist[q]['pp']:
            add_leg(dict(kind='ppp', p=q, line=0.5, label=f"{nameof(q)} over 0.5 power-play points"), 'All Power Play Props')
    for team in ('home', 'away'):
        for ln in (0.5, 1.5):
            for ov in (True, False): add_leg(dict(kind='tpp', team=team, line=ln, over=ov, label=f"{ABV[team]} {'over' if ov else 'under'} {ln:g} power-play goals"), 'All Power Play Props')
        for ov in (True, False): add_leg(dict(kind='tsh', team=team, line=0.5, over=ov, label=f"{ABV[team]} {'over' if ov else 'under'} 0.5 short-handed goals"), 'All Power Play Props')
    for yes in (True, False): add_leg(dict(kind='anypp', yes=yes, label=f"{'A' if yes else 'No'} power-play goal in the game"), 'All Power Play Props')
    # team shots on goal
    for team in ('home', 'away'):
        mu = team_col('sog', team).mean(); base = math.floor(mu) + 0.5
        for j in range(-3, 4):
            for ov in (True, False): add_leg(dict(kind='tsog', team=team, line=base + 2 * j, over=ov, label=f"{ABV[team]} {'over' if ov else 'under'} {base + 2 * j:g} shots on goal"), 'Team Shots Props')
    mu = (C['sog_h'] + C['sog_a']).mean(); base = math.floor(mu) + 0.5
    for j in range(-3, 4):
        for ov in (True, False): add_leg(dict(kind='sogtot', line=base + 3 * j, over=ov, label=f"{'Over' if ov else 'Under'} {base + 3 * j:g} total shots on goal"), 'Team Shots Props')
    for sd, lab in (('home', f"{ABV['home']} outshoot {ABV['away']}"), ('away', f"{ABV['away']} outshoot {ABV['home']}"), ('even', 'Equal shots on goal')): add_leg(dict(kind='sogmore', side=sd, label=lab), 'Team Shots Props')
    # whole-game results
    for yes in (True, False): add_leg(dict(kind='btts', yes=yes, label='Both teams score' if yes else 'A team is shut out'), 'Game Props')
    for sd, lab in (('home', f"{ABV['home']} scores first"), ('away', f"{ABV['away']} scores first"), ('none', 'No goals')): add_leg(dict(kind='firstteam', side=sd, label=lab), 'Game Props')
    for sd in ('home', 'away'):
        for k_, lab in ((1, 'by 1'), (2, 'by 2'), (3, 'by 3 or more')): add_leg(dict(kind='wmargin', side=sd, k=k_, label=f"{ABV[sd]} win {lab} (OT and shootout wins count as 1)"), 'Game Props')
        add_leg(dict(kind='shutout', side=sd, label=f"{ABV[sd]} win to nil"), 'Game Props')
    for odd in (True, False): add_leg(dict(kind='oddeven', per=0, odd=odd, label=f"{'Odd' if odd else 'Even'} total goals"), 'Game Props')
    for yes in (True, False): add_leg(dict(kind='otg', yes=yes, label='Goes to overtime' if yes else 'Decided in 60 minutes'), 'Game Props')
    # periods
    for sd, lab in (('home', f"{ABV['home']} lead after 1"), ('away', f"{ABV['away']} lead after 1"), ('draw', 'Tied after 1')): add_leg(dict(kind='pres', per=1, side=sd, label=lab), 'Period Props')
    for per, ord_ in ((2, '2nd'), (3, '3rd')):
        for sd, lab in (('home', f"{ABV['home']} win the {ord_}"), ('away', f"{ABV['away']} win the {ord_}"), ('draw', f"{ord_} period tied")): add_leg(dict(kind='pres', per=per, side=sd, label=lab), 'Period Props')
        for ln in (0.5, 1.5, 2.5, 3.5):
            for ov in (True, False): add_leg(dict(kind='ptot', per=per, line=ln, over=ov, label=f"{'Over' if ov else 'Under'} {ln:g} goals in the {ord_}"), 'Period Props')
    for q in range(Pn):
        if plist[q]['dressed'] and float((C[f'f{q}'] >= 1).mean()) >= 0.05: add_leg(dict(kind='pg', p=q, per=1, label=f"{nameof(q)} goal in the 1st period"), 'Period Props')
    for per in (1, 2, 3):
        for team in ('home', 'away'):
            for ln in (0.5, 1.5):
                for ov in (True, False): add_leg(dict(kind='ptt', per=per, team=team, line=ln, over=ov, label=f"{ABV[team]} {'over' if ov else 'under'} {ln:g} goals in the {ORD[per]}"), 'Period Props')
        for yes in (True, False): add_leg(dict(kind='pbtts', per=per, yes=yes, label=(f"Both teams score in the {ORD[per]}" if yes else f"A team is blanked in the {ORD[per]}")), 'Period Props')
        if per >= 2:
            for sd in ('home', 'away'):
                for hp_ in (0.5, 1.5):
                    for sgn in (1, -1):
                        hp = hp_ * sgn if sd == 'home' else -hp_ * sgn; shown = hp if sd == 'home' else -hp
                        add_leg(dict(kind='pspread', per=per, hp=hp, side=sd, label=f"{ABV[sd]} {'+' if shown > 0 else chr(0x2212)}{abs(shown):g} in the {ORD[per]}"), 'Period Props')
    # penalties
    for ln in (1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5):
        for ov in (True, False): add_leg(dict(kind='pentot', line=ln, over=ov, label=f"{'Over' if ov else 'Under'} {ln:g} total penalties"), 'Penalty Props')
    for team in ('home', 'away'):
        for ln in (0.5, 1.5, 2.5, 3.5):
            for ov in (True, False): add_leg(dict(kind='tpen', team=team, line=ln, over=ov, label=f"{ABV[team]} {'over' if ov else 'under'} {ln:g} penalties"), 'Penalty Props')
    def regroup(l):
        k, per, g_ = l['kind'], l.get('per'), l['group']
        if g_ in ('1st Period Puck Line', '1st Period Total'): return '1st Period Props'
        if g_ == 'Goal In First Ten': return 'Game Props'
        if k in ('pg', 'p1pts', 'p1ast'): return 'Player Period Props'
        if g_ == 'Period Props': return {1: '1st Period Props', 2: '2nd Period Props', 3: '3rd Period Props'}[per]
        if k in ('fgs', 'lgs'): return 'First & Last Goalscorer'
        if k in ('tpp', 'tsh', 'anypp'): return 'Team Power Play Props'
        return g_
    for l_ in legs: l_['group'] = regroup(l_)
    nflag = collections.Counter(sim['shot_flags'].values()); print('   shots layer:', dict(nflag), flush=True)
    cnt = collections.Counter(rr['market_id'] for rows in game['odds'].values() for rr in rows)
    warnings = list(sim['lu']['home'].get('unmatched', [])) + list(sim['lu']['away'].get('unmatched', []))
    if cnt['player_points'] < 20: warnings.append('Few points and assists props are posted for this game, so those player prices come from line deployment.')
    if cnt['player_shots_on_goal'] < 20: warnings.append('Few shots-on-goal props are posted, so shot prices are estimates.')
    if cnt['player_saves'] == 0 or all(g_['default'] for g_ in gl): warnings.append('No goalie saves lines are posted yet, so saves use a typical mean for the starting goalie.')
    meta = {'id': f"{ABV['away'].lower()}-{ABV['home'].lower()}", 'fixtureId': fx['fixture_id'], 'away': dict(name=fit['away'], **se.TEAMS[fit['away']]), 'home': dict(name=fit['home'], **se.TEAMS[fit['home']]), 'start': fx['start'],
            'N': n_export, 'cols': list(cols.keys()), 'players': plist, 'goalies': gl, 'legs': legs, 'fav': 'home' if sim['home_win'].mean() >= 0.5 else 'away', 'asof': p['generatedAt'], 'asofExtra': p.get('extraPulledAt'), 'starters': p.get('starters'),
            'linesUpdated': {s_: (p['lines'][s_] or {}).get('updatedAt') for s_ in ('home', 'away')}, 'warnings': warnings,
            'stats': {'winHome': float(sim['home_win'].mean()), 'total': float((sim['hg'] + sim['ag']).mean())}}
    try:
        import nhl_matchup
        meta['matchup'] = nhl_matchup.build_matchup(p, meta, cols, sim)        # MoneyPuck and ShotPropz matchup analysis for the page
    except Exception as ex_:
        meta['matchup'] = None
    return {'meta': meta, 'cols': cols}

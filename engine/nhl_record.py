"""Turns graded games into the 'How the sim did' summary shown on the page."""
import numpy as np


def _units(odds, win):
    if win:
        return odds / 100.0 if odds > 0 else 100.0 / abs(odds)
    return -1.0


def build_record(graded):
    """graded: list of graded-game dicts ({'date', 'game', 'bets': [{'group', 'p', 'sbP', 'sbOdds', 'win'}]}). Returns a summary dict, or None if there is nothing graded."""
    bets = [b for g in graded for b in g['bets'] if b['win'] is not None]
    if not bets:
        return None
    p = np.array([b['p'] for b in bets]); y = np.array([1.0 if b['win'] else 0.0 for b in bets])
    dates = sorted({g['date'] for g in graded if g.get('date')})
    rec = {'games': len(graded), 'bets': len(bets), 'from': dates[0] if dates else None, 'to': dates[-1] if dates else None,
           'expected': round(float(p.sum()), 1), 'actual': int(y.sum()), 'brier': round(float(((p - y) ** 2).mean()), 4)}
    cal = []; edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum():
            cal.append({'lo': int(round(lo * 100)), 'hi': int(round(hi * 100)), 'n': int(m.sum()), 'sim': round(float(p[m].mean() * 100), 1), 'hit': round(float(y[m].mean() * 100), 1)})
    rec['calibration'] = cal
    groups = {}
    for b in bets:
        gr = groups.setdefault(b['group'], {'n': 0, 'e': 0.0, 'a': 0, 'brier': 0.0})
        gr['n'] += 1; gr['e'] += b['p']; gr['a'] += 1 if b['win'] else 0; gr['brier'] += (b['p'] - (1.0 if b['win'] else 0.0)) ** 2
    rec['groups'] = sorted([{'group': k, 'n': v['n'], 'expected': round(v['e'], 1), 'actual': v['a'], 'brier': round(v['brier'] / v['n'], 4)} for k, v in groups.items()], key=lambda x: -x['n'])
    pr = [b for b in bets if b.get('sbP') is not None]
    if pr:
        pp = np.array([b['p'] for b in pr]); bp = np.array([b['sbP'] for b in pr]); yy = np.array([1.0 if b['win'] else 0.0 for b in pr])
        rec['priced'] = {'n': len(pr), 'sim': round(float(pp.sum()), 1), 'book': round(float(bp.sum()), 1), 'actual': int(yy.sum()),
                         'brierSim': round(float(((pp - yy) ** 2).mean()), 4), 'brierBook': round(float(((bp - yy) ** 2).mean()), 4)}
        liked = [b for b in pr if b['p'] - b['sbP'] >= 0.02 and b['p'] >= 0.25]
        if liked:
            u = [_units(b['sbOdds'], b['win']) for b in liked]
            rec['liked'] = {'n': len(liked), 'sim': round(float(np.mean([b['p'] for b in liked]) * 100), 1), 'book': round(float(np.mean([b['sbP'] for b in liked]) * 100), 1),
                            'hit': round(float(np.mean([1.0 if b['win'] else 0.0 for b in liked]) * 100), 1), 'units': round(float(sum(u)), 1), 'perBet': round(float(np.mean(u)), 3)}
    return rec

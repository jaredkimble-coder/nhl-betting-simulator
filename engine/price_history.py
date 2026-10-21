"""Keeps each game's earlier prices so the page can show how a price has moved since the first pull."""
import json, os


def _key(l):
    return l['group'] + '|' + l['label']


def snapshot(pack):
    """The best sportsbook price for every bet, as of this pull."""
    m = pack['meta']
    return {'at': m['asof'], 'prices': {_key(l): {'o': l['sb']['odds'], 'b': l['sb']['book'], 'p': l['sb']['p']} for l in m['legs'] if l.get('sb')}}


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return []


def attach_moves(pack, history):
    """Adds `mv` (the earliest price we saw, and when) to each bet that has an earlier price. Returns how many bets got one."""
    asof = pack['meta']['asof']; first = {}
    for snap in sorted(history, key=lambda s: s['at']):
        if snap['at'] >= asof:
            continue
        for k, v in snap['prices'].items():
            first.setdefault(k, (snap['at'], v))
    n = 0
    for l in pack['meta']['legs']:
        l.pop('mv', None)
        f = first.get(_key(l)) if l.get('sb') else None
        if f:
            l['mv'] = {'at': f[0], 'o': f[1]['o'], 'b': f[1]['b'], 'p': f[1]['p']}; n += 1
    return n


def record(history, pack, cap=24):
    """Adds this pull's snapshot (once per pull time), keeping the first one and the most recent ones."""
    snap = snapshot(pack)
    if any(s['at'] == snap['at'] for s in history):
        return history
    h = sorted(history + [snap], key=lambda s: s['at'])
    return h if len(h) <= cap else [h[0]] + h[-(cap - 1):]


def save(path, history):
    json.dump(history, open(path, 'w'))

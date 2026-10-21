"""Statistics layer for the simulator: real ice-time splits, per-player rates, ShotPropz shots- and goals-against-by-position matchup data, and MoneyPuck team and player expected-goals data."""
import json, math, unicodedata
import numpy as np


def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s or '') if not unicodedata.combining(c)).upper().replace('.', '').replace("'", '').replace('-', ' ').strip()


MP_XG_WEIGHT = 0.4        # best blend of expected goals and goals for predicting next season's scoring (336 forwards, 2024-25 to 2025-26)
# How much of a team's difference from the league carries into the next season (2024-25 to 2025-26, 32 teams). A team's number is pulled toward the league by the rest.
MP_SLOPE = {'takes': 0.59, 'draws': 0.58, 'ppq': 0.34, 'shq': 0.14, 'sf': 0.76, 'sa': 0.59}
LEAGUE_PEN = 3.49         # penalties taken per team per game, 2025-26, when no MoneyPuck data is available


class Stats:
    def __init__(self, d):
        self.toi = {norm(r['n']): r for r in d.get('iceTime', {}).get('rows', [])}
        self.summ = {norm(r['n']): r for r in d.get('summary', {}).get('rows', [])}
        self.sog = d.get('shotPropz', {}).get('sogAgainst', {})
        self.sog_avg = {pos: float(np.mean([v['sa'] for v in t.values()])) for pos, t in self.sog.items() if t}
        self.ga = d.get('shotPropz', {}).get('goalsAgainst', {})
        self.ga_avg = {pos: float(np.mean([v['ga'] for v in t.values()])) for pos, t in self.ga.items() if t}
        mp = d.get('moneyPuck') or {}
        self.mp_lg = mp.get('league') or {}
        self.mp_teams = mp.get('teams') or {}                       # {'home': {...}, 'away': {...}} season totals by game situation
        self.mp_sk = {norm(r['n']): r for r in mp.get('skaters', [])}
        self.pulled = d.get('pulledAt')

    # ---- ice time (seconds per game)
    def ev(self, name):
        r = self.toi.get(norm(name)); return r['ev'] if r and r.get('gp', 0) >= 10 else None
    def pp(self, name):
        r = self.toi.get(norm(name)); return r['pp'] if r and r.get('gp', 0) >= 10 else None
    def sh(self, name):
        r = self.toi.get(norm(name)); return r['sh'] if r and r.get('gp', 0) >= 10 else None

    # ---- per-game rates, shrunk toward a slot prior by games played
    def rate(self, name, field, slot_rate, k=20.0):
        r = self.summ.get(norm(name))
        if not r or not r.get('gp'):
            return slot_rate, 0.0
        w = r['gp'] / (r['gp'] + k)
        per = r[field] / r['gp']
        if field == 'g' and MP_XG_WEIGHT:      # expected goals are steadier than goals: blend them with the goals scored
            m = self.mp_sk.get(norm(name))
            if m and m.get('gp', 0) >= 10:
                per = (1 - MP_XG_WEIGHT) * per + MP_XG_WEIGHT * (m['xg'] / m['gp'])
        return w * per + (1 - w) * slot_rate, w

    # ---- ShotPropz: shots a team allows to a position, per game, shrunk toward the league average (early-season samples are small)
    def matchup(self, pos, opp_abbr, k=15.0):
        t = self.sog.get(pos, {}).get(opp_abbr)
        if not t or not self.sog_avg.get(pos):
            return 1.0
        w = t['gp'] / (t['gp'] + k)
        return 1.0 + w * (t['sa'] / self.sog_avg[pos] - 1.0)

    # ---- ShotPropz: goals a team allows to a position, per game. Goals are rarer and noisier than shots, so it is shrunk harder (k=30):
    # after 8 games the tilt is about a fifth of the raw gap to the league average.
    def goal_matchup(self, pos, opp_abbr, k=30.0):
        t = self.ga.get(pos, {}).get(opp_abbr)
        if not t or not self.ga_avg.get(pos):
            return 1.0
        w = t['gp'] / (t['gp'] + k)
        return 1.0 + w * (t['ga'] / self.ga_avg[pos] - 1.0)

    # ---- MoneyPuck, last season's team data. Each difference from the league is shrunk by how much it carried from 2024-25 to 2025-26.
    def _mp(self, side):
        t = self.mp_teams.get(side)
        return t if (t and t.get('gp') and t.get('all')) else None

    def mp_team_shots(self, side):
        """Shots on goal a team should take against this opponent (no market data)."""
        me, opp = self._mp(side), self._mp('away' if side == 'home' else 'home')
        lg = self.mp_lg.get('sogPerTeamGame')
        if not (me and opp and lg):
            return None
        sf = 1 + MP_SLOPE['sf'] * (me['all']['sogF'] / me['gp'] / lg - 1)
        sa = 1 + MP_SLOPE['sa'] * (opp['all']['sogA'] / opp['gp'] / lg - 1)
        return lg * sf * sa

    def mp_pen_rate(self, side):
        """How often `side` should take penalties in this game relative to the league (1.0 is typical): its own rate of taking them times the opponent's rate of drawing them."""
        me, opp = self._mp(side), self._mp('away' if side == 'home' else 'home')
        lg = self.mp_lg.get('penPerTeamGame')
        if not (me and opp and lg) or 'penF' not in me['all'] or 'penA' not in opp['all']:
            return None
        takes = me['all']['penF'] / me['gp'] / lg
        draws = opp['all']['penA'] / opp['gp'] / lg
        return (1 + MP_SLOPE['takes'] * (takes - 1)) * (1 + MP_SLOPE['draws'] * (draws - 1))

    def mp_st_defaults(self, side):
        """(expected power-play goals, expected shorthanded goals) for `side`, used only when no book posts a line. The league rate, scaled by how many
        penalties each team should take (so how many power plays) and by the team's own power-play and shorthanded scoring."""
        me, other = self._mp(side), ('away' if side == 'home' else 'home')
        lp, ls, lx = self.mp_lg.get('ppgPerTeamGame'), self.mp_lg.get('shgPerTeamGame'), self.mp_lg.get('ppXgF60')
        r_opp, r_me = self.mp_pen_rate(other), self.mp_pen_rate(side)
        if not (me and lp and ls and lx and r_opp and r_me and me.get('5on4') and me['5on4'].get('toi') and me.get('4on5')):
            return None
        ppq = (me['5on4']['xgF'] / (me['5on4']['toi'] / 3600.0)) / lx
        shq = (me['4on5']['gF'] / me['gp']) / ls
        return lp * (1 + MP_SLOPE['ppq'] * (ppq - 1)) * r_opp, ls * (1 + MP_SLOPE['shq'] * (shq - 1)) * r_me

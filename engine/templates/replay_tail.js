  /* ---------- entities ---------- */
  var layer = document.createElementNS(SVGNS, 'g'); rink.appendChild(layer);
  var tagLayer = document.createElementNS(SVGNS, 'g'); rink.appendChild(tagLayer);
  var trail = document.createElementNS(SVGNS, 'polyline');
  trail.setAttribute('fill', 'none'); trail.setAttribute('stroke-width', '3.5'); trail.setAttribute('stroke-linecap', 'round'); trail.setAttribute('stroke-linejoin', 'round'); trail.setAttribute('opacity', '0');
  function mk(tag, attrs, parent) { var e = document.createElementNS(SVGNS, tag); for (var k in attrs) { e.setAttribute(k, attrs[k]); } (parent || layer).appendChild(e); return e; }
  layer.appendChild(trail);
  function netLight(x) { var q = proj(x, 38.5); return mk('circle', { cx: q.x.toFixed(1), cy: (q.y - 14).toFixed(1), r: 10, fill: '#E6322B', opacity: 0 }); }
  var lightL = netLight(9), lightR = netLight(191);
  var FF = 'DM Sans, Arial, sans-serif';
  function makeSkater(side, idx, kind) {
    var g = mk('g', {}); var sh = mk('ellipse', { fill: '#000', opacity: .28 }, g);
    var c = mk('circle', { 'stroke-width': 2.2 }, g); var n = mk('text', { 'text-anchor': 'middle', 'font-weight': 700, 'font-family': FF }, g);
    var th = mk('text', { 'text-anchor': 'middle', 'font-weight': 700, 'font-size': 13, 'font-family': FF, fill: '#0B121D', stroke: '#0B121D', 'stroke-width': 4, 'stroke-linejoin': 'round' }, tagLayer);
    var t = mk('text', { 'text-anchor': 'middle', 'font-weight': 700, 'font-size': 13, 'font-family': FF, fill: '#fff' }, tagLayer);
    return { kind: kind || 'sk', side: side, idx: idx, g: g, sh: sh, c: c, n: n, th: th, t: t, x: 100, y: 90, tx: 100, ty: 90, pl: null, tag: null };
  }
  var ent = { away: { sk: [], gk: null, box: [], on: [] }, home: { sk: [], gk: null, box: [], on: [] }, puck: null };
  ['away', 'home'].forEach(function (s) {
    for (var i = 0; i < 6; i++) { ent[s].sk.push(makeSkater(s, i, 'sk')); }
    for (var b = 0; b < 2; b++) { ent[s].box.push(makeSkater(s, b, 'box')); }
    var g = mk('g', {}); var sh = mk('ellipse', { fill: '#000', opacity: .28 }, g); var r = mk('rect', { rx: 4, 'stroke-width': 2.2 }, g);
    var gth = mk('text', { 'text-anchor': 'middle', 'font-weight': 700, 'font-size': 13, 'font-family': FF, fill: '#0B121D', stroke: '#0B121D', 'stroke-width': 4, 'stroke-linejoin': 'round' }, tagLayer);
    var t = mk('text', { 'text-anchor': 'middle', 'font-weight': 700, 'font-size': 13, 'font-family': FF, fill: '#fff' }, tagLayer);
    ent[s].gk = { kind: 'gk', side: s, g: g, sh: sh, c: r, t: t, th: gth, x: 100, y: 42.5, tx: 100, ty: 42.5 };
  });
  var pg = mk('g', {}); var puckSh = mk('ellipse', { fill: '#000', opacity: .3 }, pg); var puckB = mk('ellipse', { fill: '#0A0A0A', stroke: '#fff', 'stroke-width': 1.4 }, pg);
  ent.puck = { g: pg, sh: puckSh, b: puckB, x: 100, y: 42.5 };
  var drawList = [];
  ['away', 'home'].forEach(function (s) { ent[s].sk.forEach(function (p) { drawList.push(p); }); ent[s].box.forEach(function (p) { drawList.push(p); }); drawList.push(ent[s].gk); });

  /* ---------- data and state ---------- */
  var G = GAMES[0], RO = G.roster, U = G.units, EV = G.events;
  var S = { gi: 0, playing: false, speed: 1, clock: 0, next: 0, play: null, stats: null, hot: {}, log: [], tickAcc: 0, goalAt: -99, goalRight: null };
  var dirty = true, lastLogKey = '', scrubDrag = false;
  function prep() { RO = G.roster; U = G.units; EV = G.events; }
  function newStats() {
    return { score: { away: 0, home: 0 }, players: {}, gk: { away: { saves: 0, ga: 0 }, home: { saves: 0, ga: 0 } }, teamSog: { away: 0, home: 0 }, p1: { away: 0, home: 0 },
             firstGoal: null, tenGoal: null, pulled: { away: null, home: null }, pim: { away: 0, home: 0 }, ppg: { away: 0, home: 0 }, pens: [], done: false };
  }
  function pl(st, i) { var r = RO[i], k = r.n; if (!st.players[k]) { st.players[k] = { n: r.n, no: r.no, team: r.side, g: 0, a: 0, sog: 0, pim: 0 }; } return st.players[k]; }
  function applyEvent(ev, st) {
    if (ev.type === 'goal') {
      st.score[ev.team] += 1; var sc = pl(st, ev.scorer); sc.g += 1; sc.sog += 1;
      ev.assists.forEach(function (a) { pl(st, a).a += 1; });
      st.gk[other(ev.team)].ga += 1; st.teamSog[ev.team] += 1;
      if (ev.strength === 'pp') { st.ppg[ev.team] += 1; }
      if (!st.firstGoal) { st.firstGoal = ev; }
      if (ev.t < 600 && !st.tenGoal) { st.tenGoal = ev; }
      if (ev.t < PERIOD) { st.p1[ev.team] += 1; }
    } else if (ev.type === 'save') { pl(st, ev.shooter).sog += 1; st.gk[other(ev.team)].saves += 1; st.teamSog[ev.team] += 1; }
    else if (ev.type === 'penalty') { st.pim[ev.team] += 2; pl(st, ev.player).pim += 2; st.pens.push({ team: ev.team, t: ev.t, end: ev.end, player: ev.player, kind: ev.kind }); }
    else if (ev.type === 'pull') { st.pulled[ev.team] = ev.t; }
    else if (ev.type === 'final') { st.done = true; }
  }
  function rebuild(upTo) { var st = newStats(); for (var i = 0; i < upTo; i++) { applyEvent(EV[i], st); } return st; }
  function activePens(side, t) { return S.stats.pens.filter(function (p) { return p.team === side && p.t <= t && t < p.end; }); }
  function skaters(side, t) { return 5 - Math.min(activePens(side, t).length, 2); }
  function special(side, t) {
    var o = other(side), a = skaters(side, t), b = skaters(o, t);
    if (a > b) { return { s: 'pp', a: a, b: b, rem: Math.min.apply(null, activePens(o, t).map(function (p) { return p.end - t; })) }; }
    if (a < b) { return { s: 'pk', a: a, b: b, rem: Math.min.apply(null, activePens(side, t).map(function (p) { return p.end - t; })) }; }
    return { s: a < 5 ? 'four' : 'ev', a: a, b: b, rem: 0 };
  }
  function gkOut(side) { return S.stats.pulled[side] !== null; }

  /* ---------- captions ---------- */
  function setCap(c) {
    var el = $('cap');
    if (!c) { el.className = 'cap'; return; }
    el.innerHTML = '<span>' + c.html + '</span>' + c.chips.map(function (x) { return '<span class="chip ' + x.c + '">' + esc(x.t) + '</span>'; }).join('');
    el.style.setProperty('--tc', c.tint); el.className = 'cap on';
  }
  var bigTimer = null;
  function showBig(html, ms) { var b = $('big'); b.innerHTML = html; b.className = 'big show'; if (bigTimer) { clearTimeout(bigTimer); bigTimer = null; } if (ms) { bigTimer = setTimeout(function () { b.className = 'big'; }, ms); } }
  function hideBig() { $('big').className = 'big'; if (bigTimer) { clearTimeout(bigTimer); bigTimer = null; } }
  function nm(i) { return RO[i].n; }
  function capFor(ev) {
    var T = G[ev.team || 'home'], st = S.stats;
    if (ev.type === 'save') {
      var gn = G.goalies[other(ev.team)], sp = special(ev.team, ev.t);
      var ch = [{ t: T.abbr + ' shot ' + (st.teamSog[ev.team] + 1), c: '' }, { t: 'Save ' + (st.gk[other(ev.team)].saves + 1), c: '' }];
      if (sp.s === 'pp') { ch.unshift({ t: 'Power play ' + sp.a + 'v' + sp.b, c: 'warn' }); } else if (sp.s === 'pk') { ch.unshift({ t: 'Shorthanded', c: '' }); }
      return { html: esc(nm(ev.shooter)) + ' shot, saved by ' + esc(gn ? lastName(gn.n) : 'the goalie'), chips: ch, tint: T.glow };
    }
    if (ev.type === 'goal') {
      var as = ev.assists.length ? ' from ' + ev.assists.map(function (a) { return lastName(nm(a)); }).join(' and ') : ', unassisted';
      var chips = [{ t: 'Goal', c: 'goal' }];
      var lab = { pp: 'Power-play goal', sh: 'Shorthanded goal', en: 'Empty net', tie: 'Extra attacker', ev4: '4 on 4', ot: 'Overtime' }[ev.strength];
      if (lab) { chips.push({ t: lab, c: 'warn' }); }
      return { html: esc(nm(ev.scorer)) + ' scores' + esc(as), chips: chips, tint: T.glow };
    }
    if (ev.type === 'penalty') { var sp2 = special(other(ev.team), ev.t); return { html: esc(nm(ev.player)) + ' (' + esc(T.abbr) + '), ' + esc(ev.kind) + ', 2 minutes', chips: [{ t: 'Penalty', c: 'warn' }, { t: G[other(ev.team)].abbr + ' power play', c: '' }], tint: G[other(ev.team)].glow }; }
    if (ev.type === 'pull') { return { html: esc(T.short) + ' pull the goalie for the extra attacker', chips: [{ t: '6 on 5', c: 'warn' }], tint: T.glow }; }
    if (ev.type === 'faceoff') { return { html: ev.k === 0 ? 'Puck drop' : 'Puck drop, ' + ORD[ev.k] + ' period', chips: [], tint: '#9DB0C1' }; }
    return null;
  }

  /* ---------- who is on the ice, and where ---------- */
  function dirOf(side, t) { var a = (Math.min(2, Math.floor(Math.min(t, TOTAL - 0.001) / PERIOD)) % 2 === 0) ? 1 : -1; return side === 'away' ? a : -a; }
  function benchIn(side) { return [100 + (side === 'away' ? -16 : 16), 88]; }
  function benchGk(side) { return [100 + (side === 'away' ? -10 : 10), -5]; }
  var ORDER = { D: 0, C: 1, W: 2 };
  function pickExtra(side, list) {
    for (var ln = 1; ln <= 3; ln++) { for (var i = 0; i < RO.length; i++) { if (RO[i].side === side && RO[i].pos !== 'D' && RO[i].line === ln && list.indexOf(i) < 0) { return i; } } }
    return null;
  }
  function setOn(side, uid) {
    var list = (U[uid] || []).slice();
    if (gkOut(side) && list.length === 5) { var ex = pickExtra(side, list); if (ex !== null) { list.push(ex); } }
    list.sort(function (a, b) { return ORDER[RO[a].pos] - ORDER[RO[b].pos] || a - b; });
    ent[side].sk.forEach(function (p, i) {
      var pli = i < list.length ? list[i] : null;
      if (pli !== p.pl && pli !== null) { var b = benchIn(side); p.x = b[0]; p.y = b[1]; }
      if (pli === null) { p.x = p.tx = 100; p.y = p.ty = 92; }
      p.pl = pli; p.tag = null;
    });
    ent[side].on = list;
  }
  function role(i, r) { return RO[i].pos === r; }
  function offSpots(list, nx, d, pp) {
    var o = {}, Ds = list.filter(function (i) { return role(i, 'D'); }), Cs = list.filter(function (i) { return role(i, 'C'); }), Ws = list.filter(function (i) { return role(i, 'W'); });
    Ds.forEach(function (i, k) { o[i] = [nx - d * (pp ? 42 : 38), 42.5 + (k % 2 ? 1 : -1) * (pp ? 15 : 17)]; });
    var wing = 0;                                     // extra centers play the wings
    Cs.forEach(function (i, k) { if (k === 0) { o[i] = [nx - d * (pp ? 18 : 22), 42.5]; } else { o[i] = [nx - d * (pp ? 28 : 26), 42.5 + (wing % 2 ? 1 : -1) * (pp ? 26 : 24)]; wing++; } });
    Ws.forEach(function (i) { o[i] = [nx - d * (pp ? 28 : 26), 42.5 + (wing % 2 ? 1 : -1) * (pp ? 26 : 24)]; wing++; });
    return o;
  }
  function defSpots(list, nx, d, pk) {
    var o = {}, Ds = list.filter(function (i) { return role(i, 'D'); }), Fs = list.filter(function (i) { return !role(i, 'D'); });
    Ds.forEach(function (i, k) { o[i] = [nx - d * (pk ? 11 : 10), 42.5 + (k % 2 ? 1 : -1) * (pk ? 9 : 13)]; });
    Fs.forEach(function (i, k) { o[i] = pk ? [nx - d * 22, 42.5 + (k % 2 ? 1 : -1) * 8] : (role(i, 'C') ? [nx - d * 24, 42.5] : [nx - d * 28, 42.5 + (k % 2 ? 1 : -1) * 20]); });
    return o;
  }
  function faceSpots(list, d) {
    var o = {}, Ds = list.filter(function (i) { return role(i, 'D'); }), Cs = list.filter(function (i) { return role(i, 'C'); }), Ws = list.filter(function (i) { return role(i, 'W'); });
    Cs.forEach(function (i, k) { o[i] = [100 - d * (3 + k * 8), 42.5 + k * 6]; }); Ws.forEach(function (i, k) { o[i] = [100 - d * 9, 42.5 + (k % 2 ? 1 : -1) * 14]; }); Ds.forEach(function (i, k) { o[i] = [100 - d * 26, 42.5 + (k % 2 ? 1 : -1) * 10]; });
    return o;
  }
  function applySpots(side, spots) { ent[side].sk.forEach(function (p) { if (p.pl !== null && spots[p.pl]) { p.tx = spots[p.pl][0]; p.ty = spots[p.pl][1]; } }); }
  function setGoalie(side, t, tx, ty) {
    var g = ent[side].gk;
    if (gkOut(side) && t >= S.stats.pulled[side]) { var b = benchGk(side); g.tx = b[0]; g.ty = b[1]; }
    else { g.tx = tx; g.ty = ty; }
  }
  function shotSpot(i, nx, d, k) {
    var r1 = rnd(k * 3 + 1), r2 = rnd(k * 3 + 2), sg = rnd(k * 3 + 3) > 0.5 ? 1 : -1;
    if (role(i, 'D')) { return [nx - d * (36 + r1 * 14), 42.5 + sg * (8 + r2 * 16)]; }          // defensemen shoot from the point
    if (role(i, 'C')) { return [nx - d * (11 + r1 * 13), 42.5 + (r2 - 0.5) * 16]; }               // centers from the slot
    return [nx - d * (14 + r1 * 15), 42.5 + sg * (13 + r2 * 13)];                                 // wingers from the circles
  }
  function startSpot(i, nx, d, k, sg) {
    var r1 = rnd(k * 5 + 1), r2 = rnd(k * 5 + 2);
    if (role(i, 'D')) { return [nx - d * (40 + r1 * 8), 42.5 + sg * (11 + r2 * 9)]; }
    if (role(i, 'C')) { return [nx - d * (20 + r1 * 10), 42.5 + sg * (2 + r2 * 8)]; }
    return [nx - d * (28 + r1 * 8), 42.5 + sg * (22 + r2 * 6)];
  }

  /* ---------- plays ---------- */
  function startPlay(ev, idx) {
    var P = { ev: ev, idx: idx, t: 0, si: -1, fromT: S.clock, toT: ev.t, applied: false, trail: [] };
    var side = ev.team, d = side ? dirOf(side, ev.t) : 1, nx = d > 0 ? 189 : 11;
    P.side = side; P.d = d; P.nx = nx;
    if (ev.type === 'save') { P.segs = [{ k: 'setup', len: 0.9 }, { k: 'shot', len: 0.45 }, { k: 'save', len: 0.9 }]; }
    else if (ev.type === 'goal') {
      var chain = ev.assists.slice().reverse().concat([ev.scorer]);
      P.chain = chain; P.L = chain.length;
      P.segs = [{ k: 'setup', len: 1.0 }];
      if (P.L === 1) { P.segs.push({ k: 'carry', len: 1.2 }); } else { for (var j = 0; j < P.L - 1; j++) { P.segs.push({ k: 'pass', i: j, len: 0.75 }); } }
      P.segs.push({ k: 'shot', len: 0.45 }, { k: 'goal', len: 1.9 }, { k: 'reset', len: 0.7 });
    } else if (ev.type === 'penalty') { P.segs = [{ k: 'whistle', len: 1.0 }, { k: 'box', len: 1.4 }]; }
    else if (ev.type === 'pull') { P.segs = [{ k: 'hold', len: 1.7 }]; }
    else if (ev.type === 'faceoff') { P.segs = [{ k: 'face', len: 1.3 }]; }
    else if (ev.type === 'period_end') { P.segs = [{ k: 'end', len: 1.9 }]; }
    else { P.segs = [{ k: 'final', len: 0.8 }]; }
    if (ev.t - S.clock > 1.5) { P.segs.unshift({ k: 'roll', len: 0.5 }); }   // run the clock up to the event, then stage the play at that moment
    P.total = 0; P.segs.forEach(function (s) { P.total += s.len; });
    if (ev.type === 'save' || ev.type === 'goal') { setCap(null); }
    S.play = P; trail.setAttribute('opacity', '0'); trail.setAttribute('points', '');
  }
  function bothOn(ev) { if (ev.on) { setOn('away', ev.on.away); setOn('home', ev.on.home); } }
  function formations(ev, att, d, nx) {
    var opp = other(att), spA = special(att, ev.t), spO = special(opp, ev.t);
    applySpots(att, offSpots(ent[att].on, nx, d, spA.s === 'pp'));
    applySpots(opp, defSpots(ent[opp].on, nx, d, spO.s === 'pk' || spO.s === 'four'));
  }
  function enterSeg(P) {
    var seg = P.segs[P.si], ev = P.ev, side = P.side, d = P.d, nx = P.nx, opp = side ? other(side) : null;
    if (seg.k === 'setup' && ev.type === 'save') {
      bothOn(ev); clearTags(); formations(ev, side, d, nx);
      var sp = shotSpot(ev.shooter, nx, d, P.idx); P.spot = sp;
      var o = {}; o[ev.shooter] = sp; applySpots(side, o);
      ent[side].sk.forEach(function (p) { if (p.pl === ev.shooter) { p.tag = { n: nm(ev.shooter), role: 'shot' }; } });
      if (role(ev.shooter, 'D')) { var scr = ent[side].on.filter(function (i) { return !role(i, 'D') && i !== ev.shooter; })[0]; if (scr !== undefined) { var o2 = {}; o2[scr] = [nx - d * 7, 42.5 + (rnd(P.idx) - 0.5) * 6]; applySpots(side, o2); } }
      setGoalie(opp, ev.t, nx - d * 4, 42.5 + (sp[1] - 42.5) * 0.22); setGoalie(side, ev.t, d > 0 ? 14 : 186, 42.5);
    } else if (seg.k === 'setup' && ev.type === 'goal') {
      bothOn(ev); clearTags(); formations(ev, side, d, nx);
      var pts = [], sgn = rnd(P.idx * 7) > 0.5 ? 1 : -1;
      var shotAt = ev.strength === 'en' ? [nx - d * (46 + rnd(P.idx) * 12), 42.5 + (rnd(P.idx + 1) - 0.5) * 14] : shotSpot(ev.scorer, nx, d, P.idx);
      P.shotAt = shotAt;
      if (P.L === 1) { pts = [[nx - d * 66, 42.5 + sgn * 18], shotAt]; }
      else { for (var k = 0; k < P.L - 1; k++) { pts.push(startSpot(P.chain[k], nx, d, P.idx + k, (k % 2 ? -1 : 1) * sgn)); } pts.push(shotAt); }
      P.pts = pts; P.shotY = shotAt[1] * 0 + 42.5 + (rnd(P.idx + 9) - 0.5) * 5;
      var spots = {}; if (P.L === 1) { spots[P.chain[0]] = pts[0]; } else { P.chain.forEach(function (i, k) { spots[i] = pts[k]; }); }
      applySpots(side, spots);
      ent[side].sk.forEach(function (p) { var k = p.pl === null ? -1 : P.chain.indexOf(p.pl); if (k >= 0) { p.tag = { n: nm(p.pl), role: k === P.chain.length - 1 ? 'goal' : 'assist' }; } });
      if (role(ev.scorer, 'D')) { var scr2 = ent[side].on.filter(function (i) { return !role(i, 'D') && P.chain.indexOf(i) < 0; })[0]; if (scr2 !== undefined) { var o3 = {}; o3[scr2] = [nx - d * 7, 42.5]; applySpots(side, o3); } }
      setGoalie(opp, ev.t, nx - d * 4, 42.5); setGoalie(side, ev.t, d > 0 ? 14 : 186, 42.5);
    } else if (seg.k === 'carry') {
      var carrier = ent[side].sk.filter(function (p) { return p.pl === P.chain[0]; })[0]; if (carrier) { carrier.tx = P.pts[1][0]; carrier.ty = P.pts[1][1]; }
    } else if (seg.k === 'shot') {
      var src = ent[side].sk.filter(function (p) { return p.pl === (ev.type === 'goal' ? ev.scorer : ev.shooter); })[0] || ent[side].sk[0];
      P.shotFrom = [src.x, src.y];
      if (ev.type === 'save') { P.shotTo = [nx - d * 3.8, ent[opp].gk.ty + (rnd(P.idx + 4) - 0.5) * 3]; setGoalie(opp, ev.t, nx - d * 4, P.shotTo[1]); }
      else { P.shotTo = [nx + d * 1.4, P.shotY]; if (!(gkOut(opp) && ev.t >= S.stats.pulled[opp])) { ent[opp].gk.ty = clamp(P.shotY + (P.shotY > 42.5 ? -5 : 5), 36, 49); } }
      var c = capFor(ev); S.tint = c.tint; setCap(c);
    } else if (seg.k === 'save') {
      applyEvent(ev, S.stats); P.applied = true; markHot(ev.shooter); dirty = true;
      P.saveTo = [nx - d * (9 + rnd(P.idx + 5) * 6), 42.5 + (rnd(P.idx + 6) > 0.5 ? 1 : -1) * (14 + rnd(P.idx + 7) * 8)];
    } else if (seg.k === 'goal') {
      applyEvent(ev, S.stats); P.applied = true; markHot(ev.scorer); ev.assists.forEach(markHot); dirty = true; S.goalRight = nx > 100; goalFx(ev);
    } else if (seg.k === 'reset') {
      clearTags(); var nd = dirOf(side, ev.t); applySpots(side, faceSpots(ent[side].on, nd)); applySpots(opp, faceSpots(ent[opp].on, -nd));
      setGoalie(opp, ev.t, dirOf(opp, ev.t) > 0 ? 14 : 186, 42.5);
    } else if (seg.k === 'whistle') {
      bothOn(ev); clearTags(); var nd2 = dirOf(ev.team, ev.t);
      applySpots('away', faceSpots(ent.away.on, dirOf('away', ev.t))); applySpots('home', faceSpots(ent.home.on, dirOf('home', ev.t)));
      var c2 = capFor(ev); S.tint = c2.tint; setCap(c2);
      applyEvent(ev, S.stats); P.applied = true; dirty = true; markHot(ev.player);
    } else if (seg.k === 'box') {
      // the penalty taker leaves the ice; the next play puts the shorthanded unit out
      ent[ev.team].sk.forEach(function (p) { if (p.pl === ev.player) { var b = boxPos(ev.team, 0); p.tx = b[0]; p.ty = b[1] + 14; } });
    } else if (seg.k === 'hold' && ev.type === 'pull') {
      applyEvent(ev, S.stats); P.applied = true; dirty = true; bothOn(ev); clearTags();
      var td = dirOf(ev.team, ev.t), tnx = td > 0 ? 189 : 11; formations(ev, ev.team, td, tnx);
      setGoalie(ev.team, ev.t, 0, 0); var c3 = capFor(ev); S.tint = c3.tint; setCap(c3);
    } else if (seg.k === 'face') {
      bothOn(ev); clearTags(); hideBig(); applySpots('away', faceSpots(ent.away.on, dirOf('away', ev.t))); applySpots('home', faceSpots(ent.home.on, dirOf('home', ev.t)));
      ['away', 'home'].forEach(function (s) { setGoalie(s, ev.t, dirOf(s, ev.t) > 0 ? 14 : 186, 42.5); });
      var c4 = capFor(ev); setCap(c4);
    } else if (seg.k === 'end') {
      clearTags(); setCap(null);
      showBig('End of the ' + ORD[ev.k - 1] + ' period<small>' + esc(G.away.abbr) + ' ' + S.stats.score.away + ', ' + esc(G.home.abbr) + ' ' + S.stats.score.home + '</small>', 0);
    }
  }
  function clearTags() { ['away', 'home'].forEach(function (s) { ent[s].sk.forEach(function (p) { p.tag = null; }); }); }
  function markHot(i) { S.hot[nm(i)] = performance.now() / 1000 + 2; dirty = true; }
  function goalFx(ev) {
    var T = G[ev.team];
    var sc = $(ev.team === 'away' ? 'scA' : 'scH'); sc.className = 'sc'; void sc.offsetWidth; sc.className = 'sc bump';
    S.goalAt = performance.now() / 1000;
    var fl = $('flash'); fl.style.setProperty('--tc', T.glow); fl.className = 'flash'; void fl.offsetWidth; fl.className = 'flash go';
    $('live').textContent = 'Goal, ' + T.name + '. ' + nm(ev.scorer) + '. ' + G.away.abbr + ' ' + S.stats.score.away + ', ' + G.home.abbr + ' ' + S.stats.score.home + '.';
  }
  function logFor(ev, st) {
    var T = G[ev.team || 'home'];
    if (ev.type === 'goal') { var tag = { pp: ' (power play)', sh: ' (shorthanded)', en: ' (empty net)', ot: ' (overtime)' }[ev.strength] || ''; return { t: ev.t, g: true, html: 'Goal, ' + esc(T.abbr) + ': ' + esc(nm(ev.scorer)) + (ev.assists.length ? ' (' + ev.assists.map(function (a) { return esc(lastName(nm(a))); }).join(', ') + ')' : ' (unassisted)') + tag + ' \u2013 ' + st.score.away + '\u2013' + st.score.home }; }
    if (ev.type === 'save') { return { t: ev.t, g: false, html: esc(nm(ev.shooter)) + ' (' + esc(T.abbr) + ') shot, saved' }; }
    if (ev.type === 'penalty') { return { t: ev.t, g: false, html: 'Penalty, ' + esc(T.abbr) + ': ' + esc(nm(ev.player)) + ', ' + esc(ev.kind) + ' (2:00)' }; }
    if (ev.type === 'pull') { return { t: ev.t, g: false, html: esc(T.short) + ' pull the goalie' }; }
    if (ev.type === 'period_end') { return { t: ev.t, g: false, html: 'End of the ' + ORD[ev.k - 1] + ' period \u2013 ' + st.score.away + '\u2013' + st.score.home }; }
    if (ev.type === 'faceoff') { return { t: ev.t, g: false, html: ev.k === 0 ? 'Puck drop' : 'Puck drop, ' + ORD[ev.k] + ' period' }; }
    return null;
  }
  function boxPos(side, slot) { return [100 + (side === 'away' ? -1 : 1) * (14 + slot * 8), -6]; }

  function stepPlay(dt) {
    var P = S.play, cs = calm ? 3 : Math.min(S.speed, 3.2);
    P.t += dt * cs;
    var acc = 0, idx = P.segs.length - 1, u = 1;
    for (var i = 0; i < P.segs.length; i++) { if (P.t < acc + P.segs[i].len) { idx = i; u = (P.t - acc) / P.segs[i].len; break; } acc += P.segs[i].len; }
    if (P.t >= P.total) { idx = P.segs.length - 1; u = 1; }
    while (P.si < idx) { P.si += 1; enterSeg(P); }
    var seg = P.segs[idx], ev = P.ev, pu = ent.puck, side = P.side, nx = P.nx, d = P.d;
    S.clock = seg.k === 'roll' ? lerp(P.fromT, P.toT, ease(u)) : P.toT;
    if (seg.k === 'roll') { trail.setAttribute('opacity', '0'); }
    else if (ev.type === 'save' || ev.type === 'goal') {
      var shooterId = ev.type === 'goal' ? P.chain[0] : ev.shooter;
      var holder = ent[side].sk.filter(function (p) { return p.pl === shooterId; })[0];
      if (seg.k === 'setup') { if (holder) { pu.x = holder.x; pu.y = holder.y; } }
      else if (seg.k === 'carry') { if (holder) { pu.x = holder.x + d * 1.5; pu.y = holder.y; } }
      else if (seg.k === 'pass') {
        var a = ent[side].sk.filter(function (p) { return p.pl === P.chain[seg.i]; })[0], b = ent[side].sk.filter(function (p) { return p.pl === P.chain[seg.i + 1]; })[0], e1 = ease(u);
        if (a && b) { pu.x = lerp(a.x, b.x, e1); pu.y = lerp(a.y, b.y, e1); }
      } else if (seg.k === 'shot') { var e2 = u * u; pu.x = lerp(P.shotFrom[0], P.shotTo[0], e2); pu.y = lerp(P.shotFrom[1], P.shotTo[1], e2); }
      else if (seg.k === 'save') { var e3 = ease(u); pu.x = lerp(P.shotTo[0], P.saveTo[0], e3); pu.y = lerp(P.shotTo[1], P.saveTo[1], e3); }
      else if (seg.k === 'goal') { pu.x = P.shotTo[0]; pu.y = P.shotTo[1]; }
      else if (seg.k === 'reset') { var e4 = ease(u); pu.x = lerp(P.shotTo[0], 100, e4); pu.y = lerp(P.shotTo[1], 42.5, e4); }
      if (seg.k !== 'reset' && seg.k !== 'goal' && !(seg.k === 'save' && u > 0.6)) { P.trail.push([pu.x, pu.y]); if (P.trail.length > 90) { P.trail.shift(); } }
      trail.setAttribute('stroke', G[side].glow); trail.setAttribute('points', poly(P.trail, -2)); trail.setAttribute('opacity', seg.k === 'goal' ? clamp(1 - u, 0, 1) * 0.7 : (seg.k === 'reset' ? 0 : 0.75));
    } else { pu.x = lerp(pu.x, 100, 0.1); pu.y = lerp(pu.y, 42.5, 0.1); trail.setAttribute('opacity', '0'); }
    if (P.t >= P.total) { finishPlay(); }
  }
  function finishPlay() {
    var P = S.play, ev = P.ev;
    if (!P.applied) { applyEvent(ev, S.stats); dirty = true; }
    var lg = logFor(ev, S.stats); if (lg) { S.log.push(lg); }
    S.clock = ev.t; S.next = P.idx + 1; S.play = null; clearTags(); trail.setAttribute('opacity', '0');
    if (ev.type === 'period_end') { hideBig(); }
    if (ev.type === 'final') { endGame(); }
    dirty = true;
  }
  function endGame() {
    S.playing = false; S.clock = TOTAL; setCap(null);
    var f = G.final, w = f.home > f.away ? G.home : G.away;
    showBig('Final<small>' + esc(G.away.short) + ' ' + f.away + ', ' + esc(G.home.short) + ' ' + f.home + '</small>', 0);
    $('live').textContent = 'Final. ' + G.away.name + ' ' + f.away + ', ' + G.home.name + ' ' + f.home + '.';
  }

  /* ---------- rendering ---------- */
  function renderEntities(dt, k) {
    var f = 1 - Math.exp(-k * dt);
    var boxes = { away: activePens('away', S.clock), home: activePens('home', S.clock) };
    ['away', 'home'].forEach(function (s) {
      ent[s].box.forEach(function (p, i) {
        var pen = boxes[s][i];
        if (pen) { p.pl = pen.player; var b = boxPos(s, i); p.x = p.tx = b[0]; p.y = p.ty = b[1]; p.rem = pen.end - S.clock; } else { p.pl = null; }
      });
    });
    drawList.forEach(function (p) { p.x += (p.tx - p.x) * f; p.y += (p.ty - p.y) * f; });
    var order = drawList.slice().sort(function (a, b) { return a.y - b.y; });
    order.forEach(function (p) {
      var q = proj(p.x, p.y), r = 3.7 * KX * q.s, T = G[p.side];
      if (p.kind === 'sk' || p.kind === 'box') {
        var vis = p.pl !== null;
        p.g.setAttribute('opacity', vis ? 1 : 0);
        p.sh.setAttribute('cx', q.x.toFixed(1)); p.sh.setAttribute('cy', q.y.toFixed(1)); p.sh.setAttribute('rx', (r * 1.05).toFixed(1)); p.sh.setAttribute('ry', (r * 0.4).toFixed(1));
        p.c.setAttribute('cx', q.x.toFixed(1)); p.c.setAttribute('cy', (q.y - r * 0.95).toFixed(1)); p.c.setAttribute('r', r.toFixed(1)); p.c.setAttribute('fill', T.fill); p.c.setAttribute('stroke', p.kind === 'box' ? '#F5C542' : T.ring);
        var dark = parseInt(T.fill.substr(1, 2), 16) * 0.3 + parseInt(T.fill.substr(3, 2), 16) * 0.59 + parseInt(T.fill.substr(5, 2), 16) * 0.11 < 128;
        p.n.setAttribute('x', q.x.toFixed(1)); p.n.setAttribute('y', (q.y - r * 0.95 + r * 0.36).toFixed(1)); p.n.setAttribute('font-size', (r * 0.95).toFixed(1)); p.n.setAttribute('fill', dark ? '#fff' : '#111');
        p.n.textContent = vis && RO[p.pl].no !== null && RO[p.pl].no !== undefined ? RO[p.pl].no : '';
        var label = '';
        if (vis && p.kind === 'box') { var rm = Math.max(0, Math.round(p.rem || 0)); label = lastName(RO[p.pl].n) + ' ' + Math.floor(rm / 60) + ':' + pad2(rm % 60); }
        else if (vis && p.tag) { label = lastName(p.tag.n); }
        p.t.textContent = label; p.th.textContent = label;
        p.t.setAttribute('x', q.x.toFixed(1)); p.t.setAttribute('y', (q.y - r * 2.3).toFixed(1)); p.th.setAttribute('x', q.x.toFixed(1)); p.th.setAttribute('y', (q.y - r * 2.3).toFixed(1));
        p.t.setAttribute('opacity', vis ? 1 : 0); p.th.setAttribute('opacity', vis ? 1 : 0);
      } else {
        var w = r * 2.5, h = r * 2.1;
        p.sh.setAttribute('cx', q.x.toFixed(1)); p.sh.setAttribute('cy', q.y.toFixed(1)); p.sh.setAttribute('rx', (w * 0.6).toFixed(1)); p.sh.setAttribute('ry', (r * 0.45).toFixed(1));
        p.c.setAttribute('x', (q.x - w / 2).toFixed(1)); p.c.setAttribute('y', (q.y - h).toFixed(1)); p.c.setAttribute('width', w.toFixed(1)); p.c.setAttribute('height', h.toFixed(1)); p.c.setAttribute('fill', T.fill); p.c.setAttribute('stroke', T.ring);
        var gi = G.goalies[p.side], showLbl = !!(S.play && S.play.ev && S.play.ev.type === 'save' && p.side !== S.play.side && S.play.si >= 1), ltxt = showLbl && gi ? lastName(gi.n) : '';
        p.t.textContent = ltxt; p.th.textContent = ltxt;
        p.t.setAttribute('x', q.x.toFixed(1)); p.t.setAttribute('y', (q.y + r * 1.9).toFixed(1)); p.th.setAttribute('x', q.x.toFixed(1)); p.th.setAttribute('y', (q.y + r * 1.9).toFixed(1));
      }
      layer.appendChild(p.g);
    });
    var pq = proj(ent.puck.x, ent.puck.y), pr = 6.4 * pq.s;
    ent.puck.sh.setAttribute('cx', pq.x.toFixed(1)); ent.puck.sh.setAttribute('cy', pq.y.toFixed(1)); ent.puck.sh.setAttribute('rx', (pr * 1.1).toFixed(1)); ent.puck.sh.setAttribute('ry', (pr * 0.5).toFixed(1));
    ent.puck.b.setAttribute('cx', pq.x.toFixed(1)); ent.puck.b.setAttribute('cy', (pq.y - 3).toFixed(1)); ent.puck.b.setAttribute('rx', pr.toFixed(1)); ent.puck.b.setAttribute('ry', (pr * 0.55).toFixed(1));
    layer.appendChild(trail); layer.appendChild(ent.puck.g);
  }
  function fmtRem(s) { s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ':' + pad2(s % 60); }
  function skaterRows(side) {
    var rows = [], st = S.stats; for (var k in st.players) { if (st.players[k].team === side) { rows.push(st.players[k]); } }
    rows.sort(function (a, b) { return (b.g * 3 + b.a * 2 + b.sog * 0.5) - (a.g * 3 + a.a * 2 + a.sog * 0.5); });
    return rows.slice(0, 5);
  }
  function panelHtml(side) {
    var T = G[side], st = S.stats, gi = G.goalies[side], gs = st.gk[side], now = performance.now() / 1000, t = S.clock, html;
    html = '<h3>' + esc(T.abbr) + '<span>' + esc(T.name) + '</span></h3>';
    html += '<div class="sec" style="--sec:' + T.glow + '">Goalie</div><div class="row"><b>' + esc(gi ? gi.n : 'Goalie') + '</b><i>' + (gkOut(side) && t >= st.pulled[side] ? 'pulled' : gs.saves + '/' + (gs.saves + gs.ga) + ' SV') + '</i></div>';
    html += '<div class="sec" style="--sec:' + T.glow + '">On ice</div><div class="oi">' + (ent[side].on.length ? ent[side].on.map(function (i) { return '#' + (RO[i].no === null ? '' : RO[i].no) + ' ' + esc(lastName(RO[i].n)); }).join(', ') : '\u2014') + '</div>';
    var bx = activePens(side, t); if (bx.length) { html += '<div class="box">Penalty box: ' + bx.map(function (p) { return esc(lastName(RO[p.player].n)) + ' ' + fmtRem(p.end - t); }).join(', ') + '</div>'; }
    html += '<div class="sec" style="--sec:' + T.glow + '">Skaters</div>';
    var rows = skaterRows(side); if (!rows.length) { html += '<div class="row"><b>\u2014</b><i></i></div>'; }
    rows.forEach(function (r) {
      var bits = []; if (r.g) { bits.push(r.g + ' G'); } if (r.a) { bits.push(r.a + ' A'); } bits.push(r.sog + ' SOG'); if (r.pim) { bits.push(r.pim + ' PIM'); }
      html += '<div class="row' + ((S.hot[r.n] || 0) > now ? ' hot' : '') + '"><b>' + esc(lastName(r.n)) + '</b><i>' + bits.join(' \u00b7 ') + '</i></div>';
    });
    html += '<div class="row" style="margin-top:4px"><b>Shots / PIM / PPG</b><i>' + st.teamSog[side] + ' / ' + st.pim[side] + ' / ' + st.ppg[side] + '</i></div>';
    return html;
  }
  function renderPanels() { ['away', 'home'].forEach(function (s) { var el = $(s === 'away' ? 'pA' : 'pH'); if (el) { el.innerHTML = panelHtml(s); el.style.setProperty('--tc', G[s].glow); } }); }
  function propStatus(row, st, clock, done) {
    var sc = st.score, fav = row.team, tot = sc.away + sc.home;
    function res(text, state) { return { text: text, state: state }; }
    function over(val, line, unit) { if (val > line) { return res(val + ' ' + unit, 'hit'); } return done ? res(val + ' ' + unit, 'miss') : res(val + ' ' + unit + ', needs ' + (Math.floor(line) + 1 - val) + ' more', 'live'); }
    switch (row.kind) {
      case 'ml': { var diff = sc[fav] - sc[other(fav)]; var t = G[fav].abbr + ' ' + (diff > 0 ? 'up ' + diff : (diff < 0 ? 'down ' + (-diff) : 'tied')); return done ? res(t, diff > 0 ? 'hit' : 'miss') : res(t, 'live'); }
      case 'spread': { var m = sc[fav] - sc[other(fav)] + row.pts; var t2 = 'margin ' + (sc[fav] - sc[other(fav)] >= 0 ? '+' : '') + (sc[fav] - sc[other(fav)]); return done ? res(t2, m > 0 ? 'hit' : 'miss') : res(t2 + (m > 0 ? ', covering' : ', not covering'), 'live'); }
      case 'total': return over(tot, row.line, 'goals');
      case 'tt': return over(sc[row.team], row.line, G[row.team].abbr + ' goals');
      case 'atg': { var p = st.players[row.player]; var gg = p ? p.g : 0; return gg >= 1 ? res(gg + ' goal' + (gg > 1 ? 's' : ''), 'hit') : (done ? res('0 goals', 'miss') : res('0 goals', 'live')); }
      case 'pts': { var p2 = st.players[row.player]; return over(p2 ? p2.g + p2.a : 0, row.line, 'pts'); }
      case 'ast': { var p3 = st.players[row.player]; return over(p3 ? p3.a : 0, row.line, 'assists'); }
      case 'fgs': { var fg = st.firstGoal; if (!fg) { return done ? res('no goals', 'miss') : res('no goal yet', 'live'); } return nm(fg.scorer) === row.player ? res('scored first', 'hit') : res('first: ' + lastName(nm(fg.scorer)), 'miss'); }
      case 'saves': return over(st.gk[row.goalie].saves, row.line, 'saves');
      case 'sog': { var p4 = st.players[row.player]; return over(p4 ? p4.sog : 0, row.line, 'shots'); }
      case 'reg3': { var tied = sc.away === sc.home; return done ? res(tied ? 'tied after 60' : 'decided in 60', tied ? 'hit' : 'miss') : res(tied ? 'tied now' : 'not tied now', 'live'); }
      case 'ten': { if (st.tenGoal) { return res('yes, ' + elapsedText(st.tenGoal.t), 'hit'); } return clock >= 600 ? res('none in first 10', 'miss') : res('no goal yet', 'live'); }
      case 'p1puck': { var mm = st.p1[fav] - st.p1[other(fav)] + row.pts, t3 = '1st-period margin ' + (st.p1[fav] - st.p1[other(fav)] >= 0 ? '+' : '') + (st.p1[fav] - st.p1[other(fav)]); return clock >= PERIOD ? res(t3, mm > 0 ? 'hit' : 'miss') : res(t3, 'live'); }
      case 'p1total': { var n1 = st.p1.away + st.p1.home; if (n1 > row.line) { return res(n1 + ' goals', 'hit'); } return clock >= PERIOD ? res(n1 + ' goals', 'miss') : res(n1 + ' goals so far', 'live'); }
    }
    return res('', 'live');
  }
  function renderProps() {
    var st = S.stats, out = [], lastG = '';
    out.push('<table><thead><tr><th scope="col">Market group</th><th scope="col">Reference line</th><th scope="col" class="n">Sim chance</th><th scope="col">Right now</th><th scope="col">Status</th></tr></thead><tbody>');
    G.props.forEach(function (row) {
      var s = propStatus(row, st, S.clock, st.done), grp = row.group === lastG ? '' : row.group; lastG = row.group;
      out.push('<tr><td>' + (grp ? '<b>' + esc(grp) + '</b>' : '') + '</td><td>' + esc(row.label) + '</td><td class="n">' + (row.p === null ? '' : pct(row.p) + ' <span class="grp">(' + fmtPrice(row.fair) + ')</span>') + '</td><td>' + esc(s.text) + '</td><td><span class="st ' + s.state + '">' + (s.state === 'hit' ? 'Hit' : (s.state === 'miss' ? 'Miss' : 'Live')) + '</span></td></tr>');
    });
    out.push('</tbody></table>'); $('propsBox').innerHTML = out.join('');
  }
  function renderLog() {
    var box = $('logBox'); if (!S.log.length) { box.innerHTML = '<p class="empty">Press play to start the game.</p>'; return; }
    box.innerHTML = '<ol class="log">' + S.log.slice(-70).reverse().map(function (l) { var f = clockText(l.t); return '<li class="' + (l.g ? 'g' : '') + '"><span class="w">' + ORD[f.p] + ' ' + f.text + '</span><span>' + l.html + '</span></li>'; }).join('') + '</ol>';
  }
  function renderWhy() {
    var st = G.stats, f = G.final, winner = f.home > f.away ? G.home : G.away, top = G.scorelines, max = top[0].p;
    var bars = top.map(function (s) { var T = s.winner === 'home' ? G.home : G.away, me = s.s === (f.away + '\u2013' + f.home); return '<div class="bar' + (me ? ' me' : '') + '" style="--tc:' + T.glow + '"><span>' + s.s + '</span><i style="width:' + (s.p / max * 100).toFixed(1) + '%"></i><span>' + pct(s.p, 1) + '</span></div>'; }).join('');
    var ag = Object.keys(G.agree).map(function (k) { return '<div class="bar" style="--tc:#6FA8F0"><span style="grid-column:1/3;text-align:left">' + esc(k) + '</span><span>' + pct(G.agree[k]) + '</span></div>'; }).join('');
    $('whyBox').innerHTML =
      '<p>This is sim #' + G.simIndex.toLocaleString('en-US') + ' of ' + G.simCount.toLocaleString('en-US') + '. Every simulated game was scored on how often its results were the most likely result for each line, across all 14 market groups plus penalties and special teams, with each group counting equally. This one scored highest. It finished ' + f.away + '\u2013' + f.home + ', with ' + esc(winner.name) + ' winning.</p>' +
      '<h3 style="margin-top:14px">How closely it follows the favorite outcome, by market group</h3><div class="bars" role="img" aria-label="Share of each market group\u2019s lines where this game landed on the likelier side">' + ag + '</div>' +
      '<h3 style="margin-top:14px">Six most likely final scores</h3><div class="bars">' + bars + '</div>' +
      '<p class="fine" style="margin-top:14px">Penalties, power plays, lines, and who is on the ice come from the simulation. Lines and pairs are inferred from market production, and power-play units from the power-play-points props. Shot and save times and the movement are illustrative; shots and saves are made consistent here while the pricing workbook keeps shots on goal and saves on their own market numbers. Teams average ' + st.penA.toFixed(1) + ' and ' + st.penH.toFixed(1) + ' penalties; the late-game goalie pull ties about ' + pct(st.qTie) + ' of the time and the leader adds an empty-netter about ' + pct(st.rEN) + '.</p>';
  }
  function renderHud() {
    var c = Math.min(S.clock, TOTAL), f = clockText(c), st = S.stats;
    $('ck').textContent = S.clock >= TOTAL ? '0:00' : f.text; $('pd').textContent = S.clock >= TOTAL ? 'Final' : ORD[f.p] + ' period';
    $('scA').textContent = st.score.away; $('scH').textContent = st.score.home;
    ['away', 'home'].forEach(function (s) {
      var sp = special(s, c), el = $(s === 'away' ? 'spA' : 'spH');
      el.textContent = sp.s === 'pp' ? 'PP ' + sp.a + 'v' + sp.b + ' ' + fmtRem(sp.rem) : (sp.s === 'pk' ? 'PK ' + fmtRem(sp.rem) : (sp.s === 'four' ? '4v4' : '')); el.className = 'spc' + (sp.s === 'pk' ? ' pk' : '');
    });
    if (!scrubDrag) { $('scrub').value = Math.round(c); }
    $('play').textContent = S.playing ? 'Pause' : (st.done ? 'Replay' : (S.clock > 0 ? 'Resume' : 'Play'));
    $('next').disabled = !nextGoalIdx();
    var badge = 'Simulated game \u00b7 sim #' + G.simIndex.toLocaleString('en-US') + ' of ' + G.simCount.toLocaleString('en-US'); if ($('badge').textContent !== badge) { $('badge').textContent = badge; }
  }
  function nextGoalIdx() { for (var i = S.next; i < EV.length; i++) { if (EV[i].type === 'goal') { return i + 1; } } return 0; }

  /* ---------- flow ---------- */
  function idleOn(idx) {
    var e = null; for (var i = Math.min(idx, EV.length) - 1; i >= 0; i--) { if (EV[i].on) { e = EV[i]; break; } }
    if (!e) { for (var j = 0; j < EV.length; j++) { if (EV[j].on) { e = EV[j]; break; } } }
    if (e) { setOn('away', e.on.away); setOn('home', e.on.home); }
    applySpots('away', faceSpots(ent.away.on, dirOf('away', S.clock))); applySpots('home', faceSpots(ent.home.on, dirOf('home', S.clock)));
    ['away', 'home'].forEach(function (s) { setGoalie(s, S.clock, dirOf(s, S.clock) > 0 ? 14 : 186, 42.5); });
    drawList.forEach(function (p) { p.x = p.tx; p.y = p.ty; }); ent.puck.x = 100; ent.puck.y = 42.5;
  }
  function seek(sec) {
    S.play = null; hideBig(); setCap(null); trail.setAttribute('opacity', '0');
    var idx = 0; while (idx < EV.length && EV[idx].t <= sec && EV[idx].type !== 'final') { idx++; }
    if (sec >= TOTAL) { idx = EV.length - 1; }
    S.stats = newStats(); S.log = [];
    for (var i = 0; i < idx; i++) { applyEvent(EV[i], S.stats); var lg = logFor(EV[i], S.stats); if (lg && EV[i].type !== 'save') { S.log.push(lg); } }
    S.next = idx; S.clock = Math.min(sec, TOTAL); S.hot = {};
    idleOn(idx); dirty = true; lastLogKey = '';
    if (sec >= TOTAL) { applyEvent(EV[EV.length - 1], S.stats); endGame(); }
  }
  function loadGame(i) {
    S.gi = i; G = GAMES[i]; prep(); S.playing = false; S.play = null;
    $('abA').textContent = G.away.abbr; $('abH').textContent = G.home.abbr; $('tmA').style.setProperty('--tc', G.away.glow); $('tmH').style.setProperty('--tc', G.home.glow);
    var tk = $('ticks'); tk.innerHTML = '';
    EV.forEach(function (e) {
      if (e.type === 'goal' || e.type === 'penalty' || e.type === 'pull') {
        var s = document.createElement('div'); s.className = 'tick'; s.style.left = (e.t / TOTAL * 100) + '%';
        s.style.setProperty('--tc', e.type === 'goal' ? G[e.team].glow : '#F5C542'); if (e.type !== 'goal') { s.style.opacity = '.75'; s.style.height = '8px'; } tk.appendChild(s);
      }
    });
    var tabs = $('tabs').children; for (var k = 0; k < tabs.length; k++) { tabs[k].setAttribute('aria-pressed', k === i ? 'true' : 'false'); }
    buildPanels(); seek(0); renderWhy();
  }
  function buildPanels() {
    var mob = window.innerWidth <= 760, host = mob ? $('panelsMob') : $('panelsDesk');
    $('panelsMob').hidden = !mob; $('panelsDesk').hidden = mob;
    host.innerHTML = '<div class="panel away" id="pA"></div><div class="panel home" id="pH"></div>';
  }
  function tick(dt) {
    if (S.playing && !S.stats.done) {
      if (!S.play) { if (S.next < EV.length) { startPlay(EV[S.next], S.next); } else { endGame(); } }
      if (S.play) { stepPlay(dt); }
    }
    renderEntities(dt, S.play ? 4.6 * Math.min(S.speed, 3) : 3);
    var age = performance.now() / 1000 - S.goalAt, lop = 0;
    if (age >= 0 && age < 2.4) { lop = calm ? 0.9 : (0.35 + 0.65 * Math.abs(Math.sin(age * 9))) * (1 - age / 2.4); }
    lightL.setAttribute('opacity', S.goalRight === false ? lop : 0); lightR.setAttribute('opacity', S.goalRight === true ? lop : 0);
    S.tickAcc += dt;
    if (dirty || S.tickAcc > 0.3) { S.tickAcc = 0; dirty = false; renderPanels(); renderProps(); var lk = S.gi + ':' + S.log.length; if (lk !== lastLogKey) { lastLogKey = lk; renderLog(); } }
    renderHud();
  }

  /* ---------- wiring ---------- */
  var tabsEl = $('tabs');
  GAMES.forEach(function (g, i) {
    var b = document.createElement('button'); b.type = 'button'; b.className = 'tab'; b.setAttribute('aria-pressed', 'false');
    b.innerHTML = esc(g.away.short) + ' at ' + esc(g.home.short) + '<small>' + g.final.away + '\u2013' + g.final.home + '</small>'; b.addEventListener('click', function () { loadGame(i); }); tabsEl.appendChild(b);
  });
  $('play').addEventListener('click', function () { if (S.stats.done) { seek(0); S.playing = true; return; } S.playing = !S.playing; if (S.playing) { hideBig(); } });
  $('restart').addEventListener('click', function () { seek(0); S.playing = true; });
  $('next').addEventListener('click', function () { var ni = nextGoalIdx(); if (!ni) { return; } seek(Math.max(0, EV[ni - 1].t - 45)); S.playing = true; });
  var speeds = $('speeds').querySelectorAll('button');
  for (var si = 0; si < speeds.length; si++) { (function (btn) { btn.addEventListener('click', function () { S.speed = parseFloat(btn.getAttribute('data-s')); for (var j = 0; j < speeds.length; j++) { speeds[j].setAttribute('aria-pressed', speeds[j] === btn ? 'true' : 'false'); } }); })(speeds[si]); }
  var scrub = $('scrub');
  scrub.addEventListener('input', function () { scrubDrag = true; seek(parseFloat(scrub.value)); });
  scrub.addEventListener('change', function () { scrubDrag = false; });
  document.addEventListener('keydown', function (e) { if (e.code === 'Space' && !/^(BUTTON|INPUT)$/.test((e.target && e.target.tagName) || '')) { e.preventDefault(); $('play').click(); } });
  function fit() { try { var w = $('field').getBoundingClientRect().width; if (w) { $('stage').style.setProperty('--u', (w / 100) + 'px'); } } catch (e) { /* no layout */ } var mob = window.innerWidth <= 760; if (S.stats && (mob === $('panelsMob').hidden)) { buildPanels(); dirty = true; } }
  window.addEventListener('resize', fit);

  loadGame(0); fit();
  var lastTs = 0;
  function frame(ts) { var dt = lastTs ? Math.min((ts - lastTs) / 1000, 0.05) : 0.016; lastTs = ts; tick(dt); window.requestAnimationFrame(frame); }
  window.requestAnimationFrame(frame);
  window.__replay = { S: S, tick: tick, load: loadGame, games: GAMES, ent: ent, seek: seek, propStatus: propStatus, getG: function () { return G; }, special: special };
})();
</script>
</body>
</html>

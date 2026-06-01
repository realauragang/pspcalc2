#!/usr/bin/env python3
"""
psp_live.py - LIVE MLB pre-stat-pick board
===========================================
Run this on your own machine (where outside API calls aren't sandboxed). It:

  1. pulls the day's schedule + probable pitchers + posted lineups,
  2. pulls every player's season stats and handedness,
  3. pulls ballpark weather (temperature + wind),
  4. runs the PSP engine (odds-ratio matchup blend + L/R platoon + park/weather),
  5. estimates popularity (well-known = chalk, obscure = niche),
  6. prints the SAFEST and NICHE picks for one criterion, and writes CSV + HTML.

All data comes from public endpoints (statsapi.mlb.com - the feed behind
mlb.com/stats - and open-meteo.com). NO pip installs needed; standard library only.

USAGE
  python3 psp_live.py --date today --stat TB --threshold 1
  python3 psp_live.py --date 2026-05-29 --stat K --threshold 5
  python3 psp_live.py --date today --stat RBI --threshold 2 --team Dodgers
  python3 psp_live.py --selftest                 # no network: prove the engine works

NOTES
  * Official lineups post ~2-4 hours before first pitch. Run close to game time
    for real batting orders; games without posted lineups are listed and their
    hitters are skipped (probable pitchers still work for K props anytime).
  * Park HR factors and field orientations below are APPROXIMATE - edit freely
    or drop in Statcast values. Temperature physics applies regardless.
"""

import argparse, csv, datetime, json, math, random, sys, urllib.parse, urllib.request

API = "https://statsapi.mlb.com/api/v1"
UA = {"User-Agent": "psp-live/1.0 (personal use)"}

# ---------------------------------------------------------------- engine
CATS = ["1B", "2B", "3B", "HR", "BB", "K", "OUT"]
TBV = [1, 2, 3, 4, 0, 0, 0]
HITV = [1, 1, 1, 1, 0, 0, 0]
HRV = [0, 0, 0, 1, 0, 0, 0]
LEAGUE = {"1B": 0.142, "2B": 0.045, "3B": 0.004, "HR": 0.033, "BB": 0.09, "K": 0.225, "OUT": 0.461}
LEAGUE_K = 0.225
PA_BY_SPOT = {1: 4.6, 2: 4.5, 3: 4.4, 4: 4.3, 5: 4.1, 6: 4.0, 7: 3.9, 8: 3.8, 9: 3.7}
EPS = 1e-6


def clamp(x, a, b):
    return min(max(x, a), b)


def _odds(p):
    p = clamp(p, EPS, 1 - EPS)
    return p / (1 - p)


def odds_ratio(player, opp, league):
    o = _odds(player) * _odds(opp) / _odds(league)
    return o / (1 + o)


def fnum(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def inum(x, d=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return d


def rates_from_counts(pa, h, d2, t3, hr, bb, so):
    """Exact per-PA outcome rates from counting stats (works for batters faced too)."""
    pa = max(pa, 1)
    singles = max(h - d2 - t3 - hr, 0)
    out = max(pa - (h + bb + so), 0)
    raw = {"1B": singles, "2B": d2, "3B": t3, "HR": hr, "BB": bb, "K": so, "OUT": out}
    s = sum(raw.values()) or 1
    return {k: max(v / s, EPS) for k, v in raw.items()}


def platoon(bat_hand, p_hand):
    """Batters fare better vs opposite-handed pitchers; switch hitters always do."""
    if not bat_hand or not p_hand:
        return {"hit": 1.0, "hr": 1.0, "k": 1.0, "adv": 0}
    eff = ("R" if p_hand == "L" else "L") if bat_hand == "S" else bat_hand
    adv = eff != p_hand
    if adv:
        return {"hit": 1.06, "hr": 1.12, "k": 0.94, "adv": 1}
    return {"hit": 0.95, "hr": 0.90, "k": 1.07, "adv": -1}


def adjusted_dist(b, p, park_hr, park_hits, temp, wind_out, plat, co):
    adj = {c: odds_ratio(b[c], p[c], LEAGUE[c]) for c in CATS}
    hr = park_hr / 100.0
    hr *= 1 + co["temp_hr"] * (temp - 70)
    hr *= 1 + co["wind_hr"] * wind_out
    hr = max(hr, 0.05)
    adj["HR"] *= hr
    bip = (1 + co["bip"] * (hr - 1)) * (park_hits / 100.0)
    for c in ("1B", "2B", "3B"):
        adj[c] *= bip
    adj["HR"] *= plat["hr"]
    for c in ("1B", "2B", "3B"):
        adj[c] *= plat["hit"]
    adj["K"] *= plat["k"]
    arr = [adj[c] for c in CATS]
    s = sum(arr) or 1
    return [x / s for x in arr]


def _poisson(lam):
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1


def _binom(n, p):
    return sum(1 for _ in range(n) if random.random() < p)


def sim_batter(dist, spot, stat, thr, team_obp, n):
    cum, a = [], 0.0
    for p in dist:
        a += p
        cum.append(a)
    e = PA_BY_SPOT.get(spot, 4.0)
    base, frac = int(e), e - int(e)
    mr = clamp(team_obp * 1.8, 0.3, 1.1)
    hit = 0
    for _ in range(n):
        pa = base + (1 if random.random() < frac else 0)
        tb = h = hr = rbi = 0
        for _ in range(pa):
            u = random.random()
            c = 0
            while c < 6 and u > cum[c]:
                c += 1
            tb += TBV[c]; h += HITV[c]; hr += HRV[c]
            if stat == "RBI":
                R = min(_poisson(mr), 3)
                if c == 3: rbi += 1 + R
                elif c == 2: rbi += R
                elif c == 1: rbi += _binom(R, 0.6)
                elif c == 0: rbi += _binom(R, 0.3)
        val = {"TB": tb, "H": h, "HR": hr, "RBI": rbi}[stat]
        if val >= thr:
            hit += 1
    return hit / n


def sim_pitcher_k(p_k, opp_k, exp_ip, thr, n):
    kr = odds_ratio(opp_k, p_k, LEAGUE_K)
    bf_e = exp_ip * 4.3
    cnt = 0
    for _ in range(n):
        bf = max(int(round(random.gauss(bf_e, 2.0))), 6)
        ks = _binom(bf, kr)
        if ks >= thr:
            cnt += 1
    return cnt / n


# popularity proxy ------------------------------------------------------------
def hitter_prominence(ops, pa):
    pr = clamp((ops - 0.6) / 0.4, 0, 1)
    pr = pr * 0.85 + clamp(pa / 250.0, 0, 1) * 0.15
    return clamp(pr, 0, 1)


def pitcher_prominence(kpct, era):
    pr = clamp((kpct - 0.15) / 0.2, 0, 1)
    pr = pr * 0.7 + clamp((5.0 - era) / 3.0, 0, 1) * 0.3
    return clamp(pr, 0, 1)


def est_rank(pr, max_rank):
    return max(1, round(1 + (1 - pr) * (max_rank - 1)))


def tier_of(rank):
    if rank <= 5: return "CHALK"
    if rank <= 15: return "POPULAR"
    if rank <= 30: return "MID"
    if rank <= 45: return "NICHE"
    return "DEEP"


# ------------------------------------------------ approximate park config (edit me)
PARK_HR = {
    "Coors Field": 118, "Great American Ball Park": 112, "Yankee Stadium": 110,
    "Citizens Bank Park": 107, "Globe Life Field": 103, "Fenway Park": 104,
    "Wrigley Field": 103, "Dodger Stadium": 104, "Truist Park": 103,
    "Chase Field": 103, "Rogers Centre": 103, "Nationals Park": 101,
    "Target Field": 99, "Busch Stadium": 97, "Comerica Park": 95,
    "Kauffman Stadium": 95, "loanDepot park": 92, "Oracle Park": 90,
    "Petco Park": 94, "T-Mobile Park": 93, "Angel Stadium": 99, "Citi Field": 97,
}
# Bearing (deg) from home plate toward center field, for the wind-out component.
# Approximate; leave a park out -> wind effect set to 0 there (temp still applies).
PARK_CF_BEARING = {
    "Coors Field": 0, "Yankee Stadium": 25, "Fenway Park": 50, "Wrigley Field": 30,
    "Oracle Park": 90, "Dodger Stadium": 25, "Great American Ball Park": 10,
}


def wind_out_component(speed, direction_from, cf_bearing):
    """Signed mph of wind blowing toward center (out). +out, -in."""
    if speed is None or direction_from is None or cf_bearing is None:
        return 0.0
    wind_to = (direction_from + 180) % 360
    return speed * math.cos(math.radians(wind_to - cf_bearing))


# ----------------------------------------------------------------- networking
def http_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def load_all_stats(season):
    hit = http_json(f"{API}/stats?stats=season&group=hitting&season={season}&sportId=1&gameType=R&playerPool=all&limit=4000")
    pit = http_json(f"{API}/stats?stats=season&group=pitching&season={season}&sportId=1&gameType=R&playerPool=all&limit=4000")
    hidx = {s["player"]["id"]: s["stat"] for s in hit["stats"][0]["splits"]}
    pidx = {s["player"]["id"]: s["stat"] for s in pit["stats"][0]["splits"]}
    return hidx, pidx


def load_schedule(date):
    url = f"{API}/schedule?sportId=1&date={date}&hydrate=probablePitcher,lineups,team,venue"
    d = http_json(url)
    return d.get("dates", [{}])[0].get("games", []) if d.get("dates") else []


def load_hands(ids):
    out = {}
    ids = [i for i in ids if i]
    for i in range(0, len(ids), 60):
        chunk = ids[i:i + 60]
        d = http_json(f"{API}/people?personIds={','.join(map(str, chunk))}")
        for p in d.get("people", []):
            out[p["id"]] = (p.get("batSide", {}).get("code"), p.get("pitchHand", {}).get("code"))
    return out


def load_venue_coords(ids):
    out = {}
    ids = [i for i in ids if i]
    if not ids:
        return out
    d = http_json(f"{API}/venues?venueIds={','.join(map(str, set(ids)))}&hydrate=location")
    for v in d.get("venues", []):
        loc = v.get("location", {}).get("defaultCoordinates", {})
        out[v["id"]] = (loc.get("latitude"), loc.get("longitude"))
    return out


_weather_cache = {}
def get_weather(lat, lng):
    if lat is None or lng is None:
        return None, None, None
    key = (round(lat, 2), round(lng, 2))
    if key in _weather_cache:
        return _weather_cache[key]
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}"
               f"&current=temperature_2m,wind_speed_10m,wind_direction_10m"
               f"&temperature_unit=fahrenheit&wind_speed_unit=mph")
        c = http_json(url).get("current", {})
        res = (c.get("temperature_2m"), c.get("wind_speed_10m"), c.get("wind_direction_10m"))
    except Exception:
        res = (None, None, None)
    _weather_cache[key] = res
    return res


# --------------------------------------------------------------- build & rank
def batter_rates_from_stat(st):
    return rates_from_counts(
        inum(st.get("plateAppearances")), inum(st.get("hits")),
        inum(st.get("doubles")), inum(st.get("triples")), inum(st.get("homeRuns")),
        inum(st.get("baseOnBalls")) + inum(st.get("hitByPitch")), inum(st.get("strikeOuts")))


def pitcher_allowed_rates(st):
    bf = inum(st.get("battersFaced"))
    if bf < 10:
        return None
    return rates_from_counts(
        bf, inum(st.get("hits")), inum(st.get("doubles")), inum(st.get("triples")),
        inum(st.get("homeRuns")), inum(st.get("baseOnBalls")) + inum(st.get("hitByPitch")),
        inum(st.get("strikeOuts")))


LEAGUE_PITCHER = {"1B": 0.142, "2B": 0.045, "3B": 0.004, "HR": 0.033, "BB": 0.09, "K": 0.225, "OUT": 0.461}


def park_for(venue):
    return PARK_HR.get(venue, 100), PARK_HR.get(venue, 100) // 2 + 50 if False else 100  # hits factor ~neutral default


def build_board(args):
    season = args.season
    co = {"temp_hr": args.temp_hr, "wind_hr": args.wind_hr, "bip": 0.40}
    stat, thr = args.stat, args.threshold
    is_pitcher_line = (stat == "K")

    print(f"Pulling {season} season stats…", file=sys.stderr)
    hidx, pidx = load_all_stats(season)
    print(f"Pulling schedule for {args.date}…", file=sys.stderr)
    games = load_schedule(args.date)
    if not games:
        print("No games found for that date.", file=sys.stderr)
        return [], {}

    if args.team:
        tl = args.team.lower()
        games = [g for g in games
                 if tl in g["teams"]["home"]["team"]["name"].lower()
                 or tl in g["teams"]["away"]["team"]["name"].lower()]

    # collect ids we need handedness for
    pid_ids, hit_ids, venue_ids = set(), set(), set()
    parsed = []
    for g in games:
        venue = g.get("venue", {})
        venue_ids.add(venue.get("id"))
        home, away = g["teams"]["home"], g["teams"]["away"]
        hp = home.get("probablePitcher", {}) or {}
        ap = away.get("probablePitcher", {}) or {}
        lu = g.get("lineups", {}) or {}
        home_lu = lu.get("homePlayers", []) or []
        away_lu = lu.get("awayPlayers", []) or []
        for pl in home_lu + away_lu:
            hit_ids.add(pl["id"])
        for pp in (hp, ap):
            if pp.get("id"):
                pid_ids.add(pp["id"])
        parsed.append({
            "venue": venue.get("name"), "venue_id": venue.get("id"),
            "home_team": home["team"]["name"], "away_team": away["team"]["name"],
            "home_pp": hp, "away_pp": ap,
            "home_lu": home_lu, "away_lu": away_lu,
        })

    print("Pulling handedness…", file=sys.stderr)
    hands = load_hands(list(pid_ids | hit_ids))
    coords = {}
    if not args.no_weather:
        print("Pulling venue coords + weather…", file=sys.stderr)
        coords = load_venue_coords(list(venue_ids))

    def weather_for(vid, vname):
        if args.no_weather or vid not in coords:
            return 70.0, 0.0
        lat, lng = coords[vid]
        t, ws, wd = get_weather(lat, lng)
        temp = fnum(t, 70.0)
        wout = wind_out_component(fnum(ws, 0.0), fnum(wd, None) if wd is not None else None,
                                  PARK_CF_BEARING.get(vname))
        return temp, wout

    rows = []
    skipped_lineups = []

    if not is_pitcher_line:
        for g in parsed:
            if not g["home_lu"] and not g["away_lu"]:
                skipped_lineups.append(f"{g['away_team']} @ {g['home_team']}")
            temp, wout = weather_for(g["venue_id"], g["venue"])
            phr = PARK_HR.get(g["venue"], 100)
            for side, lineup, opp_pp, team in (
                ("home", g["home_lu"], g["away_pp"], g["home_team"]),
                ("away", g["away_lu"], g["home_pp"], g["away_team"])):
                opp_id = (opp_pp or {}).get("id")
                opp_st = pidx.get(opp_id)
                p_rates = pitcher_allowed_rates(opp_st) if opp_st else None
                p_hand = hands.get(opp_id, (None, None))[1]
                opp_name = (opp_pp or {}).get("fullName", "TBD")
                for i, pl in enumerate(lineup):
                    st = hidx.get(pl["id"])
                    if not st or inum(st.get("plateAppearances")) < 10:
                        continue
                    b_rates = batter_rates_from_stat(st)
                    bat_hand = hands.get(pl["id"], (None, None))[0]
                    plat = platoon(bat_hand, p_hand)
                    dist = adjusted_dist(b_rates, p_rates or LEAGUE_PITCHER,
                                         phr, 100, temp, wout, plat, co)
                    p = sim_batter(dist, i + 1, stat, thr, fnum(st.get("obp"), 0.32), args.sims)
                    ops = fnum(st.get("ops"), fnum(st.get("obp"), .31) + fnum(st.get("slg"), .4))
                    prom = hitter_prominence(ops, inum(st.get("plateAppearances")))
                    rank = est_rank(prom, args.max_rank)
                    pts = args.points_base * rank
                    rows.append({
                        "name": pl.get("fullName", str(pl["id"])), "team": team,
                        "vs": f"{opp_name} ({p_hand or '?'})", "hand": bat_hand or "?",
                        "spot": i + 1, "p": p, "rank": rank, "tier": tier_of(rank),
                        "points": pts, "ev": p * pts, "plat": plat["adv"],
                    })
    else:
        for g in parsed:
            for side, pp, opp_lu, team in (
                ("home", g["home_pp"], g["away_lu"], g["home_team"]),
                ("away", g["away_pp"], g["home_lu"], g["away_team"])):
                pid = (pp or {}).get("id")
                st = pidx.get(pid)
                if not st:
                    continue
                bf = inum(st.get("battersFaced"))
                if bf < 20:
                    continue
                p_k = inum(st.get("strikeOuts")) / max(bf, 1)
                ip_parts = str(st.get("inningsPitched", "0")).split(".")
                ip = inum(ip_parts[0]) + (inum(ip_parts[1]) / 3 if len(ip_parts) > 1 else 0)
                gs = inum(st.get("gamesStarted")) or inum(st.get("gamesPlayed")) or 1
                exp_ip = ip / max(gs, 1)
                # opposing lineup K rate
                so = pa = 0
                for pl in opp_lu:
                    hs = hidx.get(pl["id"])
                    if hs:
                        so += inum(hs.get("strikeOuts")); pa += inum(hs.get("plateAppearances"))
                opp_k = (so / pa) if pa > 50 else 0.22
                p = sim_pitcher_k(p_k, opp_k, exp_ip, thr, args.sims)
                prom = pitcher_prominence(p_k, fnum(st.get("era"), 4.5))
                rank = est_rank(prom, args.max_rank)
                pts = args.points_base * rank
                p_hand = hands.get(pid, (None, None))[1]
                rows.append({
                    "name": (pp or {}).get("fullName", str(pid)), "team": team,
                    "vs": "opp lineup", "hand": p_hand or "?", "spot": "-",
                    "p": p, "rank": rank, "tier": tier_of(rank),
                    "points": pts, "ev": p * pts, "plat": 0,
                })

    meta = {"games": len(parsed), "skipped_lineups": skipped_lineups,
            "weather": not args.no_weather, "label": f"{thr}+ {stat}"}
    return rows, meta


# ------------------------------------------------------------------- output
def print_board(rows, meta, args):
    label = meta["label"]
    print("\n" + "=" * 78)
    print(f" PSP EDGE  ·  {args.date}  ·  {label}  ·  {meta['games']} games  ·  "
          f"weather {'ON' if meta['weather'] else 'OFF'}")
    print("=" * 78)
    if meta["skipped_lineups"]:
        print(f" Lineups not posted yet (hitters skipped): {', '.join(meta['skipped_lineups'])}")
    if not rows:
        print(" No qualifying picks. (For hitters, run closer to game time so lineups are posted.)")
        return

    by_p = sorted(rows, key=lambda r: r["p"], reverse=True)
    by_ev = sorted(rows, key=lambda r: r["ev"], reverse=True)
    niche = [r for r in by_ev if r["rank"] >= args.niche_rank]

    def line(r):
        plat = "+" if r["plat"] > 0 else "-" if r["plat"] < 0 else "~"
        return (f"  {r['name'][:20]:<20} {r['team'][:14]:<14} "
                f"P={r['p']*100:5.1f}%  {r['tier']:<7}#{r['rank']:<3} "
                f"{r['points']:>4}pts  EV={r['ev']:6.1f}  plat {plat}  vs {r['vs'][:24]}")

    print("\n RECOMMENDATIONS")
    print(" SAFEST :"); print(line(by_p[0]))
    print(" BEST EV:"); print(line(by_ev[0]))
    if niche:
        print(f" NICHE  (rank >= {args.niche_rank}):"); print(line(niche[0]))

    sort = by_ev if args.sort == "ev" else by_p
    print(f"\n FULL BOARD (sorted by {args.sort})")
    for i, r in enumerate(sort[:args.top], 1):
        print(f"{i:>2}." + line(r))
    print()


def write_csv(rows, args):
    path = f"psp_board_{args.date}_{args.stat}{args.threshold}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "team", "vs", "bat/throw", "spot", "P(hit)", "tier", "est_rank", "points", "EV", "platoon"])
        for r in sorted(rows, key=lambda x: x["ev"], reverse=True):
            w.writerow([r["name"], r["team"], r["vs"], r["hand"], r["spot"],
                        f"{r['p']:.4f}", r["tier"], r["rank"], r["points"], f"{r['ev']:.1f}", r["plat"]])
    return path


def write_html(rows, meta, args):
    path = f"psp_board_{args.date}_{args.stat}{args.threshold}.html"
    sort = sorted(rows, key=lambda x: (x["ev"] if args.sort == "ev" else x["p"]), reverse=True)
    cells = ""
    for i, r in enumerate(sort[:args.top], 1):
        col = "#3fd99b" if r["p"] >= .78 else "#ffb000" if r["p"] >= .55 else "#ff6363"
        plat = "+" if r["plat"] > 0 else "−" if r["plat"] < 0 else "~"
        cells += (f"<tr><td>{i}</td><td class=n>{r['name']}</td><td>{r['team']}</td>"
                  f"<td style='color:{col}'>{r['p']*100:.1f}%</td><td>{r['tier']} #{r['rank']}</td>"
                  f"<td>{r['points']}</td><td class=ev>{r['ev']:.0f}</td><td>{plat}</td><td class=vs>{r['vs']}</td></tr>")
    html = f"""<!doctype html><meta charset=utf-8><title>PSP {args.date} {meta['label']}</title>
<style>body{{background:#070b11;color:#e7eef6;font:14px ui-monospace,Menlo,monospace;padding:24px}}
h1{{color:#ffb000;letter-spacing:2px}}table{{border-collapse:collapse;width:100%;margin-top:14px}}
th,td{{padding:7px 10px;border-bottom:1px solid #21303f;text-align:left}}th{{color:#8597a9;font-size:11px;text-transform:uppercase}}
.n{{font-weight:700}}.ev{{color:#ffb000}}.vs{{color:#8597a9}}small{{color:#5b6b7c}}</style>
<h1>PSP EDGE</h1><small>{args.date} · {meta['label']} · {meta['games']} games · weather {'ON' if meta['weather'] else 'OFF'}</small>
<table><tr><th>#</th><th>player</th><th>team</th><th>P(hit)</th><th>tier</th><th>pts</th><th>EV</th><th>plat</th><th>vs</th></tr>{cells}</table>"""
    with open(path, "w") as f:
        f.write(html)
    return path


# ------------------------------------------------------------------- selftest
def selftest(args):
    print("SELFTEST (synthetic data, no network)\n")
    co = {"temp_hr": 0.01, "wind_hr": 0.015, "bip": 0.40}
    star = rates_from_counts(600, 170, 35, 5, 45, 95, 140)        # big bat
    contact = rates_from_counts(520, 156, 30, 6, 9, 45, 90)       # contact
    soft_p = rates_from_counts(680, 192, 40, 4, 30, 70, 120)      # hittable pitcher allowed
    avg_p = LEAGUE_PITCHER
    samples = [
        ("Big Bat (R vs R)", sim_batter(adjusted_dist(star, avg_p, 100, 100, 70, 0, platoon("R", "R"), co), 2, "TB", 1, .33, args.sims), 2),
        ("Contact (L vs R, hot/out)", sim_batter(adjusted_dist(contact, soft_p, 112, 100, 88, 10, platoon("L", "R"), co), 3, "TB", 1, .34, args.sims), 24),
        ("Big Bat 3+RBI", sim_batter(adjusted_dist(star, avg_p, 100, 100, 70, 0, platoon("R", "R"), co), 4, "RBI", 3, .33, args.sims), 2),
        ("Ace 5+K", sim_pitcher_k(0.31, 0.27, 5.8, 5, args.sims), 4),
        ("Niche arm 6+K", sim_pitcher_k(0.26, 0.25, 5.6, 6, args.sims), 18),
    ]
    for name, p, rank in samples:
        pts = args.points_base * rank
        print(f"  {name:<28} P={p*100:5.1f}%  rank#{rank:<3} {pts:>4}pts  EV={p*pts:6.1f}")
    print("\nEngine OK.")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Live MLB pre-stat-pick board")
    ap.add_argument("--date", default="today", help="YYYY-MM-DD or 'today'")
    ap.add_argument("--season", type=int, default=datetime.date.today().year)
    ap.add_argument("--stat", default="TB", choices=["TB", "H", "HR", "RBI", "K"])
    ap.add_argument("--threshold", type=int, default=1)
    ap.add_argument("--team", default=None, help="filter to a team name substring")
    ap.add_argument("--sims", type=int, default=8000)
    ap.add_argument("--points-base", dest="points_base", type=int, default=10)
    ap.add_argument("--max-rank", dest="max_rank", type=int, default=60)
    ap.add_argument("--niche-rank", dest="niche_rank", type=int, default=16)
    ap.add_argument("--temp-hr", dest="temp_hr", type=float, default=0.01)
    ap.add_argument("--wind-hr", dest="wind_hr", type=float, default=0.015)
    ap.add_argument("--no-weather", dest="no_weather", action="store_true")
    ap.add_argument("--sort", default="ev", choices=["ev", "p"])
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.date == "today":
        args.date = datetime.date.today().isoformat()

    if args.selftest:
        selftest(args)
        return

    try:
        rows, meta = build_board(args)
    except urllib.error.URLError as e:
        print(f"\nNetwork error reaching the API: {e}\n"
              f"Run --selftest to confirm the engine works, then check your connection.", file=sys.stderr)
        return
    if not meta:
        return
    print_board(rows, meta, args)
    if rows:
        c = write_csv(rows, args)
        h = write_html(rows, meta, args)
        print(f"Saved: {c}\n       {h}")


if __name__ == "__main__":
    main()

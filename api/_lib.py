"""
Shared logic for the FPL Companion API.
Imported by the other files in this directory via `from _lib import ...`.

No third-party HTTP client is used (urllib is stdlib); the only extra
dependencies are numpy/scipy for the MILP squad optimizer.
"""

import json
import math
import os
import urllib.error
import urllib.request

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}  # N/A for GKP

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-App-Password",
}


# ----------------------------------------------------------------------
# Fetching live FPL data
# ----------------------------------------------------------------------

def _http_get_json(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_fpl_data():
    """Return (bootstrap, fixtures) as parsed JSON."""
    headers = {"User-Agent": "Mozilla/5.0 (fpl-companion)"}
    bootstrap = _http_get_json(FPL_BOOTSTRAP_URL, headers=headers)
    fixtures = _http_get_json(FPL_FIXTURES_URL, headers=headers)
    return bootstrap, fixtures


# ----------------------------------------------------------------------
# Poisson helper (no scipy.stats needed)
# ----------------------------------------------------------------------

def _poisson_pmf(lam, k):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _poisson_sf_at_least(lam, threshold):
    """P(X >= threshold) for X ~ Poisson(lam)."""
    if threshold <= 0:
        return 1.0
    lam = max(lam, 0.0)
    cdf_below = sum(_poisson_pmf(lam, k) for k in range(threshold))
    return max(0.0, 1.0 - cdf_below)


# ----------------------------------------------------------------------
# XPModel
# ----------------------------------------------------------------------

class XPModel:
    def __init__(self, bootstrap, fixtures):
        self.bootstrap = bootstrap
        self.fixtures = fixtures
        self.scoring = bootstrap.get("game_config", {}).get("scoring", {})
        self.teams_by_id = {t["id"]: t for t in bootstrap.get("teams", [])}

        strengths = []
        for t in bootstrap.get("teams", []):
            for key in ("strength_overall_home", "strength_overall_away"):
                val = t.get(key)
                if val:
                    strengths.append(val)
        self.league_avg_strength = (sum(strengths) / len(strengths)) if strengths else 3.0

        # gameweek -> list of fixtures in that gameweek
        self._fixtures_by_gw = {}
        for f in fixtures:
            gw = f.get("event")
            if gw is None:
                continue
            self._fixtures_by_gw.setdefault(gw, []).append(f)

    # -- internal helpers -------------------------------------------------

    def _opponent_strength(self, opponent_team_id, opponent_is_home):
        team = self.teams_by_id.get(opponent_team_id, {})
        key = "strength_overall_home" if opponent_is_home else "strength_overall_away"
        return team.get(key) or self.league_avg_strength

    def _team_fixtures_in_gw(self, team_id, gw):
        """Return list of (fixture, is_home) for this team in this gameweek."""
        out = []
        for f in self._fixtures_by_gw.get(gw, []):
            if f.get("team_h") == team_id:
                out.append((f, True))
            elif f.get("team_a") == team_id:
                out.append((f, False))
        return out

    def _fixture_points(self, element, position, fixture_multiplier, scoring):
        """Sum of xP components a-h for a single fixture."""
        chance = element.get("chance_of_playing_next_round")

        # A player flagged injured/suspended/unavailable is ruled out even if
        # chance_of_playing_next_round hasn't caught up to 0 yet -- the status
        # field is the more reliable signal for "definitely not playing".
        if chance is None and element.get("status") in ("i", "s", "u"):
            chance = 0

        availability = (chance / 100.0) if chance is not None else 1.0

        starts_p90 = min(element.get("starts_per_90") or 0.0, 1.0)
        p_60 = availability * starts_p90
        p_any = availability * min(starts_p90 + 0.15, 1.0)

        if chance == 0:
            # Ruled out for the next round -- no fractional-minutes floor.
            # (Previously this only zeroed out when starts_per_90 was also 0,
            # so an injured regular starter still got a 5%-of-normal projection
            # instead of true zero, occasionally outscoring a healthy squad-filler.)
            minutes_fraction = 0.0
        else:
            minutes_fraction = max(p_any, 0.05)

        points = 0.0

        # a) Appearance
        points += p_60 * scoring.get("long_play", 0) + max(p_any - p_60, 0) * scoring.get("short_play", 0)

        # b) Goals / assists
        exp_goals = (element.get("expected_goals_per_90") or 0.0) * minutes_fraction * fixture_multiplier
        exp_assists = (element.get("expected_assists_per_90") or 0.0) * minutes_fraction * fixture_multiplier
        goals_scored_scoring = scoring.get("goals_scored", {}).get(position, 0)
        points += exp_goals * goals_scored_scoring + exp_assists * scoring.get("assists", 0)

        # c) Clean sheet
        p_cs = min((element.get("clean_sheets_per_90") or 0.0) * fixture_multiplier * availability, 0.9)
        points += p_cs * scoring.get("clean_sheets", {}).get(position, 0)

        # d) Defensive contribution
        threshold = DC_THRESHOLD.get(position)
        if threshold:
            lam = (element.get("defensive_contribution_per_90") or 0.0) * minutes_fraction
            p_reach = _poisson_sf_at_least(lam, threshold)
            points += p_reach * scoring.get("defensive_contribution", {}).get(position, 0)

        # e) Bonus
        season_minutes = element.get("minutes") or 0
        if season_minutes > 0:
            bonus_per90 = (element.get("bonus") or 0.0) / (season_minutes / 90.0)
            points += bonus_per90 * minutes_fraction * (0.5 + 0.5 * fixture_multiplier)

        # f) Saves (GKP only)
        if position == "GKP":
            exp_saves = (element.get("saves_per_90") or 0.0) * minutes_fraction / max(fixture_multiplier, 0.5)
            points += (exp_saves / 3.0) * scoring.get("saves", 0)

        # g) Goals conceded penalty (GKP/DEF only)
        if position in ("GKP", "DEF"):
            exp_gc = (element.get("goals_conceded_per_90") or 0.0) * minutes_fraction / fixture_multiplier
            points += (exp_gc / 2.0) * scoring.get("goals_conceded", {}).get(position, 0)

        # h) Cards
        minutes_per90_denom = max(season_minutes / 90.0, 0.1)
        yc_rate = (element.get("yellow_cards") or 0.0) / minutes_per90_denom
        rc_rate = (element.get("red_cards") or 0.0) / minutes_per90_denom
        points += (yc_rate * scoring.get("yellow_cards", 0) + rc_rate * scoring.get("red_cards", 0)) * minutes_fraction

        return points

    def _player_gw_xp(self, element, gw):
        """Return (xp, n_fixtures) for this player in this single gameweek."""
        position = POSITION_MAP.get(element.get("element_type"))
        if position is None:
            return 0.0, 0

        team_id = element.get("team")
        team_fixtures = self._team_fixtures_in_gw(team_id, gw)
        if not team_fixtures:
            return 0.0, 0

        total = 0.0
        for fixture, is_home in team_fixtures:
            opponent_id = fixture.get("team_a") if is_home else fixture.get("team_h")
            opponent_strength = self._opponent_strength(opponent_id, opponent_is_home=not is_home)
            fixture_multiplier = self.league_avg_strength / opponent_strength if opponent_strength else 1.0
            total += self._fixture_points(element, position, fixture_multiplier, self.scoring)

        return total, len(team_fixtures)

    # -- public API ---------------------------------------------------

    def project_gameweek(self, gw):
        results = []
        for element in self.bootstrap.get("elements", []):
            if not element.get("can_select", True):
                continue
            xp, n_fixtures = self._player_gw_xp(element, gw)
            position = POSITION_MAP.get(element.get("element_type"))
            team = self.teams_by_id.get(element.get("team"), {})
            results.append({
                "player_id": element.get("id"),
                "web_name": element.get("web_name"),
                "team_id": element.get("team"),
                "team_short": team.get("short_name"),
                "position": position,
                "cost": (element.get("now_cost") or 0) / 10.0,
                "xp": round(xp, 2),
                "fixtures_this_gw": n_fixtures,
            })
        return results

    def project_horizon(self, start_gw, n_gws, decay=0.9):
        by_id = {}
        for element in self.bootstrap.get("elements", []):
            if not element.get("can_select", True):
                continue
            position = POSITION_MAP.get(element.get("element_type"))
            team = self.teams_by_id.get(element.get("team"), {})
            by_id[element["id"]] = {
                "player_id": element.get("id"),
                "web_name": element.get("web_name"),
                "team_id": element.get("team"),
                "team_short": team.get("short_name"),
                "position": position,
                "cost": (element.get("now_cost") or 0) / 10.0,
                "xp": 0.0,
                "fixtures_this_gw": 0,
                "_element": element,
            }

        for i in range(n_gws):
            gw = start_gw + i
            weight = decay ** i
            for pid, row in by_id.items():
                xp, n_fixtures = self._player_gw_xp(row["_element"], gw)
                row["xp"] += xp * weight
                row["fixtures_this_gw"] += n_fixtures

        out = []
        for row in by_id.values():
            row = dict(row)
            row.pop("_element")
            row["xp"] = round(row["xp"], 2)
            out.append(row)
        return out


# ----------------------------------------------------------------------
# MILP squad optimizer
# ----------------------------------------------------------------------

REQUIRED_COUNTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
START_MIN = {"GKP": 1, "DEF": 3, "FWD": 1}
START_MAX = {"GKP": 1, "DEF": 5, "FWD": 3}


def _normalize_player_ids(players):
    """XPModel.project_gameweek/project_horizon key each player by
    'player_id'; the optimizers below key by 'id'. Normalize so both
    naming conventions work as input."""
    out = []
    for p in players:
        p = dict(p)
        if "id" not in p and "player_id" in p:
            p["id"] = p["player_id"]
        out.append(p)
    return out


def optimize_squad(players, budget=100.0, locked_ids=None, excluded_ids=None):
    """
    players: list of dicts with id (or player_id), position, team_id, cost, xp
    Returns dict: squad, starting_xi, bench, captain, vice_captain,
                  total_cost, total_xp
    """
    locked_ids = set(locked_ids or [])
    excluded_ids = set(excluded_ids or [])

    players = _normalize_player_ids(players)
    candidates = [p for p in players if p["id"] not in excluded_ids]
    n = len(candidates)
    if n == 0:
        raise ValueError("No candidate players available to optimize over")

    xp = np.array([c["xp"] for c in candidates], dtype=float)
    cost = np.array([c["cost"] for c in candidates], dtype=float)
    positions = [c["position"] for c in candidates]
    team_ids = [c["team_id"] for c in candidates]

    n_vars = 3 * n  # squad[0..n), start[n..2n), captain[2n..3n)

    def sq(i):
        return i

    def st(i):
        return n + i

    def cp(i):
        return 2 * n + i

    # BENCH_WEIGHT: the starting XI and captain are the only things that
    # score points, so they're weighted at full value. But with zero weight
    # on the other 4 squad slots, the solver is indifferent between a fit
    # player and an injured/suspended one sitting on the bench at the same
    # price -- nothing tells it a better bench is preferable. This adds a
    # small tie-breaking weight (kept well under any realistic starting-XI
    # xP gap) so it fills the bench with the best available players rather
    # than an arbitrary pick among equally-priced options.
    BENCH_WEIGHT = 0.001

    c_obj = np.zeros(n_vars)
    for i in range(n):
        c_obj[sq(i)] = -BENCH_WEIGHT * xp[i]
        c_obj[st(i)] = -xp[i]
        c_obj[cp(i)] = -xp[i]

    constraints = []

    def row(indices, coeffs, lb, ub, n_vars=n_vars):
        a = np.zeros(n_vars)
        for idx, coef in zip(indices, coeffs):
            a[idx] = coef
        return LinearConstraint(a, lb, ub)

    # squad size == 15
    a = np.zeros(n_vars)
    for i in range(n):
        a[sq(i)] = 1
    constraints.append(LinearConstraint(a, 15, 15))

    # squad composition per position
    for pos, count in REQUIRED_COUNTS.items():
        a = np.zeros(n_vars)
        for i in range(n):
            if positions[i] == pos:
                a[sq(i)] = 1
        constraints.append(LinearConstraint(a, count, count))

    # starting XI size == 11
    a = np.zeros(n_vars)
    for i in range(n):
        a[st(i)] = 1
    constraints.append(LinearConstraint(a, 11, 11))

    # start[i] <= squad[i]  ->  start[i] - squad[i] <= 0
    for i in range(n):
        a = np.zeros(n_vars)
        a[st(i)] = 1
        a[sq(i)] = -1
        constraints.append(LinearConstraint(a, -np.inf, 0))

    # captain[i] <= start[i]
    for i in range(n):
        a = np.zeros(n_vars)
        a[cp(i)] = 1
        a[st(i)] = -1
        constraints.append(LinearConstraint(a, -np.inf, 0))

    # exactly one captain
    a = np.zeros(n_vars)
    for i in range(n):
        a[cp(i)] = 1
    constraints.append(LinearConstraint(a, 1, 1))

    # starting position bounds
    for pos, lo in START_MIN.items():
        hi = START_MAX[pos]
        a = np.zeros(n_vars)
        for i in range(n):
            if positions[i] == pos:
                a[st(i)] = 1
        constraints.append(LinearConstraint(a, lo, hi))

    # club limit: at most 3 per team in the squad
    for team_id in set(team_ids):
        a = np.zeros(n_vars)
        for i in range(n):
            if team_ids[i] == team_id:
                a[sq(i)] = 1
        constraints.append(LinearConstraint(a, -np.inf, 3))

    # budget
    a = np.zeros(n_vars)
    for i in range(n):
        a[sq(i)] = cost[i]
    constraints.append(LinearConstraint(a, -np.inf, budget))

    # bounds (binary), locked players forced into squad
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    for i in range(n):
        if candidates[i]["id"] in locked_ids:
            lb[sq(i)] = 1

    bounds = Bounds(lb, ub)
    integrality = np.ones(n_vars)

    result = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds)

    if not result.success:
        raise ValueError(f"Squad optimization infeasible: {result.message}")

    x = result.x
    squad_idx = [i for i in range(n) if x[sq(i)] > 0.5]
    start_idx = [i for i in range(n) if x[st(i)] > 0.5]
    captain_idx = [i for i in range(n) if x[cp(i)] > 0.5]

    def to_player(i):
        p = dict(candidates[i])
        return p

    squad = [to_player(i) for i in squad_idx]
    starting_xi = [to_player(i) for i in start_idx]
    bench_idx = [i for i in squad_idx if i not in start_idx]
    bench = sorted([to_player(i) for i in bench_idx], key=lambda p: -p["xp"])

    captain = to_player(captain_idx[0]) if captain_idx else None

    vice_candidates = sorted(
        [to_player(i) for i in start_idx if not captain_idx or i != captain_idx[0]],
        key=lambda p: -p["xp"],
    )
    vice_captain = vice_candidates[0] if vice_candidates else None

    total_cost = sum(p["cost"] for p in squad)
    total_xp = sum(p["xp"] for p in starting_xi) + (captain["xp"] if captain else 0)

    return {
        "squad": squad,
        "starting_xi": starting_xi,
        "bench": bench,
        "captain": captain,
        "vice_captain": vice_captain,
        "total_cost": round(total_cost, 1),
        "total_xp": round(total_xp, 2),
    }


# ----------------------------------------------------------------------
# Transfer optimizer
# ----------------------------------------------------------------------

def optimize_transfers(players, current_squad_ids, bank=0.0, free_transfers=1, max_transfers=5):
    players = _normalize_player_ids(players)
    current_squad_ids = set(current_squad_ids)
    current_players = [p for p in players if p["id"] in current_squad_ids]
    current_cost = sum(p["cost"] for p in current_players)
    budget = current_cost + bank

    best = None
    best_net = -math.inf

    for k in range(0, max_transfers + 1):
        n = len(players)
        # Build a scoped optimization: add extra constraint capping the
        # number of squad players NOT in current_squad_ids to <= k.
        try:
            result = _optimize_squad_capped_new_players(
                players, budget=budget, current_squad_ids=current_squad_ids, max_new=k
            )
        except ValueError:
            continue

        hit = max(0, k - free_transfers) * 4
        net = result["total_xp"] - hit

        if net > best_net:
            best_net = net
            new_squad_ids = {p["id"] for p in result["squad"]}
            transfers_out = [p for p in current_players if p["id"] not in new_squad_ids]
            transfers_in = [p for p in result["squad"] if p["id"] not in current_squad_ids]
            best = {
                "new_squad": result,
                "transfers_out": transfers_out,
                "transfers_in": transfers_in,
                "n_transfers": k,
                "hit_cost": hit,
                "net_xp_gain": round(net, 2),
                "captain": result["captain"],
                "vice_captain": result["vice_captain"],
            }

    if best is None:
        raise ValueError("No feasible transfer plan found for any number of transfers")

    return best


def _optimize_squad_capped_new_players(players, budget, current_squad_ids, max_new):
    """Same MILP as optimize_squad, plus a constraint capping the number
    of squad players not already in current_squad_ids to <= max_new."""
    candidates = players
    n = len(candidates)
    if n == 0:
        raise ValueError("No candidate players available to optimize over")

    xp = np.array([c["xp"] for c in candidates], dtype=float)
    cost = np.array([c["cost"] for c in candidates], dtype=float)
    positions = [c["position"] for c in candidates]
    team_ids = [c["team_id"] for c in candidates]

    n_vars = 3 * n

    def sq(i):
        return i

    def st(i):
        return n + i

    def cp(i):
        return 2 * n + i

    # See BENCH_WEIGHT comment in optimize_squad -- same tie-break, so an
    # equally-priced fit player is preferred over an injured/suspended one
    # for the bench-only squad slots.
    BENCH_WEIGHT = 0.001

    c_obj = np.zeros(n_vars)
    for i in range(n):
        c_obj[sq(i)] = -BENCH_WEIGHT * xp[i]
        c_obj[st(i)] = -xp[i]
        c_obj[cp(i)] = -xp[i]

    constraints = []

    a = np.zeros(n_vars)
    for i in range(n):
        a[sq(i)] = 1
    constraints.append(LinearConstraint(a, 15, 15))

    for pos, count in REQUIRED_COUNTS.items():
        a = np.zeros(n_vars)
        for i in range(n):
            if positions[i] == pos:
                a[sq(i)] = 1
        constraints.append(LinearConstraint(a, count, count))

    a = np.zeros(n_vars)
    for i in range(n):
        a[st(i)] = 1
    constraints.append(LinearConstraint(a, 11, 11))

    for i in range(n):
        a = np.zeros(n_vars)
        a[st(i)] = 1
        a[sq(i)] = -1
        constraints.append(LinearConstraint(a, -np.inf, 0))

    for i in range(n):
        a = np.zeros(n_vars)
        a[cp(i)] = 1
        a[st(i)] = -1
        constraints.append(LinearConstraint(a, -np.inf, 0))

    a = np.zeros(n_vars)
    for i in range(n):
        a[cp(i)] = 1
    constraints.append(LinearConstraint(a, 1, 1))

    for pos, lo in START_MIN.items():
        hi = START_MAX[pos]
        a = np.zeros(n_vars)
        for i in range(n):
            if positions[i] == pos:
                a[st(i)] = 1
        constraints.append(LinearConstraint(a, lo, hi))

    for team_id in set(team_ids):
        a = np.zeros(n_vars)
        for i in range(n):
            if team_ids[i] == team_id:
                a[sq(i)] = 1
        constraints.append(LinearConstraint(a, -np.inf, 3))

    a = np.zeros(n_vars)
    for i in range(n):
        a[sq(i)] = cost[i]
    constraints.append(LinearConstraint(a, -np.inf, budget))

    # cap number of "new" (non-current-squad) players entering the squad
    a = np.zeros(n_vars)
    for i in range(n):
        if candidates[i]["id"] not in current_squad_ids:
            a[sq(i)] = 1
    constraints.append(LinearConstraint(a, -np.inf, max_new))

    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)

    bounds = Bounds(lb, ub)
    integrality = np.ones(n_vars)

    result = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds)

    if not result.success:
        raise ValueError(f"infeasible: {result.message}")

    x = result.x
    squad_idx = [i for i in range(n) if x[sq(i)] > 0.5]
    start_idx = [i for i in range(n) if x[st(i)] > 0.5]
    captain_idx = [i for i in range(n) if x[cp(i)] > 0.5]

    def to_player(i):
        return dict(candidates[i])

    squad = [to_player(i) for i in squad_idx]
    starting_xi = [to_player(i) for i in start_idx]
    bench_idx = [i for i in squad_idx if i not in start_idx]
    bench = sorted([to_player(i) for i in bench_idx], key=lambda p: -p["xp"])

    captain = to_player(captain_idx[0]) if captain_idx else None
    vice_candidates = sorted(
        [to_player(i) for i in start_idx if not captain_idx or i != captain_idx[0]],
        key=lambda p: -p["xp"],
    )
    vice_captain = vice_candidates[0] if vice_candidates else None

    total_cost = sum(p["cost"] for p in squad)
    total_xp = sum(p["xp"] for p in starting_xi) + (captain["xp"] if captain else 0)

    return {
        "squad": squad,
        "starting_xi": starting_xi,
        "bench": bench,
        "captain": captain,
        "vice_captain": vice_captain,
        "total_cost": round(total_cost, 1),
        "total_xp": round(total_xp, 2),
    }


# ----------------------------------------------------------------------
# Chip advisor (Wildcard / Free Hit / Bench Boost / Triple Captain)
# ----------------------------------------------------------------------
#
# 2026/27 rules (confirmed from premierleague.com, Aug 2026): managers get
# TWO complete sets of the four chips (Wildcard, Free Hit, Bench Boost,
# Triple Captain) = 8 chips total. The first set must be activated by the
# Gameweek 19 deadline (13:30 GMT, Sat 2 Jan 2027) or it's lost; the second
# set unlocks for Gameweeks 20-38. Only one chip can be played per
# gameweek. Free Hit cannot be played in Gameweek 1, and if the first Free
# Hit is used in GW19 the second cannot be used in GW20 (the two would
# effectively double up across the boundary).
#
# There's no "Assistant Manager" chip this season (it was introduced in
# 2024/25 and has since been dropped).
#
# Rather than hardcoding which gameweeks are blank/double -- those depend
# on live FA Cup/League Cup/European fixture congestion and aren't known
# this far ahead -- this advisor detects them from the same live fixtures
# feed the rest of the app already pulls (fixtures_this_gw per player) and
# scores each chip/gameweek combination on actual projected xP, not a
# blog's guess at the calendar.

CHIP_TYPES = ["wildcard", "free_hit", "bench_boost", "triple_captain"]
FIRST_HALF_LAST_GW = 19   # first chip set must be used by this gameweek
SECOND_HALF_FIRST_GW = 20  # second chip set becomes available here


def chip_half_for_gw(gw):
    return "first" if gw <= FIRST_HALF_LAST_GW else "second"


def best_xi_from_squad(pool, squad_ids):
    """Given a player pool (with xp for a specific gameweek/horizon) and a
    FIXED 15-man squad, pick the best starting XI/captain/vice from it --
    no transfers. Reuses the transfer-optimizer MILP with max_new=0 and an
    unbounded budget (the squad's cost is already fixed, so it can't bind)."""
    pool = _normalize_player_ids(pool)
    squad_ids = set(squad_ids)
    return _optimize_squad_capped_new_players(
        pool, budget=float("inf"), current_squad_ids=squad_ids, max_new=0
    )


def _fixture_signal(pool, squad_ids):
    """Return (blanks, doubles) counts for a squad in a single gameweek's pool,
    and (league_blanks, league_doubles) across every selectable player."""
    squad_ids = set(squad_ids)
    pool = _normalize_player_ids(pool)
    squad_rows = [p for p in pool if p["id"] in squad_ids]
    blanks = sum(1 for p in squad_rows if p["fixtures_this_gw"] == 0)
    doubles = sum(1 for p in squad_rows if p["fixtures_this_gw"] >= 2)
    league_blanks = sum(1 for p in pool if p["fixtures_this_gw"] == 0)
    league_doubles = sum(1 for p in pool if p["fixtures_this_gw"] >= 2)
    return {
        "squad_blanks": blanks,
        "squad_doubles": doubles,
        "league_blanks": league_blanks,
        "league_doubles": league_doubles,
    }


def analyze_chip_windows(model, current_squad_ids, bank=0.0, start_gw=1, n_gws=8,
                          decay=0.9, chips_used=None):
    """
    Score each of the next n_gws gameweeks (starting at start_gw) for each
    chip, using the model's live projections. Returns a per-gameweek table
    plus a headline recommendation per chip type.

    chips_used: list of chip-type strings already burned in the CURRENT
    half (first half = GW<=19, second half = GW>=20). start_gw determines
    which half this analysis covers; if the window crosses the GW19/20
    boundary it's capped to the half start_gw falls in, since chip sets
    don't carry over.
    """
    chips_used = set(chips_used or [])
    half = chip_half_for_gw(start_gw)
    half_last_gw = FIRST_HALF_LAST_GW if half == "first" else start_gw + n_gws - 1
    end_gw = min(start_gw + n_gws - 1, half_last_gw)

    current_squad_ids = list(current_squad_ids)
    if not current_squad_ids:
        raise ValueError("current_squad_ids is required to analyze chip windows")

    # First, work out a sensible budget ceiling for "fresh squad" scenarios
    # (Wildcard / Free Hit): current squad's cost (at today's prices) + bank.
    gw0_pool = model.project_gameweek(start_gw)
    gw0_pool = _normalize_player_ids(gw0_pool)
    current_cost = sum(p["cost"] for p in gw0_pool if p["id"] in current_squad_ids)
    budget = current_cost + bank

    rows = []
    for gw in range(start_gw, end_gw + 1):
        pool_gw = model.project_gameweek(gw)
        signal = _fixture_signal(pool_gw, current_squad_ids)

        baseline = best_xi_from_squad(pool_gw, current_squad_ids)
        bench_boost_value = round(sum(p["xp"] for p in baseline["bench"]), 2)
        triple_captain_value = round(baseline["captain"]["xp"], 2) if baseline["captain"] else 0.0

        try:
            free_hit_squad = optimize_squad(pool_gw, budget=budget)
            free_hit_gain = round(free_hit_squad["total_xp"] - baseline["total_xp"], 2)
        except ValueError:
            free_hit_squad = None
            free_hit_gain = 0.0

        remaining = end_gw - gw + 1
        horizon_players = model.project_horizon(gw, remaining, decay=decay)
        wildcard_baseline = best_xi_from_squad(horizon_players, current_squad_ids)
        try:
            wildcard_squad = optimize_squad(horizon_players, budget=budget)
            wildcard_gain = round(wildcard_squad["total_xp"] - wildcard_baseline["total_xp"], 2)
        except ValueError:
            wildcard_squad = None
            wildcard_gain = 0.0

        # Cumulative gain grows just by covering more weeks, which would bias
        # every recommendation toward "play it in the earliest gameweek
        # available". Rank by the per-gameweek average instead so windows of
        # different lengths are compared fairly; still report the cumulative
        # figure since that's the number that actually lands in the bank.
        wildcard_gain_per_gw = round(wildcard_gain / remaining, 2) if remaining else 0.0

        rows.append({
            "gameweek": gw,
            "squad_blanks": signal["squad_blanks"],
            "squad_doubles": signal["squad_doubles"],
            "league_blanks": signal["league_blanks"],
            "league_doubles": signal["league_doubles"],
            "baseline_xp": baseline["total_xp"],
            "bench_boost_value": bench_boost_value,
            "triple_captain_value": triple_captain_value,
            "free_hit_gain": free_hit_gain,
            "wildcard_gain_remaining_horizon": wildcard_gain,
            "wildcard_gain_per_gw": wildcard_gain_per_gw,
        })

    def eligible(chip):
        return chip not in chips_used

    def best_row(key, chip_name, report_key=None, block_gw1=False):
        report_key = report_key or key
        candidates = [r for r in rows if not (block_gw1 and r["gameweek"] == 1)]
        if not candidates or not eligible(chip_name):
            return None
        best = max(candidates, key=lambda r: r[key])
        return {"gameweek": best["gameweek"], "projected_gain": best[report_key]}

    recommendations = {
        # Free Hit can't be played in Gameweek 1, so it's excluded from that
        # candidate window (block_gw1=True).
        "wildcard": best_row("wildcard_gain_per_gw", "wildcard", report_key="wildcard_gain_remaining_horizon"),
        "free_hit": best_row("free_hit_gain", "free_hit", block_gw1=True),
        "bench_boost": best_row("bench_boost_value", "bench_boost"),
        "triple_captain": best_row("triple_captain_value", "triple_captain"),
    }

    # A rough headline: among eligible chips, which single chip/gameweek
    # pairing has the highest projected gain, using a common-ish threshold
    # (~4 points, roughly one transfer hit) below which "hold" is better
    # than burning a chip on a marginal week.
    headline = None
    HOLD_THRESHOLD = 4.0
    scored = [(name, rec) for name, rec in recommendations.items() if rec]
    scored = [(name, rec) for name, rec in scored if rec["projected_gain"] >= HOLD_THRESHOLD]
    if scored:
        name, rec = max(scored, key=lambda item: item[1]["projected_gain"])
        headline = {
            "chip": name,
            "gameweek": rec["gameweek"],
            "projected_gain": rec["projected_gain"],
        }

    return {
        "half": half,
        "window": {"start_gw": start_gw, "end_gw": end_gw},
        "gameweeks": rows,
        "recommendations": recommendations,
        "headline": headline,
    }


# ----------------------------------------------------------------------
# Supabase persistence (REST API via urllib, service-role key)
# ----------------------------------------------------------------------

def _supabase_config():
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    return url, key


def get_squad():
    url, key = _supabase_config()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY not configured")

    endpoint = f"{url}/rest/v1/app_state?id=eq.1&select=*"
    req = urllib.request.Request(endpoint, headers={
        "apikey": key,
        "Authorization": f"Bearer {key}",
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        rows = json.loads(resp.read().decode("utf-8"))
    return rows[0] if rows else None


def save_squad(payload):
    url, key = _supabase_config()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY not configured")

    body = dict(payload)
    body["id"] = 1

    endpoint = f"{url}/rest/v1/app_state?on_conflict=id"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=representation",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        rows = json.loads(resp.read().decode("utf-8"))
    return rows[0] if rows else body


# ----------------------------------------------------------------------
# Small helper for endpoint handlers
# ----------------------------------------------------------------------

def read_json_body(handler):
    length = int(handler.headers.get("Content-Length", 0) or 0)
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def send_json(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    for k, v in CORS_HEADERS.items():
        handler.send_header(k, v)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def send_cors_preflight(handler):
    handler.send_response(204)
    for k, v in CORS_HEADERS.items():
        handler.send_header(k, v)
    handler.send_header("Content-Length", "0")
    handler.end_headers()

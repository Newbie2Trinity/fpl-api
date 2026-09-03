"""
Shared logic for the FPL Companion API.
Imported by the other files in this directory via `from _lib import ...`.

No third-party HTTP client is used (urllib is stdlib); the only extra
dependencies are numpy/scipy for the MILP squad optimizer.
"""

import json
import math
import os
import unicodedata
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{}/"
FPL_ENTRY_PICKS_URL = "https://fantasy.premierleague.com/api/entry/{}/event/{}/picks/"

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}  # N/A for GKP

# Best-effort recovery-time estimates (in gameweeks) used to ramp an
# injured/suspended player's projection back up across a multi-gameweek
# horizon -- see XPModel._estimated_weeks_out(). FPL's API gives no return
# date, so this is a coarse keyword read of the free-text `news` field, not
# a medical prediction. Kept deliberately conservative (short) as the
# fallback so an unclassified knock doesn't get treated as a long absence.
LONG_TERM_NEWS_KEYWORDS = (
    "cruciate", "acl", "achilles", "fracture", "broken", "ruptured",
    "rupture", "long-term", "long term", "surgery", "months", "ineligible",
)
SHORT_TERM_NEWS_KEYWORDS = (
    "knock", "illness", "virus", "flu", "toe", "rest", "precaution", "minor",
)
ROTATION_RISK_STARTS_P90 = 0.5  # below this, flag as a fringe starter

# starts_per_90 is a season-to-date rate: it's 0.0 for EVERY player before
# a ball's been kicked (or briefly, after a long injury lay-off), which
# means a third-choice keeper looks statistically identical to the club's
# undisputed #1 -- both show 0 starts. Two fallback signals fill that gap,
# preferred in this order (see XPModel._preseason_prior()):
#   1. Last season's actual game-time % (player_history_cache, populated by
#      refresh_history_cache() below) -- real playing-time evidence, when
#      the player has Premier League history to draw on.
#   2. selected_by_percent (ownership) relative to the most-owned player at
#      the same club/position -- the market's live read on who's likely to
#      start, for anyone with no last-season history (new signings,
#      promoted-club debutants).
PRESEASON_NAILED_STARTS_P90 = 0.85  # assumed starts_per_90 for a club's clear #1 at a position, pre-data
TRUST_GAMES = 3  # real gameweeks played league-wide before starts_per_90 fully overrides both priors above
MIN_PEER_OWNERSHIP = 2.0  # below this, nobody at the club/position is meaningfully owned -- treat as no signal
LAST_SEASON_MINUTES_DENOM = 3420  # 38 games x 90 mins -- last_season_game_time_pct's denominator
REFRESH_CONCURRENCY = 30  # concurrent element-summary fetches in refresh_history_cache()
ELEMENT_SUMMARY_TIMEOUT = 8  # seconds -- keep short so one slow player can't stall the whole refresh

# The three clubs promoted into this season's Premier League have no FPL
# history at all -- player_history_cache above only covers players who had
# top-flight minutes last season, so every promoted-club player would
# otherwise drop straight to the ownership-based guess (_nailed_prior),
# which is a weak signal for a club nobody's picked yet. This is a third,
# middle tier: last season's Championship game-time %, hand-compiled from
# Wikipedia's 2025-26 season pages (no free source gave scrapable per-player
# data -- FBref/ESPN/WhoScored/FotMob are all JS-rendered with nothing in
# the static HTML, FBref blocks automated fetches outright, and Coventry's
# own Wikipedia page has no player-appearances table at all, unlike Hull's
# and Ipswich's). Values are starts+subs converted to an estimated % of a
# 46-game Championship season's available minutes (start = 90 mins, sub
# appearance = 25 mins, /4140), not exact minutes -- good enough to tell a
# nailed starter from a fringe player, which is all this tier is for.
# Keyed by this season's FPL team id -> normalized surname -> 0-1 pct.
# Coventry (team 7) has no entries -- see above -- so its players fall
# through to _nailed_prior same as before this feature existed.
CHAMPIONSHIP_2025_26_GAME_TIME = {
    7: {},  # Coventry City -- no scrapable source found, see comment above
    11: {  # Hull City
        "pandur": 0.74, "phillips": 0.0,
        "coyle": 0.60, "giles": 0.59, "hughes": 0.66, "ajayi": 0.21,
        "egan": 0.64, "jacob": 0.05, "drameh": 0.23, "famewo": 0.19,
        "mcnair": 0.06,
        "lundstram": 0.28, "hadziahmetovic": 0.52, "gyabi": 0.21,
        "crooks": 0.41, "slater": 0.66, "collyer": 0.01, "palmer": 0.10,
        "millar": 0.30, "mcburnie": 0.48, "belloumi": 0.15, "akintola": 0.20,
        "gelhardt": 0.54, "joseph": 0.57, "dowell": 0.06, "koumas": 0.05,
        "destan": 0.17,
    },
    12: {  # Ipswich Town
        "walton": 0.77, "palmer": 0.22,
        "oshea": 0.98, "davis": 0.75, "greaves": 0.48, "furlong": 0.85,
        "kipre": 0.65, "johnson": 0.24, "young": 0.13,
        "matusiwa": 0.96, "taylor": 0.59, "cajuste": 0.39, "burns": 0.23,
        "neil": 0.22, "nunez": 0.57,
        "clarke": 0.63, "hirst": 0.61, "philogene": 0.55, "azon": 0.54,
        "mcateer": 0.38, "egeli": 0.45, "akpom": 0.29,
    },
}


def _normalize_surname(name):
    """Lowercase, accent/punctuation-stripped surname for matching against
    the hand-compiled CHAMPIONSHIP_2025_26_GAME_TIME table above -- FPL's
    API and Wikipedia don't always agree on exact accent characters or
    apostrophes (e.g. Nunez vs Núñez, O'Shea vs OShea), so comparing on
    letters-only ASCII avoids silent misses."""
    stripped = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in stripped.lower() if ch.isalpha() or ch == " ").strip()


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-App-Password",
}


# ----------------------------------------------------------------------
# Fetching live FPL data
# ----------------------------------------------------------------------

def _http_get_json(url, headers=None, timeout=25):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_fpl_data():
    """Return (bootstrap, fixtures) as parsed JSON."""
    headers = {"User-Agent": "Mozilla/5.0 (fpl-companion)"}
    bootstrap = _http_get_json(FPL_BOOTSTRAP_URL, headers=headers)
    fixtures = _http_get_json(FPL_FIXTURES_URL, headers=headers)
    return bootstrap, fixtures


def current_gameweek(bootstrap):
    """Best-guess 'right now' gameweek from bootstrap's events list: the
    live/just-finished one if there is one, else the next upcoming deadline.
    Used by import_team.py as the default gameweek to pull picks for when
    the caller doesn't specify one."""
    events = bootstrap.get("events", [])
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        return current.get("id")
    nxt = next((e for e in events if e.get("is_next")), None)
    return nxt.get("id") if nxt else None


def fetch_entry_picks(entry_id, gameweek):
    """One FPL manager's squad picks for one gameweek, from FPL's public
    entry endpoint -- this is the same data anyone can see on that entry's
    public 'Points' page, no login needed. Used by import_team.py to pull a
    real, currently-live squad in as this app's saved squad, since until now
    the only way to populate a saved squad here was to run it through this
    app's own optimizer first."""
    headers = {"User-Agent": "Mozilla/5.0 (fpl-companion)"}
    url = FPL_ENTRY_PICKS_URL.format(entry_id, gameweek)
    try:
        return _http_get_json(url, headers=headers, timeout=15)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(
                f"No picks found for FPL team {entry_id}, gameweek {gameweek} -- "
                "double check the team ID and that this gameweek's picks are set"
            )
        raise


def fetch_player_last_season_minutes(player_id):
    """Most recent past-season's total minutes for one player, from FPL's
    per-player summary endpoint, or None if they have no Premier League/FPL
    history at all (new signing, promoted-club debutant, first season)."""
    headers = {"User-Agent": "Mozilla/5.0 (fpl-companion)"}
    data = _http_get_json(
        FPL_ELEMENT_SUMMARY_URL.format(player_id), headers=headers, timeout=ELEMENT_SUMMARY_TIMEOUT
    )
    history_past = data.get("history_past") or []
    if not history_past:
        return None
    return history_past[-1].get("minutes")


def refresh_history_cache(bootstrap):
    """Populate player_history_cache with every selectable player's last-
    season game-time %, used as the model's preseason 'will they actually
    start' prior (see PRESEASON_NAILED_STARTS_P90 above). FPL has no bulk
    endpoint for this -- it's one HTTP call per player (element-summary),
    so this fetches them concurrently rather than in a loop; ~650 players
    serially would be far too slow for a single request. Meant to be called
    occasionally (e.g. a weekly cron), not on every page load -- the
    ordinary API endpoints just read whatever's already cached via
    get_history_cache().
    """
    player_ids = [e["id"] for e in bootstrap.get("elements", []) if e.get("can_select", True)]

    def fetch_one(player_id):
        try:
            minutes = fetch_player_last_season_minutes(player_id)
        except Exception:
            # One player's summary call failing (timeout, transient FPL
            # error) shouldn't sink the whole refresh -- they just don't
            # get cached this round and fall back to the ownership prior.
            minutes = None
        return player_id, minutes

    rows = []
    now = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=REFRESH_CONCURRENCY) as pool:
        futures = [pool.submit(fetch_one, pid) for pid in player_ids]
        for future in as_completed(futures):
            player_id, minutes = future.result()
            if minutes is None:
                continue
            pct = min(minutes / LAST_SEASON_MINUTES_DENOM, 1.0)
            rows.append({
                "player_id": player_id,
                "last_season_minutes": minutes,
                "last_season_game_time_pct": round(pct, 4),
                "updated_at": now,
            })

    save_history_cache_rows(rows)
    return {"players_checked": len(player_ids), "players_cached": len(rows)}


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
    def __init__(self, bootstrap, fixtures, history_cache=None):
        self.bootstrap = bootstrap
        self.fixtures = fixtures
        self.scoring = bootstrap.get("game_config", {}).get("scoring", {})
        self.teams_by_id = {t["id"]: t for t in bootstrap.get("teams", [])}
        # {player_id: last_season_game_time_pct}, from get_history_cache() --
        # see _preseason_prior(). Optional: an empty/missing cache just means
        # every player falls back to the ownership-based prior.
        self._history_cache = history_cache or {}

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

        # (team_id, position) -> highest selected_by_percent among that
        # club's selectable players at that position. Used by _nailed_prior()
        # below to guess who's actually first-choice before real playing-time
        # data exists to prove it -- see the big comment there.
        self._max_ownership_by_team_pos = {}
        for element in bootstrap.get("elements", []):
            if not element.get("can_select", True):
                continue
            position = POSITION_MAP.get(element.get("element_type"))
            if position is None:
                continue
            ownership = _safe_float(element.get("selected_by_percent"))
            key = (element.get("team"), position)
            if ownership > self._max_ownership_by_team_pos.get(key, 0.0):
                self._max_ownership_by_team_pos[key] = ownership

        # How much to trust real minutes/starts data over the ownership-based
        # prior, based on how many real Premier League gameweeks have been
        # played SO FAR THIS SEASON -- not this player's own minutes. Using
        # the player's own minutes as the trust signal (the previous version
        # of this) can never recover for a player who's genuinely never
        # picked: their own minutes stay 0 forever, so trust never rises and
        # the model would lean on the ownership prior indefinitely even once
        # real matches have conclusively shown they don't play. Basing trust
        # on games played league-wide fixes that: after TRUST_GAMES real
        # gameweeks, a 0-minute "available" player's own starts_per_90 (also
        # 0) is fully trusted on its own -- actual pitch time, not ownership.
        games_played = sum(1 for e in bootstrap.get("events", []) if e.get("finished"))
        self.data_trust = min(games_played / TRUST_GAMES, 1.0)

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

    def _nailed_prior(self, element):
        """Rough 0-1 read on 'is this player actually first-choice at their
        position for their club', from selected_by_percent relative to the
        most-owned player in the same (team, position) group. Only meant to
        fill the gap starts_per_90 leaves at the start of a season (or
        after a long injury) -- see the module-level comment above
        PRESEASON_NAILED_STARTS_P90."""
        position = POSITION_MAP.get(element.get("element_type"))
        ownership = _safe_float(element.get("selected_by_percent"))
        peer_max = self._max_ownership_by_team_pos.get((element.get("team"), position), 0.0)
        if peer_max < MIN_PEER_OWNERSHIP:
            # Nobody at this club/position is meaningfully owned yet (e.g. a
            # promoted side nobody's looked at) -- no real signal either way,
            # so don't suppress anyone off the back of noise.
            return 1.0
        return min(ownership / peer_max, 1.0)

    def _promoted_club_prior(self, element):
        """Second tier: last season's Championship game-time % for the
        three promoted clubs (CHAMPIONSHIP_2025_26_GAME_TIME), for players
        who have no FPL/top-flight history but DO have real evidence from
        the division below -- still real playing-time evidence, just not
        from the Premier League. Returns None (not 0) when there's no entry,
        so callers know to fall through to the ownership-based guess rather
        than treating "no data" as "never plays"."""
        club_table = CHAMPIONSHIP_2025_26_GAME_TIME.get(element.get("team"))
        if not club_table:
            return None
        surname = _normalize_surname(element.get("second_name"))
        return club_table.get(surname)

    def _preseason_prior(self, element):
        """Best available 0-1 read on 'will this player actually start',
        for filling the gap starts_per_90 leaves before real current-season
        data exists. Preferred in order: (1) last season's actual top-flight
        game-time % when cached for this player, (2) last season's
        Championship game-time % for promoted-club players
        (_promoted_club_prior), (3) the ownership-based comparison
        (_nailed_prior) for anyone with no real playing-time history at all
        (new signings, and Coventry -- see the comment on
        CHAMPIONSHIP_2025_26_GAME_TIME)."""
        cached_pct = self._history_cache.get(element.get("id"))
        if cached_pct is not None:
            return cached_pct
        promoted_pct = self._promoted_club_prior(element)
        if promoted_pct is not None:
            return promoted_pct
        return self._nailed_prior(element)

    def _fixture_points(self, element, position, fixture_multiplier, scoring, availability):
        """Sum of xP components a-h for a single fixture. `availability` is
        this player's probability-of-playing multiplier for the gameweek in
        question -- see _availability() for how it's derived (including the
        horizon recovery ramp for players currently ruled out)."""
        starts_p90_real = min(element.get("starts_per_90") or 0.0, 1.0)
        # Blend real season starts data with the ownership-based prior,
        # trusting real data more as actual gameweeks are played (see
        # self.data_trust in __init__) -- so this only matters early in the
        # season and quietly gets out of the way once starts_per_90 reflects
        # real playing time.
        prior_starts_p90 = self._preseason_prior(element) * PRESEASON_NAILED_STARTS_P90
        starts_p90 = self.data_trust * starts_p90_real + (1 - self.data_trust) * prior_starts_p90

        p_60 = availability * starts_p90
        p_any = availability * min(starts_p90 + 0.15, 1.0)

        if availability <= 0:
            # Ruled out -- no fractional-minutes floor.
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

    def _estimated_weeks_out(self, element):
        """Coarse estimate of how many gameweeks an unavailable player is
        likely to miss, used only to shape the horizon recovery ramp in
        _availability(). Not a medical prediction -- see the module-level
        keyword lists for the reasoning."""
        status = element.get("status")
        if status == "u":
            return 99  # left the club / deregistered -- not returning this season
        if status == "s":
            return 2  # most single/short bans clear inside a couple of gameweeks

        news = (element.get("news") or "").lower()
        if any(k in news for k in LONG_TERM_NEWS_KEYWORDS):
            return 10
        if any(k in news for k in SHORT_TERM_NEWS_KEYWORDS):
            return 2
        if news:
            return 4  # named injury, severity unclear from the text -- medium default
        return 2  # ruled out with no news text at all -- assume short

    def _availability(self, element, gw_offset=0):
        """Return this player's probability-of-playing multiplier for a
        gameweek `gw_offset` steps beyond the next one (0 = the very next
        gameweek, which is what chance_of_playing_next_round describes).

        Beyond gw_offset 0 there's no live per-gameweek signal from the FPL
        API. Previously project_horizon reused this same number for every
        future gameweek, so a 2-week knock and a season-ending injury looked
        identical 8 gameweeks into a Wildcard scan. Instead, once a player is
        fully ruled out at gw_offset 0, ramp their availability back toward
        fit over an estimated recovery window rather than holding them at
        zero for the whole horizon.
        """
        chance = element.get("chance_of_playing_next_round")

        # A player flagged injured/suspended/unavailable is ruled out even if
        # chance_of_playing_next_round hasn't caught up to 0 yet -- the status
        # field is the more reliable signal for "definitely not playing", and
        # it can lag or sit on a stale nonzero number, not just null. Status
        # wins outright here, whatever chance currently says.
        if element.get("status") in ("i", "s", "u"):
            chance = 0

        availability = (chance / 100.0) if chance is not None else 1.0

        if gw_offset <= 0 or availability > 0:
            return availability

        weeks_out = self._estimated_weeks_out(element)
        if weeks_out >= 99:
            return 0.0
        return min(1.0, gw_offset / weeks_out)

    def _risk_note(self, element):
        """Short, human-readable explanation for why a player's projection
        is suppressed or shaky, so the UI can show *why* -- not just a
        lower number -- for injuries, suspensions, doubts, and players who
        simply aren't a nailed starter (season-long starts_per_90 below
        ROTATION_RISK_STARTS_P90). Returns None when there's nothing to flag."""
        status = element.get("status")
        chance = element.get("chance_of_playing_next_round")
        news = (element.get("news") or "").strip()

        if status == "u":
            return "unavailable"
        if status == "s":
            return "suspended" + (f" ({news})" if news else "")
        if status == "i":
            weeks = self._estimated_weeks_out(element)
            eta = "~1 more GW" if weeks <= 2 else f"~{weeks} more GWs (estimate)"
            return f"injured, {eta}" + (f" — {news}" if news else "")
        if chance is not None and chance < 100:
            return f"{chance}% chance of playing" + (f" — {news}" if news else "")

        starts_p90_real = element.get("starts_per_90") or 0.0
        minutes = element.get("minutes") or 0

        if self.data_trust < 1.0:
            # Not enough real gameweeks played yet to trust starts_per_90 on
            # its own -- fall back to last season's game-time % (or, absent
            # that, ownership), but real evidence (even partial) that this
            # player DOES start beats a merely-low prior, so a breakout
            # player the data hasn't caught up to yet doesn't get wrongly
            # flagged.
            nailed = self._preseason_prior(element)
            if nailed < ROTATION_RISK_STARTS_P90 and starts_p90_real < ROTATION_RISK_STARTS_P90:
                if minutes == 0:
                    return "hasn't played for their club yet this season -- projected as a likely non-starter"
                return "limited minutes for their club so far this season -- projected as a likely non-starter"
        elif minutes == 0:
            return "hasn't featured for their club this season"
        elif 0 < starts_p90_real < ROTATION_RISK_STARTS_P90:
            return "rotation risk (fringe starter)"

        return None

    def _player_gw_xp(self, element, gw, gw_offset=0):
        """Return (xp, n_fixtures) for this player in this single gameweek.
        gw_offset is how many gameweeks beyond the next one this call is
        for (0 for a direct single-gameweek projection); it only affects
        the injury/suspension recovery ramp in _availability()."""
        position = POSITION_MAP.get(element.get("element_type"))
        if position is None:
            return 0.0, 0

        team_id = element.get("team")
        team_fixtures = self._team_fixtures_in_gw(team_id, gw)
        if not team_fixtures:
            return 0.0, 0

        availability = self._availability(element, gw_offset)

        total = 0.0
        for fixture, is_home in team_fixtures:
            opponent_id = fixture.get("team_a") if is_home else fixture.get("team_h")
            opponent_strength = self._opponent_strength(opponent_id, opponent_is_home=not is_home)
            fixture_multiplier = self.league_avg_strength / opponent_strength if opponent_strength else 1.0
            total += self._fixture_points(element, position, fixture_multiplier, self.scoring, availability)

        return total, len(team_fixtures)

    # -- public API ---------------------------------------------------

    def project_gameweek(self, gw):
        results = []
        for element in self.bootstrap.get("elements", []):
            if not element.get("can_select", True):
                continue
            xp, n_fixtures = self._player_gw_xp(element, gw, gw_offset=0)
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
                "risk_note": self._risk_note(element),
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
                xp, n_fixtures = self._player_gw_xp(row["_element"], gw, gw_offset=i)
                row["xp"] += xp * weight
                row["fixtures_this_gw"] += n_fixtures

        out = []
        for row in by_id.values():
            row = dict(row)
            element = row.pop("_element")
            row["xp"] = round(row["xp"], 2)
            row["risk_note"] = self._risk_note(element)
            out.append(row)
        return out


# ----------------------------------------------------------------------
# MILP squad optimizer
# ----------------------------------------------------------------------

REQUIRED_COUNTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
START_MIN = {"GKP": 1, "DEF": 3, "FWD": 1}
START_MAX = {"GKP": 1, "DEF": 5, "FWD": 3}

# The starting XI and captain are the only things that score points, so
# they're weighted at full value in every MILP below. But with zero weight
# on the other 4 squad-only (bench) slots, the solver is indifferent
# between a fit player and an injured/suspended one at the same price --
# nothing tells it a better bench is preferable, and nothing tells the
# transfer optimizer that swapping a bench player is worth anything either.
# This tie-break weight is kept well under any realistic starting-XI xP
# gap so it only ever settles ties, never overrides a real selection call.
BENCH_WEIGHT = 0.001


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

        # total_xp only counts the starting XI + captain, so a transfer that
        # purely swaps out a bench player (e.g. replacing an injured 0-xP
        # squad filler with a fit one) always shows net == the k=0 baseline's
        # net exactly -- there was nothing here to ever recommend that swap.
        # Add the same small bench tie-break used when building the squad so
        # a real, if minor, bench upgrade can actually surface as a pick.
        bench_xp = sum(p["xp"] for p in result["squad"]) - sum(p["xp"] for p in result["starting_xi"])
        compare_net = net + BENCH_WEIGHT * bench_xp

        if compare_net > best_net:
            best_net = compare_net
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


def get_history_cache():
    """Return {player_id: last_season_game_time_pct} from the
    player_history_cache table (see refresh_history_cache()). Treated as
    optional everywhere it's used: if Supabase isn't configured, the table
    doesn't exist yet, or the request fails, this returns {} rather than
    raising -- an ordinary Optimize/Transfer request shouldn't break just
    because this enhancement isn't set up or Supabase hiccups."""
    url, key = _supabase_config()
    if not url or not key:
        return {}

    endpoint = f"{url}/rest/v1/player_history_cache?select=player_id,last_season_game_time_pct"
    req = urllib.request.Request(endpoint, headers={
        "apikey": key,
        "Authorization": f"Bearer {key}",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}
    return {r["player_id"]: r["last_season_game_time_pct"] for r in rows}


def save_history_cache_rows(rows):
    """Bulk upsert into player_history_cache -- one request for the whole
    batch (PostgREST supports an array body), not one write per player."""
    url, key = _supabase_config()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY not configured")
    if not rows:
        return

    endpoint = f"{url}/rest/v1/player_history_cache?on_conflict=player_id"
    data = json.dumps(rows).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        method="POST",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        resp.read()


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

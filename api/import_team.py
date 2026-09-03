import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from _lib import (
    fetch_fpl_data,
    fetch_entry_picks,
    current_gameweek,
    get_squad,
    save_squad,
    read_json_body,
    send_json,
    send_cors_preflight,
)


class handler(BaseHTTPRequestHandler):
    """Pull your actual live squad from the official FPL app/site in as this
    app's saved squad, instead of only being able to save a squad this app
    itself built. Just needs your public FPL team ID -- no login. Free
    transfers isn't something FPL's public API exposes anywhere, so that
    stays a manual field (pre-filled from whatever's already saved); bank
    and the 15 picks come straight from FPL. Captaincy is picked up too;
    chips_used is left untouched -- it's this app's own toggle state for
    the two-chip-set rule, not something FPL's API can tell us."""

    def do_OPTIONS(self):
        send_cors_preflight(self)

    def do_POST(self):
        try:
            required_password = os.environ.get("APP_PASSWORD")
            if required_password:
                supplied = self.headers.get("X-App-Password")
                if supplied != required_password:
                    send_json(self, 401, {"error": "Invalid or missing X-App-Password"})
                    return

            body = read_json_body(self)
            entry_id = body.get("entry_id")
            if not entry_id:
                send_json(self, 400, {
                    "error": "entry_id is required -- your FPL team ID, from the URL when "
                             "you view your team on the official site "
                             "(fantasy.premierleague.com/entry/<ID>/...)"
                })
                return

            existing = get_squad() or {}

            bootstrap, _ = fetch_fpl_data()
            gameweek = body.get("gameweek") or current_gameweek(bootstrap)
            if not gameweek:
                send_json(self, 500, {
                    "error": "Couldn't work out the current gameweek from FPL's data -- pass "
                             "gameweek explicitly"
                })
                return

            try:
                picks_data = fetch_entry_picks(entry_id, gameweek)
            except ValueError as e:
                send_json(self, 404, {"error": str(e)})
                return

            picks = picks_data.get("picks") or []
            squad_ids = [p.get("element") for p in picks]
            if len(squad_ids) != 15:
                send_json(self, 502, {
                    "error": f"FPL returned {len(squad_ids)} players for team {entry_id}, "
                             f"gameweek {gameweek} instead of 15 -- check the team ID"
                })
                return

            captain_id = next((p.get("element") for p in picks if p.get("is_captain")), None)
            bank_raw = (picks_data.get("entry_history") or {}).get("bank", 0)

            payload = {
                "squad": squad_ids,
                "captain_id": captain_id,
                "bank": round((bank_raw or 0) / 10.0, 1),
                "free_transfers": body.get("free_transfers", existing.get("free_transfers", 1)),
                "gameweek": gameweek,
                "chips_used": existing.get("chips_used", []),
            }
            result = save_squad(payload)
            send_json(self, 200, result)
        except Exception as e:
            send_json(self, 500, {"error": str(e)})

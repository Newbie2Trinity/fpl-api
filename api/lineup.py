import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from _lib import fetch_fpl_data, get_history_cache, XPModel, best_xi_from_squad, read_json_body, send_json, send_cors_preflight


class handler(BaseHTTPRequestHandler):
    """Pick the best starting XI/captain/vice from a squad you already own --
    no transfers, no budget. This is the week-to-week 'who do I play and who
    do I captain' call, as opposed to /api/optimize which builds a fresh
    15 from scratch (for wildcard/free hit weeks)."""

    def do_OPTIONS(self):
        send_cors_preflight(self)

    def do_POST(self):
        try:
            body = read_json_body(self)
            gameweek = int(body.get("gameweek", 1))
            horizon = int(body.get("horizon", 1))
            current_squad_ids = body.get("current_squad_ids") or []

            if not current_squad_ids:
                send_json(self, 400, {"error": "No squad saved yet -- build one from scratch first."})
                return

            bootstrap, fixtures = fetch_fpl_data()
            history_cache = get_history_cache()
            model = XPModel(bootstrap, fixtures, history_cache=history_cache)

            if horizon <= 1:
                players = model.project_gameweek(gameweek)
            else:
                players = model.project_horizon(gameweek, horizon)

            result = best_xi_from_squad(players, current_squad_ids)
            send_json(self, 200, result)
        except Exception as e:
            send_json(self, 500, {"error": str(e)})

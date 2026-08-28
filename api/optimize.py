import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from _lib import fetch_fpl_data, get_history_cache, XPModel, optimize_squad, read_json_body, send_json, send_cors_preflight


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        send_cors_preflight(self)

    def do_POST(self):
        try:
            body = read_json_body(self)
            gameweek = int(body.get("gameweek", 1))
            horizon = int(body.get("horizon", 1))
            budget = float(body.get("budget", 100.0))
            locked_ids = body.get("locked_ids") or []
            excluded_ids = body.get("excluded_ids") or []

            bootstrap, fixtures = fetch_fpl_data()
            history_cache = get_history_cache()
            model = XPModel(bootstrap, fixtures, history_cache=history_cache)

            if horizon <= 1:
                players = model.project_gameweek(gameweek)
            else:
                players = model.project_horizon(gameweek, horizon)

            result = optimize_squad(
                players,
                budget=budget,
                locked_ids=locked_ids,
                excluded_ids=excluded_ids,
            )
            send_json(self, 200, result)
        except Exception as e:
            send_json(self, 500, {"error": str(e)})

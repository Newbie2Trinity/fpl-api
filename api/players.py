import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from _lib import fetch_fpl_data, get_history_cache, XPModel, send_json, send_cors_preflight


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        send_cors_preflight(self)

    def do_GET(self):
        try:
            query = parse_qs(urlparse(self.path).query)
            gameweek = int(query.get("gameweek", [1])[0])
            horizon = int(query.get("horizon", [1])[0])

            bootstrap, fixtures = fetch_fpl_data()
            history_cache = get_history_cache()
            model = XPModel(bootstrap, fixtures, history_cache=history_cache)

            if horizon <= 1:
                players = model.project_gameweek(gameweek)
            else:
                players = model.project_horizon(gameweek, horizon)

            players = [p for p in players if p["fixtures_this_gw"] != 0]

            send_json(self, 200, {
                "gameweek": gameweek,
                "horizon": horizon,
                "players": players,
            })
        except Exception as e:
            send_json(self, 500, {"error": str(e)})

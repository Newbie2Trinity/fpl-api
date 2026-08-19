import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from _lib import fetch_fpl_data, XPModel, optimize_transfers, read_json_body, send_json, send_cors_preflight


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        send_cors_preflight(self)

    def do_POST(self):
        try:
            body = read_json_body(self)
            gameweek = int(body.get("gameweek", 1))
            horizon = int(body.get("horizon", 1))
            current_squad_ids = body.get("current_squad_ids") or []
            bank = float(body.get("bank", 0.0))
            free_transfers = int(body.get("free_transfers", 1))
            max_transfers = int(body.get("max_transfers", 5))

            bootstrap, fixtures = fetch_fpl_data()
            model = XPModel(bootstrap, fixtures)

            if horizon <= 1:
                players = model.project_gameweek(gameweek)
            else:
                players = model.project_horizon(gameweek, horizon)

            result = optimize_transfers(
                players,
                current_squad_ids=current_squad_ids,
                bank=bank,
                free_transfers=free_transfers,
                max_transfers=max_transfers,
            )
            send_json(self, 200, result)
        except Exception as e:
            send_json(self, 500, {"error": str(e)})

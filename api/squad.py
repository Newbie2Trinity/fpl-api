import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from _lib import get_squad, save_squad, read_json_body, send_json, send_cors_preflight


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        send_cors_preflight(self)

    def do_GET(self):
        try:
            row = get_squad()
            send_json(self, 200, row)
        except Exception as e:
            send_json(self, 500, {"error": str(e)})

    def do_POST(self):
        try:
            required_password = os.environ.get("APP_PASSWORD")
            if required_password:
                supplied = self.headers.get("X-App-Password")
                if supplied != required_password:
                    send_json(self, 401, {"error": "Invalid or missing X-App-Password"})
                    return

            body = read_json_body(self)
            payload = {
                "squad": body.get("squad_ids"),
                "captain_id": body.get("captain_id"),
                "bank": body.get("bank", 0),
                "free_transfers": body.get("free_transfers", 1),
                "gameweek": body.get("gameweek"),
                "chips_used": body.get("chips_used", []),
            }
            result = save_squad(payload)
            send_json(self, 200, result)
        except Exception as e:
            send_json(self, 500, {"error": str(e)})

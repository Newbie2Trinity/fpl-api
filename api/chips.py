from http.server import BaseHTTPRequestHandler

from _lib import fetch_fpl_data, XPModel, analyze_chip_windows, read_json_body, send_json, send_cors_preflight


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        send_cors_preflight(self)

    def do_POST(self):
        try:
            body = read_json_body(self)
            gameweek = int(body.get("gameweek", 1))
            horizon = int(body.get("horizon", 8))
            current_squad_ids = body.get("current_squad_ids") or []
            bank = float(body.get("bank", 0.0))
            chips_used = body.get("chips_used") or []

            bootstrap, fixtures = fetch_fpl_data()
            model = XPModel(bootstrap, fixtures)

            result = analyze_chip_windows(
                model,
                current_squad_ids=current_squad_ids,
                bank=bank,
                start_gw=gameweek,
                n_gws=horizon,
                chips_used=chips_used,
            )
            send_json(self, 200, result)
        except Exception as e:
            send_json(self, 500, {"error": str(e)})

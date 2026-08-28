import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from _lib import fetch_fpl_data, refresh_history_cache, send_json, send_cors_preflight


class handler(BaseHTTPRequestHandler):
    """Refreshes player_history_cache (last season's game-time % per player),
    used as the xP model's preseason 'will they actually start' prior. Does
    ~650 individual HTTP calls to FPL (concurrently -- see
    _lib.refresh_history_cache), so it's meant to be triggered occasionally
    (a weekly Vercel Cron hits this via GET, see vercel.json), not on every
    page load -- the ordinary endpoints just read whatever's cached.

    GET  -- for Vercel Cron. If CRON_SECRET is set, requires the
            Authorization: Bearer <CRON_SECRET> header Vercel Cron sends
            automatically. https://vercel.com/docs/cron-jobs/manage-cron-jobs
    POST -- for a manual trigger (browser/curl). Gated the same way as the
            other write-ish endpoints, by X-App-Password / APP_PASSWORD.
    """

    def do_OPTIONS(self):
        send_cors_preflight(self)

    def _run(self):
        bootstrap, _fixtures = fetch_fpl_data()
        result = refresh_history_cache(bootstrap)
        send_json(self, 200, result)

    def do_GET(self):
        cron_secret = os.environ.get("CRON_SECRET")
        if cron_secret:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {cron_secret}":
                send_json(self, 401, {"error": "Unauthorized"})
                return
        try:
            self._run()
        except Exception as e:
            send_json(self, 500, {"error": str(e)})

    def do_POST(self):
        required_password = os.environ.get("APP_PASSWORD")
        if required_password:
            supplied = self.headers.get("X-App-Password")
            if supplied != required_password:
                send_json(self, 401, {"error": "Invalid or missing X-App-Password"})
                return
        try:
            self._run()
        except Exception as e:
            send_json(self, 500, {"error": str(e)})

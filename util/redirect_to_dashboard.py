#!/usr/bin/env python3
"""Tiny redirect front-door so you can reach the dashboard at
``https://omniai.local`` instead of ``https://omniai.local:8080``.

Listens on the privileged ports 80 and 443 and 301-redirects every request
to the real dashboard on :8080, preserving the requested host and path. The
dashboard itself keeps serving HTTPS on 8080 as before; this just lets the
port drop out of the URL the user types.

Run as root (needed to bind 80/443) via the omniai-redirect.service unit.
"""
import ssl
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DASHBOARD_PORT = 8080
CERT = '/home/admin/omniai/cert.pem'
KEY = '/home/admin/omniai/key.pem'


class RedirectHandler(BaseHTTPRequestHandler):
    def _redirect(self):
        # Preserve whatever host the client used (omniai.local, an IP, ...),
        # strip any incoming port, and point at the dashboard port.
        host = self.headers.get('Host', 'omniai.local').rsplit(':', 1)[0]
        target = f'https://{host}:{DASHBOARD_PORT}{self.path}'
        self.send_response(301)
        self.send_header('Location', target)
        self.send_header('Content-Length', '0')
        self.end_headers()

    do_GET = _redirect
    do_HEAD = _redirect

    def log_message(self, *args):  # keep journald quiet
        pass


def serve(port, use_tls):
    httpd = ThreadingHTTPServer(('0.0.0.0', port), RedirectHandler)
    if use_tls:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(CERT, KEY)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    httpd.serve_forever()


if __name__ == '__main__':
    # Plain HTTP on 80 in a background thread; HTTPS on 443 in the main thread.
    threading.Thread(target=serve, args=(80, False), daemon=True).start()
    try:
        serve(443, True)
    except FileNotFoundError:
        sys.exit(f"[redirect] cert/key not found ({CERT}, {KEY}) — "
                 "start the dashboard once to generate them, or run --no-ssl")

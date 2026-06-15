#!/usr/bin/env python3
"""Minimal static server for the Veritas demo (avoids http.server's CLI)."""
import http.server, socketserver, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("PORT", "8777"))  # go.sh leaves it unset → 8777; preview can override
socketserver.TCPServer.allow_reuse_address = True  # survive restarts (no TIME_WAIT bind fail)
with socketserver.TCPServer(("127.0.0.1", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"serving Veritas dApp on http://127.0.0.1:{PORT}/app.html", flush=True)
    httpd.serve_forever()

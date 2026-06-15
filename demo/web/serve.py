#!/usr/bin/env python3
"""Minimal static server for the Veritas demo (avoids http.server's CLI)."""
import http.server, socketserver, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
socketserver.TCPServer.allow_reuse_address = True  # survive restarts (no TIME_WAIT bind fail)
with socketserver.TCPServer(("127.0.0.1", 8777), http.server.SimpleHTTPRequestHandler) as httpd:
    print("serving Veritas dApp on http://127.0.0.1:8777/app.html", flush=True)
    httpd.serve_forever()

#!/usr/bin/env python3
"""Minimal static server for the Veritas demo (avoids http.server's CLI)."""
import http.server, socketserver, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
socketserver.TCPServer(("127.0.0.1", 8777), http.server.SimpleHTTPRequestHandler).serve_forever()

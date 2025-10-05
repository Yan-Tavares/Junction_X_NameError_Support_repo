#!/usr/bin/env python3
"""
Simple HTTP server to run the Vocal Firewall web frontend.
Usage: python run_frontend.py [port]
"""

import http.server
import socketserver
import sys
import os
from pathlib import Path

# Change to the directory containing this script
os.chdir(Path(__file__).parent)

# Get port from command line or use default
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"╔═══════════════════════════════════════════════════════╗")
        print(f"║   🗣️  Vocal Firewall - Frontend Server               ║")
        print(f"╠═══════════════════════════════════════════════════════╣")
        print(f"║   Server running at: http://localhost:{PORT}        ║")
        print(f"║   Press Ctrl+C to stop                                ║")
        print(f"╚═══════════════════════════════════════════════════════╝")
        print()
        print(f"Open your browser and navigate to: http://localhost:{PORT}")
        print()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            httpd.shutdown()

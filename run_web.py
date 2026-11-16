"""
run_web.py – Entry point for the VideoVigil Web Dashboard.

Usage:
    python run_web.py [--host 127.0.0.1] [--port 8000] [--no-browser]
"""

from webapp.server import main

if __name__ == "__main__":
    main()

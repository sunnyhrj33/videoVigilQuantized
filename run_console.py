"""
run_console.py – Entry point for the VideoVigil Desktop Operator Console.

Usage:
    python run_console.py [--demo demo1|demo2|demo3] [--source PATH] [--mode tracklet|naive|legacy] [--start]
"""

from console.app import main

if __name__ == "__main__":
    main()

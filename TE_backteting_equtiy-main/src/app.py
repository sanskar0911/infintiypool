"""
app.py

QUANTLAB — AI-Powered Equity Backtesting & Research Terminal.
Master Application Entrypoint linked directly to server.py.

Usage:
    python app.py
    python src/app.py
"""

import os
import sys

# Ensure local module imports work
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from server import start_quantum_server


def main():
    """Main application entrypoint executing QUANTUM LABS Web Server directly from app.py."""
    port_num = 8000
    if len(sys.argv) > 1:
        try:
            port_num = int(sys.argv[1])
        except ValueError:
            pass
    print("\n" + "=" * 80)
    print("🚀 QUANTUM LABS Web Dashboard Terminal — Starting via app.py ...")
    print("=" * 80)
    start_quantum_server(port_num)


if __name__ == "__main__":
    main()

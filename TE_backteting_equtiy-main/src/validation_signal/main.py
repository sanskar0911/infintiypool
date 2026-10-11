"""QuantLab runner located in validation_signal directory."""

from pathlib import Path
import sys

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from validation_signal.backtester1 import main

if __name__ == "__main__":
    main()


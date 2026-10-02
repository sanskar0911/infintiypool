"""QuantLab runner located in validation_signal directory."""

from pathlib import Path
import sys

# Add quantlab to path and run main
QUANTLAB_DIR = Path(__file__).resolve().parent.parent / "quantlab"
if str(QUANTLAB_DIR) not in sys.path:
    sys.path.insert(0, str(QUANTLAB_DIR))

from main import main

if __name__ == "__main__":
    main()

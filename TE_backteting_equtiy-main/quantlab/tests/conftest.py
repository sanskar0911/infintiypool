import sys
from pathlib import Path

# Add quantlab directory to sys.path so 'src' resolves to quantlab/src
QUANTLAB_DIR = Path(__file__).resolve().parent.parent
if str(QUANTLAB_DIR) not in sys.path:
    sys.path.insert(0, str(QUANTLAB_DIR))

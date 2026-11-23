"""Make the package importable from a source checkout without installation (pip install -e . also works)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))

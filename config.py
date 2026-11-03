from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "weights" / "best.pt"
CLASS_NAMES = ["D00", "D10", "D20", "D40"]

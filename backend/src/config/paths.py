"""Filesystem resources resolved independently of the working directory."""
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = BACKEND_ROOT / "models"
OUTPUTS_ROOT = BACKEND_ROOT / "outputs"
EMOTION_MODELS = MODELS_ROOT / "emotion"
DEFAULT_EMOTION_MODEL = EMOTION_MODELS / "fer2013_best_model.pth"
FER2013_ROOT = DATA_ROOT / "raw" / "fer2013"
DIST = PROJECT_ROOT / "web-react" / "dist"

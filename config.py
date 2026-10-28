"""All tunable settings in one place."""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"          # dataset/<person_name>/*.jpg
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "lbph_model.yml"
LABELS_PATH = MODEL_DIR / "labels.json"
LOG_PATH = BASE_DIR / "deliveries.csv"

FACE_SIZE = (200, 200)       # every face crop is resized to this before training/predicting
SAMPLES_PER_PERSON = 40      # images captured during enrollment
CONFIDENCE_THRESHOLD = 70.0  # LBPH "distance": LOWER = better match. Above this => stranger
FRAMES_REQUIRED = 10         # recipient must be matched this many frames IN A ROW
SERVO_PIN = 17               # GPIO pin of the gripper servo (Raspberry Pi only)

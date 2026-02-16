# ===== config.py =====
import os

# API Keys
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # User should replace this with their actual key
# SCENE_PATH = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"  # Modern apartment scene (Stable)
SCENE_PATH = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"  # Modern apartment scene (Stable)

YOLO_MODEL_PATH = "yolov8m.pt" # Upgraded to Medium version for better accuracy

# Detector Configuration
DETECTOR_TYPE = "YOLO" # Options: "YOLO", "GROUNDING_DINO"
# DETECTOR_TYPE = "GROUNDING_DINO"
GROUNDING_DINO_PROMPT = "door, chair, bed, table, window, floor, wall, painting, picture, person" # Comma-separated list for open-vocab detection

# Simulation Settings
AGENT_HEIGHT = 1.5
AGENT_RADIUS = 0.1
SENSOR_HEIGHT = 1.5
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOV = 90

# Exploration
NUM_STEPS = 50  # Number of random steps to take

# Models
CLIP_MODEL_NAME = "ViT-B-16" 
CLIP_PRETRAINED = "laion2b_s34b_b88k"

# Retrieval
TOP_K = 10
W_CLIP = 0.55
W_YOLO = 0.15
W_BM25 = 0.3
RETRIEVAL_THRESHOLD_COMBINED = 0.25 # Minimum combined score to be considered a match

# Memory
PERSIST_MEMORY = False


# Deduplication
SPATIAL_THRESHOLD = 1.0 # meters
VISUAL_THRESHOLD = 0.85 # cosine similarity

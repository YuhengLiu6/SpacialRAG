# ===== config.py =====
import os

# Paths (Adjust these paths to your actual environment)
SCENE_PATH = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"  # Object Detection
YOLO_MODEL_NAME = "yolov8x.pt" # Best accuracy (was yolov8m.pt)
CONFIDENCE_THRESHOLD = 0.25 # Lowered slightly for better recall with stronger model
# Will download automatically if not present

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
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "laion2b_s32b_b82k"


# Retrieval
TOP_K = 10

# Memory
PERSIST_MEMORY = False


# Deduplication
SPATIAL_THRESHOLD = 1.0 # meters
VISUAL_THRESHOLD = 0.85 # cosine similarity

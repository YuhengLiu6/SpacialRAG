# ===== config.py =====
import os

# Prevent Faiss/OpenMP deadlocks on macOS when used alongside PyTorch MPS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# API Keys
# os.environ["OPENAI_API_KEY"] = "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" # User should replace this with their actual key
os.environ["OPENAI_API_KEY"] = "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" 
# SCENE_PATH = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"  # Modern apartment scene (Stable)
# SCENE_PATH = "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"  # Modern apartment scene (Stable)
SCENE_PATH = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"  # Modern apartment scene (Stable)


YOLO_MODEL_PATH = "yolov8m.pt" # Upgraded to Medium version for better accuracy
YOLO_WORLD_MODEL_PATH = "yolov8s-world.pt"
YOLO_WORLD_CLASSES = "door, chair, bed, table, window, floor, wall, painting, picture, person"

# Detector Configuration
DETECTOR_TYPE = "YOLO_WORLD" # Options: "YOLO", "YOLO_WORLD", "GROUNDING_DINO"
# DETECTOR_TYPE = "GROUNDING_DINO"
GROUNDING_DINO_PROMPT = "door, chair, bed, table, window, floor, wall, painting, picture, person" # Comma-separated list for open-vocab detection

# Simulation Settings
AGENT_HEIGHT = 1.6
AGENT_RADIUS = 0.1
SENSOR_HEIGHT = 1.6
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
FOV = 90

# Exploration
NUM_STEPS = 50  # Number of random steps to take
# Global scan angles (degrees) captured at each waypoint.
# This is shared by Explorer, spatial_db_builder, and VPR overlay/heatmap logic.
# SCAN_ANGLES = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330)
SCAN_ANGLES = (0, 90, 180, 270)

# Models
CLIP_MODEL_NAME = "ViT-B-16" 
CLIP_PRETRAINED = "laion2b_s34b_b88k"

# Retrieval
RETRIEVAL_METHOD = "cosine" # Options: "hybrid", "cosine"
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

# Spatial DB defaults
SPATIAL_DB_DIR = "spatial_db"
SPATIAL_DB_VLM_MODEL = "gpt-5-mini"
OBJECT_TEXT_MODE = "short"  # Options: "short", "long"
OBJECT_MAX_PER_FRAME = 24
OBJECT_SURROUNDING_MAX = 5
OBJECT_VERTICAL_REL_EPS_M = 0.25
OBJECT_SCORE_WEIGHT = 0.25
OBJECT_PARSE_RETRIES = 1
OBJECT_USE_CACHE = True
OBJECT_CACHE_DIR = "vlm_object_cache"
OBJECT_RERANK_CANDIDATES = 30
VLM_ANGLE_SPLIT_ENABLE = True
VLM_ANGLE_STEP = 30
VLM_ANGLE_SPLIT_PROMPT_MODE = "three_way"
OBJECT_GEOMETRY_PIPELINE_ENABLE = True
OBJECT_PRELIST_TAXONOMY_PATH = "spatial_rag/household_taxonomy.py"
SAVE_GEOMETRY_ARTIFACTS = True
NANOSAM_ENCODER_PATH = os.environ.get("NANOSAM_ENCODER_PATH", "models/nanosam/resnet18_image_encoder.engine")
NANOSAM_DECODER_PATH = os.environ.get("NANOSAM_DECODER_PATH", "models/nanosam/mobile_sam_mask_decoder.engine")
DEPTH_PRO_MODEL_PATH = os.environ.get("DEPTH_PRO_MODEL_PATH", "models/depth_pro/depth_pro.pt")

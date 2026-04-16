# ===== config.py =====
import os

# Prevent Faiss/OpenMP deadlocks on macOS when used alongside PyTorch MPS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# API Keys
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
BBOX_CONF_THRESHOLD = 0.3
OCCLUSION_REWEIGHT_W1 = 1.0
OCCLUSION_REWEIGHT_W2 = 1.0
OCCLUSION_REWEIGHT_B = 0.0
OCCLUSION_REWEIGHT_EPS = 1e-6
OCCLUSION_SOURCE = "visible_mask"
OCCLUSION_TARGET_OVERLAP_THRESHOLD = 0.1
VISIBLE_OCC_BOUNDARY_WIDTH = 1
VISIBLE_OCC_RING_RADIUS = 5
VISIBLE_OCC_DEPTH_MARGIN_DELTA = 0.0
VISIBLE_OCC_BOUNDARY_NEIGHBOR_RADIUS = 1
VLM_ANGLE_SPLIT_ENABLE = True
VLM_ANGLE_STEP = 30
VLM_ANGLE_SPLIT_PROMPT_MODE = "three_way"
OBJECT_GEOMETRY_PIPELINE_ENABLE = True
OBJECT_PRELIST_TAXONOMY_PATH = "spatial_rag/household_taxonomy.py"
SAVE_GEOMETRY_ARTIFACTS = True
NANOSAM_ENCODER_PATH = os.environ.get("NANOSAM_ENCODER_PATH", "data/resnet18_image_encoder.engine")
NANOSAM_DECODER_PATH = os.environ.get("NANOSAM_DECODER_PATH", "data/mobile_sam_mask_decoder.engine")
NANOSAM_CHECKPOINT_PATH = os.environ.get("NANOSAM_CHECKPOINT_PATH", "models/nanosam/assets/mobile_sam.pt")
DEPTH_PRO_MODEL_PATH = os.environ.get("DEPTH_PRO_MODEL_PATH", "models/depth_pro/depth_pro.pt")

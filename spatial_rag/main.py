# ===== main.py =====
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
from tqdm import tqdm
from pprint import pprint

from spatial_rag.config import NUM_STEPS, PERSIST_MEMORY, RETRIEVAL_METHOD
from spatial_rag.explorer import Explorer
from spatial_rag.detector import Detector
from spatial_rag.embedder import Embedder
from spatial_rag.memory import Memory
from spatial_rag.retriever import Retriever

def main():
    print("Starting Spatial RAG System...")

    # 1. Initialize Modules
    try:
        explorer = Explorer()
    except Exception as e:
        print(f"Failed to initialize Explorer: {e}")
        return

    detector = Detector()
    embedder = Embedder()
    memory = Memory(dimension=512) # ViT-B-32 has 512 dims
    
    # Load existing memory if available and persistence is enabled
    if PERSIST_MEMORY and os.path.exists("memory_index.faiss") and os.path.exists("memory_meta.npy"):
        try:
            memory.load("memory_index.faiss", "memory_meta.npy")
            print(f"Loaded existing memory with {len(memory.metadata)} items.")
        except Exception as e:
            print(f"Could not load existing memory: {e}")
            
    retriever = Retriever(embedder, memory)

    import shutil
    
    # Setup image storage
    img_dir = "captured_images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    print("\nExploration started (Full House)...")
    # frames, poses = explorer.explore_custom_tour(
    #     num_steps=34,                 # 34 + start = 35
    #     step_size=1.5,                # 和之前 meters_per_step 类似
    #     scan_angles=(0, 120, 240),
    #     seed=42,                      # 固定随机种子保证可复现
    #     max_attempts_per_step=32,
    #     include_start_scan=True,
    # )
    frames, poses = explorer.explore_full_house()
    print(f"Captured {len(frames)} frames. Processing...")

    
    
    # 2. Exploration Loop
    
    # for step, (rgb_image, pose) in enumerate(tqdm(zip(frames, poses), total=len(frames))):
    #     position = pose["position"]
    #     rotation = pose["rotation"]
        
    #     # Save image to disk
    #     # Use a unique ID based on total items in memory to avoid overwriting
    #     # (A robust system would use UUIDs, here we just use linear index approximation or timestamp)
    #     import time
    #     img_id = int(time.time() * 1000) 
    #     img_filename = f"step_{img_id}_{step:04d}.jpg"
    #     img_path = os.path.join(img_dir, img_filename)
    #     # Convert RGB to BGR for OpenCV saving
    #     cv2.imwrite(img_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        
    #     # Detect objects
    #     detections = detector.detect(rgb_image)
        
    #     if not detections:
    #         continue
            
    #     # Process detections
    #     for det in detections:
    #         label = det["label"]
    #         bbox = det["bbox"] # [x1, y1, x2, y2]
    #         conf = det["confidence"]
            
    #         # Crop object image
    #         x1, y1, x2, y2 = map(int, bbox)
            
    #         # Clip coordinates to image bounds
    #         h, w, _ = rgb_image.shape
    #         x1 = max(0, x1)
    #         y1 = max(0, y1)
    #         x2 = min(w, x2)
    #         y2 = min(h, y2)
            
    #         if x2 - x1 < 5 or y2 - y1 < 5:
    #             continue # Skip too small crops
                
    #         crop = rgb_image[y1:y2, x1:x2]
            
    #         # Embed object crop
    #         embedding = embedder.embed_image(crop)
            
    #         # Create metadata
    #         meta = {
    #             "step": step,
    #             "label": label,
    #             "confidence": conf,
    #             "bbox": [x1, y1, x2, y2],
    #             "position": position.tolist(), # [x, y, z]
    #             "rotation": [rotation.w, rotation.x, rotation.y, rotation.z], # quaternion
    #             "image_path": img_path
    #         }
            
    #         # Store in memory
    #         memory.add(embedding, meta)

    # Optional: Generate and save floor plan trajectory
    floor_map = explorer.render_center_highest_view(hfov=120.0)
    cv2.imwrite("center_highest_view.jpg", floor_map)
    trajectory_map = explorer.render_center_highest_view_with_trajectory(poses, hfov=120.0) 
    cv2.imwrite("trajectory_map.jpg", trajectory_map)
    trajectory_floor_map = explorer.draw_trajectory_on_true_floor_plan(poses) 
    cv2.imwrite("trajectory_floor_map.jpg", trajectory_floor_map)
    print("\nSaved center-highest view to 'center_highest_view.jpg'")
    print("Saved trajectory map to 'trajectory_map.jpg'")
    print("Saved trajectory floor map to 'trajectory_floor_map.jpg'")

    print("\nExploration complete. Memory populated.")
    if PERSIST_MEMORY:
        memory.save()
    
    # # 3. Retrieval Example
    # # query = "Where did I see a painting with human?"
    # query = "Where did I see a pillow?"
    # # query = "Where did I see a room?"


    # print(f"\nTest Query: '{query}'")
    
    # results = retriever.retrieve(query, k=10, method=RETRIEVAL_METHOD) # Overriding config to show usage
    
    # if results:
    #     # Prepare directory for results
    #     res_dir = "retrieval_results"
    #     if os.path.exists(res_dir):
    #         shutil.rmtree(res_dir)
    #     os.makedirs(res_dir)
        
    #     print(f"\nTop {len(results)} Results ({RETRIEVAL_METHOD.capitalize()} Search):")
    #     for i, res in enumerate(results):
    #         # Safe get for new fields
    #         combined = res.get('retrieval_score', 0)
    #         clip = res.get('clip_score', 0)
    #         yolo = res.get('yolo_conf', 0)
    #         bm25 = res.get('bm25_score', 0)
    #         query_used = res.get('matched_query', 'original')
            
    #         print(f"[{i+1}] {res['label']}")
    #         print(f"    Scores: Combined={combined:.4f} (CLIP={clip:.2f}, YOLO={yolo:.2f}, BM25={bm25:.2f})")
    #         print(f"    Matched Query: '{query_used}'")
    #         print(f"    Position: {res['position']}")
    #         print(f"    Image: {res['image_path']}")
            
    #         # Visualize
    #         if os.path.exists(res['image_path']):
    #             res_img = cv2.imread(res['image_path'])
    #             x1, y1, x2, y2 = map(int, res['bbox'])
                
    #             # Draw bounding box (Red)
    #             cv2.rectangle(res_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
    #             # Draw label
    #             text = f"#{i+1} {res['label']} (S:{combined:.2f})"
    #             cv2.putText(res_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
    #             # Save
    #             save_path = os.path.join(res_dir, f"rank_{i+1:02d}_{res['label']}.jpg")
    #             cv2.imwrite(save_path, res_img)
    #         else:
    #             print(f"    Warning: Image file not found: {res['image_path']}")

    #     print(f"\n[Visual Verification] Saved {len(results)} annotated images to '{os.path.abspath(res_dir)}'")
    # else:
    #     print(f"\nI'm sorry, I didn't see any '{query.replace('Where did I see a ', '').replace('?', '')}' with high enough confidence during my walk.")

    explorer.close()

if __name__ == "__main__":
    main()

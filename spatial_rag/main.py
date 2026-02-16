# ===== main.py =====
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
from tqdm import tqdm
from pprint import pprint

from spatial_rag.config import NUM_STEPS, PERSIST_MEMORY
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
    memory = Memory(dimension=embedder.output_dim) # Dynamic dimension

    
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
    
    print(f"\nExploration started for {NUM_STEPS} steps...")
    
    # 2. Exploration Loop
    for step in tqdm(range(NUM_STEPS)):
        # Get observation from explorer
        rgb_image, position, rotation = explorer.step_random()
        
        # Save image to disk
        # Use a unique ID based on total items in memory to avoid overwriting
        # (A robust system would use UUIDs, here we just use linear index approximation or timestamp)
        import time
        img_id = int(time.time() * 1000) 
        img_filename = f"step_{img_id}_{step:04d}.jpg"
        img_path = os.path.join(img_dir, img_filename)
        # Convert RGB to BGR for OpenCV saving
        cv2.imwrite(img_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        
        # Detect objects
        detections = detector.detect(rgb_image)
        
        if not detections:
            continue
            
        # Process detections
        for det in detections:
            label = det["label"]
            bbox = det["bbox"] # [x1, y1, x2, y2]
            conf = det["confidence"]
            
            # Crop object image
            x1, y1, x2, y2 = map(int, bbox)
            
            # Clip coordinates to image bounds
            h, w, _ = rgb_image.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue # Skip too small crops
                
            crop = rgb_image[y1:y2, x1:x2]
            
            # Embed object crop
            embedding = embedder.embed_image(crop)
            
            # Create metadata
            meta = {
                "step": step,
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "position": position.tolist(), # [x, y, z]
                "rotation": [rotation.w, rotation.x, rotation.y, rotation.z], # quaternion
                "image_path": img_path
            }
            
            # Store in memory
            memory.add(embedding, meta)

    print("\nExploration complete. Memory populated.")
    if PERSIST_MEMORY:
        memory.save()
    
    # 3. Retrieval Example
    query = "Where did I see a man?"
    print(f"\nTest Query: '{query}'")
    
    results = retriever.retrieve(query, k=10) # Overriding config to show usage
    
    if results:
        # Prepare directory for results
        res_dir = "retrieval_results"
        if os.path.exists(res_dir):
            shutil.rmtree(res_dir)
        os.makedirs(res_dir)
        
        print(f"\nTop {len(results)} Results:")
        for i, res in enumerate(results):
            print(f"[{i+1}] {res['label']} (Conf: {res['confidence']:.2f}, Score: {res['retrieval_score']:.4f})")
            print(f"    Position: {res['position']}")
            print(f"    Image: {res['image_path']}")
            
            # Visualize
            if os.path.exists(res['image_path']):
                res_img = cv2.imread(res['image_path'])
                x1, y1, x2, y2 = map(int, res['bbox'])
                
                # Draw bounding box (Red)
                cv2.rectangle(res_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw label
                text = f"#{i+1} {res['label']} ({res['confidence']:.2f})"
                cv2.putText(res_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Save
                save_path = os.path.join(res_dir, f"rank_{i+1:02d}_{res['label']}.jpg")
                cv2.imwrite(save_path, res_img)
            else:
                print(f"    Warning: Image file not found: {res['image_path']}")

        print(f"\n[Visual Verification] Saved {len(results)} annotated images to '{os.path.abspath(res_dir)}'")
    else:
        print("No matches found.")

    explorer.close()

if __name__ == "__main__":
    main()

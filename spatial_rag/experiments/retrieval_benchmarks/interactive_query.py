import os
import glob
import sys
import argparse

# Add the antigravityTest root directory to sys.path so 'spatial_rag' module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
# Also add spatial_rag directory for direct local imports of the method files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from spatial_rag.embedder import Embedder
from method1_clip import Method1CLIP
from method2_vlm import Method2VLM
from method3_scorer import Method3Scorer

def main():
    parser = argparse.ArgumentParser(description="Run Interactive Retrieval Query across Methods.")
    parser.add_argument("--query", type=str, default="Find a painting of a man on a horse", help="The query text to search for.")
    parser.add_argument("--img_dir", type=str, default="../captured_images", help="Directory containing images to search.")
    parser.add_argument("--max_frames", type=int, default=30, help="Maximum number of frames to test against (to save time/cost).")
    args = parser.parse_args()

    print(f"=== Interactive Interactive Retrieval ===")
    print(f"Query: '{args.query}'")
    
    # 1. Gather Images
    image_paths = sorted(glob.glob(os.path.join(os.path.abspath(args.img_dir), '*.jpg')))
    if not image_paths:
        print(f"No images found in {args.img_dir}")
        return
        
    image_paths = image_paths[:args.max_frames]
    print(f"Testing against {len(image_paths)} frames...")

    # 2. Initialize Embedder once to share between methods
    print(f"\n[System] Loading core Embedder (CLIP)...")
    embedder = Embedder()

    # 3. Setup Methods
    m1 = Method1CLIP(embedder=embedder)
    m2 = Method2VLM(embedder=embedder)
    # Attempt to load trained method3 weights
    m3_weights_path = os.path.join(os.path.dirname(__file__), 'method3_trained.pth')
    if not os.path.exists(m3_weights_path):
        print("\nNotice: Method 3 weights not found. Please run 'run_benchmark.py' first to generate 'method3_trained.pth'.")
        print("Using untrained Method 3 model for now.")
    m3 = Method3Scorer(embedder=embedder, trained_scorer_path=m3_weights_path)


    # ==========================================
    # Execute Method 1
    # ==========================================
    print(f"\n>> Running Method 1 (Direct CLIP) <<")
    res1 = m1.run(args.query, image_paths)
    if res1['is_none']:
        print("Top-1 Prediction: None")
    else:
        print(f"Top-1 Prediction: Frame {res1['top1_idx']} (Path: {os.path.basename(res1['best_frame_path'])})")
    print(f"Confidence Score: {res1['top1_score']:.3f} (Threshold: {m1.threshold})")

    # ==========================================
    # Execute Method 2
    # ==========================================
    print(f"\n>> Running Method 2 (VLM Caption) <<")
    # Tell Method 2 to dump intermediate results into a specific folder
    output_vlm_dir = os.path.join(os.path.dirname(__file__), "vlm_intermediate_output")
    print(f"Note: VLM intermediate dense captions will be saved to: {output_vlm_dir}")
    
    res2 = m2.run(args.query, image_paths, save_intermediate_dir=output_vlm_dir)
    if res2['is_none']:
        print("Top-1 Prediction: None")
    else:
        print(f"Top-1 Prediction: Frame {res2['top1_idx']} (Path: {os.path.basename(res2['best_frame_path'])})")
        print(f"Matched Paragraph Output Snippet: \"{res2['best_frame_caption'][:150]}...\"")
    print(f"Confidence Score: {res2['top1_score']:.3f} (Threshold: {m2.threshold})")

    # ==========================================
    # Execute Method 3
    # ==========================================
    print(f"\n>> Running Method 3 (Trainable Frame Scorer) <<")
    res3 = m3.run(args.query, image_paths)
    if res3['is_none']:
        print("Top-1 Prediction: None")
    else:
        print(f"Top-1 Prediction: Frame {res3['top1_idx']} (Path: {os.path.basename(res3['best_frame_path'])})")
    print(f"Learned Probability: {res3['top1_score']:.3f} (1.0 = Max Confidence)")

    print("\n=== Execution Complete ===")


if __name__ == "__main__":
    main()

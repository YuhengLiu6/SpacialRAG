# ===== inspect_memory.py =====
import os
import argparse
from pprint import pprint
from spatial_rag.memory import Memory

def inspect(index_path="memory_index.faiss", meta_path="memory_meta.npy"):
    print(f"Loading memory from {index_path}...")
    
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print("Error: Memory files not found. Run main.py first.")
        return

    memory = Memory(dimension=512)
    try:
        memory.load(index_path, meta_path)
    except Exception as e:
        print(f"Error loading memory: {e}")
        return

    total_items = len(memory.metadata)
    print(f"\nTotal items in memory: {total_items}")
    
    if total_items == 0:
        return

    print("\n--- Content Summary ---")
    # Count objects by label
    label_counts = {}
    for meta in memory.metadata:
        label = meta.get('label', 'unknown')
        label_counts[label] = label_counts.get(label, 0) + 1
        
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    print("\n--- First 5 Entries ---")
    for i, meta in enumerate(memory.metadata[:5]):
        print(f"[{i}] {meta['label']} (Conf: {meta['confidence']:.2f}) at {meta['position']}")

    print("\n--- Detailed View of Last Entry ---")
    pprint(memory.metadata[-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Spatial RAG Memory")
    parser.add_argument("--index", default="memory_index.faiss", help="Path to FAISS index")
    parser.add_argument("--meta", default="memory_meta.npy", help="Path to metadata file")
    args = parser.parse_args()
    
    inspect(args.index, args.meta)

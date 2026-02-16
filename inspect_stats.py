import numpy as np
import faiss
from collections import Counter

# Load metadata
try:
    metadata = np.load("memory_meta.npy", allow_pickle=True).tolist()
    print(f"Total Memory Items: {len(metadata)}")
    
    label_counts = Counter([m['label'] for m in metadata])
    print("\nLabel Counts:")
    for label, count in label_counts.most_common():
        print(f"{label}: {count}")
        
    print(f"\nTotal Steps: {metadata[-1]['step'] + 1 if metadata else 0}")
    
except Exception as e:
    print(f"Error loading memory: {e}")

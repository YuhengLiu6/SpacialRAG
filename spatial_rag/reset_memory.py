# ===== reset_memory.py =====
import os
import shutil

def reset_memory():
    files_to_remove = ["memory_index.faiss", "memory_meta.npy"]
    dirs_to_remove = ["captured_images", "retrieval_results"]
    
    print("Resetting Spatial RAG Memory...")
    
    # Remove files
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        else:
            print(f"Not found (skipped): {file}")
            
    # Remove directories
    for directory in dirs_to_remove:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                print(f"Deleted directory: {directory}")
            except Exception as e:
                print(f"Error deleting {directory}: {e}")
        else:
            print(f"Not found (skipped): {directory}")

    print("Memory reset complete.")

if __name__ == "__main__":
    reset_memory()

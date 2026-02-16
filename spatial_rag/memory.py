# ===== memory.py =====
import faiss
import numpy as np

class Memory:
    def __init__(self, dimension=512):
        self.dimension = dimension
        # Using IndexFlatIP for Inner Product (Cosine Similarity if normalized)
        # Check if GPU is available for FAISS
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.IndexFlatIP(dimension)
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print("FAISS initialized on GPU.")
        except:
            self.index = faiss.IndexFlatIP(dimension)
            print("FAISS initialized on CPU.")
            
        self.metadata = [] # List to store metadata corresponding to IDs

    def add(self, embedding, meta):
        """
        Add an embedding and its metadata to memory.
        Args:
            embedding: numpy array (D,)
            meta: dict containing metadata (label, bbox, pose, etc.)
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
            
        self.index.add(embedding.astype('float32'))
        self.metadata.append(meta)

    def search(self, query_embedding, k=5):
        """
        Search for similar embeddings.
        Args:
            query_embedding: numpy array (D,)
            k: number of results
        Returns:
            list of tuples (metadata, score)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        if indices.size > 0:
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.metadata):
                    results.append((self.metadata[idx], float(distances[0][i])))
                    
        return results

    def save(self, filepath="memory_index.faiss", meta_filepath="memory_meta.npy"):
        """
        Save the FAISS index and metadata to disk.
        """
        # FAISS GPU indices cannot be pickled directly, move to CPU first
        cpu_index = faiss.index_gpu_to_cpu(self.index) if hasattr(faiss, 'index_gpu_to_cpu') else self.index
        faiss.write_index(cpu_index, filepath)
        np.save(meta_filepath, self.metadata)
        print(f"Memory saved to {filepath} and {meta_filepath}")

    def load(self, filepath="memory_index.faiss", meta_filepath="memory_meta.npy"):
        """
        Load the FAISS index and metadata from disk.
        """
        self.index = faiss.read_index(filepath)
        # Move back to GPU if available and desired (skipping for simplicity in inspection)
        self.metadata = np.load(meta_filepath, allow_pickle=True).tolist()
        print(f"Memory loaded from {filepath} and {meta_filepath}")

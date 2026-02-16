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
            
        results = []
        if self.index.ntotal > 0:
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            if indices.size > 0:
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and idx < len(self.metadata):
                        results.append((self.metadata[idx], float(distances[0][i])))
                    
        return results

    def search_hybrid(self, query_text, query_embedding, k=5):
        """
        Perform hybrid search using BM25 (keyword) + CLIP (semantic).
        """
        # 1. Get CLIP Results (Dense)
        # We fetch more candidates to allow BM25 to boost lower-ranked items
        dense_results = self.search(query_embedding, k=k*3)
        
        # 2. Get BM25 Results (Sparse)
        # We rebuild the index on the fly (simple for small memory)
        try:
            from rank_bm25 import BM25Okapi
            
            # Prepare corpus from metadata labels
            tokenized_corpus = [meta['label'].lower().split() for meta in self.metadata]
            bm25 = BM25Okapi(tokenized_corpus)
            
            tokenized_query = query_text.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Normalize BM25 scores (roughly 0-1)
            if len(bm25_scores) > 0 and max(bm25_scores) > 0:
                bm25_scores = bm25_scores / max(bm25_scores)
            
        except ImportError:
            print("Warning: rank_bm25 not installed. Skipping hybrid search.")
            bm25_scores = np.zeros(len(self.metadata))
        except Exception as e:
            print(f"BM25 Error: {e}")
            bm25_scores = np.zeros(len(self.metadata))

        # 3. Fuse Scores
        final_results = []
        
        # We need to map dense results back to their original index to get BM25 score
        # But wait, 'dense_results' is a list of (meta, score). 
        # We don't easily know the original index 'idx' from 'dense_results' unless we store acts.
        # Efficient approach: Just calculate combined score for the items returned by dense search?
        # NO, BM25 might find something that CLIP missed completely (e.g. exact match but bad visual).
        
        # correct approach for small scale: Calculate Combined Score for ALL items?
        # For < 10,000 items, this is fast.
        
        # Ensure 2D shape for FAISS
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        distances, indices = self.index.search(query_embedding.astype('float32'), len(self.metadata))
        
        all_results = []
        if indices.size > 0:
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.metadata):
                    meta = self.metadata[idx]
                    clip_score = float(distances[0][i])
                    bm25_score = float(bm25_scores[idx]) if idx < len(bm25_scores) else 0.0
                    
                    all_results.append({
                        "meta": meta,
                        "clip_score": clip_score,
                        "bm25_score": bm25_score
                    })
                    
        return all_results

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

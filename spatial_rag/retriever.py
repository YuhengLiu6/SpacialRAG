from spatial_rag.config import TOP_K, W_CLIP, W_YOLO, RETRIEVAL_THRESHOLD_COMBINED

class Retriever:
    def __init__(self, embedder, memory):
        self.embedder = embedder
        self.memory = memory

    def retrieve(self, query_text, k=TOP_K):
        """
        Retrieve objects based on text query using weighted scoring.
        """
        print(f"Retrieving for query: '{query_text}'...")
        query_emb = self.embedder.embed_text(query_text)
        
        # Fetch more results than needed to allow for filtering/re-ranking
        raw_results = self.memory.search(query_emb, k=k*3)
        
        formatted_results = []
        for meta, clip_score in raw_results:
            yolo_conf = meta.get('confidence', 0.0)
            
            # Combine scores
            combined_score = (W_CLIP * clip_score) + (W_YOLO * yolo_conf)
            
            # Filter by threshold
            if combined_score >= RETRIEVAL_THRESHOLD_COMBINED:
                res = meta.copy()
                res['clip_score'] = clip_score
                res['yolo_conf'] = yolo_conf
                res['retrieval_score'] = combined_score # Using as primary score now
                formatted_results.append(res)
        
        # Re-sort by combined score and return top k
        formatted_results.sort(key=lambda x: x['retrieval_score'], reverse=True)
        return formatted_results[:k]

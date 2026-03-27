from spatial_rag.config import TOP_K, W_CLIP, W_YOLO, W_BM25, RETRIEVAL_THRESHOLD_COMBINED, RETRIEVAL_METHOD
from spatial_rag.llm_utils import QueryExpander

class Retriever:
    def __init__(self, embedder, memory):
        self.embedder = embedder
        self.memory = memory
        self.query_expander = QueryExpander()

    def retrieve(self, query_text, k=TOP_K, method=RETRIEVAL_METHOD):
        """
        Retrieve objects based on text query using Hybrid Search and Query Expansion, or simple Cosine Similarity.
        """
        print(f"Retrieving for query: '{query_text}' using '{method}' method...")
        
        if method.lower() == "cosine":
            query_emb = self.embedder.embed_text(query_text)
            
            # Simple Cosine Search (inner product for normalized vectors)
            results = self.memory.search(query_emb, k=k*3)
            
            all_candidates = {}
            for meta, score in results:
                # Deduplicate: Keep the highest score for the same object
                unique_id = f"{meta['image_path']}_{meta['bbox']}"
                
                if unique_id not in all_candidates or score > all_candidates[unique_id]['retrieval_score']:
                    candidate = meta.copy()
                    candidate['clip_score'] = score
                    candidate['bm25_score'] = 0.0
                    candidate['yolo_conf'] = meta.get('confidence', 0.0)
                    candidate['retrieval_score'] = score # For cosine, the score is just the CLIP similarity
                    candidate['matched_query'] = query_text
                    all_candidates[unique_id] = candidate
                    
            final_results = []
            for cand in all_candidates.values():
                # We can apply a different threshold for raw cosine if needed, or just let top-k handle it.
                # CLIP cosine scores usually map differently, so we check if > 0.15 (arbitrary lower bound)
                if cand['retrieval_score'] >= 0.15: 
                    final_results.append(cand)
                    
            final_results.sort(key=lambda x: x['retrieval_score'], reverse=True)
            return final_results[:k]

        # --- Hybrid Method ---
        # 1. Expand Query
        expanded_queries = self.query_expander.expand_query(query_text)
        # Always include the original query if not present
        if query_text not in expanded_queries:
             expanded_queries.append(query_text)
             
        print(f"Searching for: {expanded_queries}")
        
        all_candidates = {} # Map ID (or index) to best result
        
        for q in expanded_queries:
            query_emb = self.embedder.embed_text(q)
            
            # Hybrid Search (Clip + BM25) from Memory
            # We get all items with their clip and bm25 scores
            results = self.memory.search_hybrid(q, query_emb, k=k)
            
            for res in results:
                meta = res['meta']
                clip_score = res['clip_score']
                bm25_score = res['bm25_score']
                yolo_conf = meta.get('confidence', 0.0)
                
                # Combine Scores
                combined_score = (W_CLIP * clip_score) + (W_YOLO * yolo_conf) + (W_BM25 * bm25_score)
                
                # Deduplicate: Keep the highest score for the same object
                # We assume 'image_path' + 'bbox' is a unique identifier proxy
                unique_id = f"{meta['image_path']}_{meta['bbox']}"
                
                if unique_id not in all_candidates or combined_score > all_candidates[unique_id]['retrieval_score']:
                    candidate = meta.copy()
                    candidate['clip_score'] = clip_score
                    candidate['bm25_score'] = bm25_score
                    candidate['yolo_conf'] = yolo_conf
                    candidate['retrieval_score'] = combined_score
                    candidate['matched_query'] = q
                    all_candidates[unique_id] = candidate

        # Filter and Sort
        final_results = []
        for cand in all_candidates.values():
            if cand['retrieval_score'] >= RETRIEVAL_THRESHOLD_COMBINED:
                final_results.append(cand)
                
        final_results.sort(key=lambda x: x['retrieval_score'], reverse=True)
        return final_results[:k]

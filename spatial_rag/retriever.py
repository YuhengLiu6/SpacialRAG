# ===== retriever.py =====
from spatial_rag.config import TOP_K

class Retriever:
    def __init__(self, embedder, memory):
        self.embedder = embedder
        self.memory = memory

    def retrieve(self, query_text, k=TOP_K):
        """
        Retrieve objects based on text query.
        Args:
            query_text: str
            k: int
        Returns:
            list of dicts (metadata + score)
        """
        print(f"Retrieving for query: '{query_text}'...")
        query_emb = self.embedder.embed_text(query_text)
        results = self.memory.search(query_emb, k=k)
        
        formatted_results = []
        for meta, score in results:
            # Create a copy to avoid modifying original info
            res = meta.copy()
            res['retrieval_score'] = score
            formatted_results.append(res)
            
        return formatted_results

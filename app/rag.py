import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class MiniRAG:
    def __init__(self, data_path: str = "data/news.txt"):
        self.data_path = data_path
        self.documents = []
        self.index = None
        # Use a small, fast model
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        self._build_index()

    def _build_index(self):
        if not os.path.exists(self.data_path):
            print(f"Warning: {self.data_path} not found.")
            return
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        self.documents = lines
        if not lines:
            return

        embeddings = self.model.encode(lines)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    # Adding Reranker for Bonus Points (+8 pkt)
    # Lazy loading or simple init
    def _get_reranker(self):
        if not hasattr(self, 'reranker'):
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                print(f"Reranker load failed: {e}")
                self.reranker = None
        return self.reranker

    def retrieve(self, query: str, top_k: int = 3) -> list:
        if not self.index or not self.documents:
            return []
        
        # 1. Retrieve more candidates (top_k * 3)
        initial_k = min(top_k * 3, len(self.documents))
        q_emb = self.model.encode([query])
        D, I = self.index.search(np.array(q_emb).astype('float32'), initial_k)
        
        candidates = []
        for idx in I[0]:
            if 0 <= idx < len(self.documents):
                candidates.append(self.documents[idx])
        
        # 2. Rerank if available
        reranker = self._get_reranker()
        if reranker and candidates:
            pairs = [[query, doc] for doc in candidates]
            scores = reranker.predict(pairs)
            # Zip and sort
            scored_candidates = sorted(zip(candidates, scores, I[0]), key=lambda x: x[1], reverse=True)
            # Select top_k
            final_selection = scored_candidates[:top_k]
        else:
            # Fallback
            final_selection = zip(candidates, [0]*len(candidates), I[0])
            final_selection = list(final_selection)[:top_k]

        results = []
        for content, score, original_idx in final_selection:
            results.append({
                "content": content,
                "meta": {
                    "source": self.data_path,
                    "chunk_id": int(original_idx),
                    "score": float(score) if reranker else 0.0
                }
            })
        return results

# Singleton instance
rag_engine = MiniRAG()

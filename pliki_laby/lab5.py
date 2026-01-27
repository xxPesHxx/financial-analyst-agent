import re, random, pandas as pd, time, faiss, numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOPICS = {
    "ai": [
        "Large language models predict the next token using transformer architectures.",
        "Embeddings map text into dense vectors enabling semantic search.",
        "RAG combines retrieval with generation to ground responses."
    ],
    "sport": [
        "Marathon training plans balance long runs and recovery days.",
        "Strength training improves running economy and power.",
        "Interval sessions develop speed and lactate threshold."
    ],
    "cooking": [
        "Sourdough starter needs regular feeding to stay active.",
        "Sous vide cooking keeps precise temperatures for tenderness.",
        "Spices bloom in hot oil enhancing aroma and flavor."
    ],
    "geo": [
        "Rivers shape valleys through erosion and sediment transport.",
        "Plate tectonics explains earthquakes and mountain building.",
        "Deserts form where evaporation exceeds precipitation."
    ],
    "health": [
        "Sleep supports memory consolidation and hormonal balance.",
        "Aerobic exercise benefits cardiovascular health and VO2 max.",
        "Protein intake supports muscle repair and satiety."
    ]
}

SEED = 42
random.seed(SEED)

def synth_docs(n_per_topic=40):
    docs = []
    for topic, seed in TOPICS.items():
        for i in range(n_per_topic):
            base = random.choice(seed)
            noise = random.choice(seed)
            txt = f"{base} {noise} ({topic} #{i})"
            docs.append({"doc_id": f"{topic}_{i}", "text": txt, "topic": topic})
    return docs

DOCS = synth_docs(40)

def simple_chunk(text, chunk_chars=280, overlap=40):
    out = []
    i = 0
    while i < len(text):
        j = min(len(text), i+chunk_chars)
        out.append((i, j, text[i:j]))
        if j == len(text): break
        i = max(0, j-overlap)
    return out


def build_chunks(docs, chunk_chars=280, overlap=40):
    rows = []
    for d in docs:
        for k,(a,b,txt) in enumerate(simple_chunk(d["text"], chunk_chars, overlap)):
            rows.append({"doc_id": d["doc_id"], "topic": d["topic"], "chunk_id": k, "start": a, "end": b, "chunk": txt})
    return pd.DataFrame(rows)

CHUNK_SIZE=280; OVERLAP=40
df = build_chunks(DOCS, CHUNK_SIZE, OVERLAP)

embedder = SentenceTransformer(MODEL_NAME)
def embed_texts(texts, batch_size=64):
    return embedder.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

chunks = df["chunk"].tolist()
t0=time.time()
embs = embed_texts(chunks, 64)
build_s = time.time()-t0

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())
bm25_corpus = [tokenize(c) for c in chunks]
bm25 = BM25Okapi(bm25_corpus)


def retrieve_dense(query: str, k: int=5):
    q = embed_texts([query], batch_size=1)
    scores, idxs = index.search(q, k)
    return [(float(scores[0][i]), df.iloc[idxs[0][i]].to_dict()) for i in range(k)]


def retrieve_bm25(query: str, k: int=5):
    toks = tokenize(query)
    scores = bm25.get_scores(toks)
    idxs = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), df.iloc[i].to_dict()) for i in idxs]


print(retrieve_dense("What improves running economy?", 3)[0])
print(retrieve_bm25("What improves running economy?", 3)[0])


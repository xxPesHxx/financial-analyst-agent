import csv
import os, math, re, random, pandas as pd, time, faiss, numpy as np
from datetime import datetime
import matplotlib.pyplot as plt # pip install matplotlib

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

GOLDEN = [
    ("How do transformers predict tokens?", "ai"),
    ("What is an embedding used for?", "ai"),
    ("How does RAG work?", "ai"),
    ("How to train for a marathon?", "sport"),
    ("What improves running economy?", "sport"),
    ("What is a threshold workout?", "sport"),
    ("How to feed sourdough starter?", "cooking"),
    ("Why sous vide is precise?", "cooking"),
    ("How to bloom spices?", "cooking"),
    ("How do rivers shape valleys?", "geo"),
    ("What causes earthquakes?", "geo"),
    ("Why do deserts form?", "geo"),
    ("Why is sleep important?", "health"),
    ("Benefits of aerobic exercise?", "health"),
    ("Why eat protein?", "health"),
]


def dcg(rels):
    return sum((rel / math.log2(i + 2) for i, rel in enumerate(rels)))


def ndcg_at_k(rels, k):
    rels_k = rels[:k]
    ideal = sorted(rels_k, reverse=True)
    denom = dcg(ideal) or 1e-9
    return dcg(rels_k) / denom


def eval_query(q, target_topic, retriever, k=5):
    hits = retriever(q, k=k)
    rels = [1 if h[1]["topic"] == target_topic else 0 for h in hits]
    rec = sum(rels) * 1.0
    prec = sum(rels) / len(rels) if rels else 0.0
    rr = 0.0
    for i, r in enumerate(rels, start=1):
        if r==1: rr = 1.0 / i; break
    ndcg = ndcg_at_k(rels, k)
    return {"recall@k": rec, "precision@k": prec, "mrr": rr, "ndcg@k": ndcg}

def evaluate(golden, retriever, k=5):
    rows = []
    for q, t in golden:
        rows.append({"query": q, "topic": t, **eval_query(q, t, retriever, k)})
    return pd.DataFrame(rows)

K = 5
dense_df = evaluate(GOLDEN, retrieve_dense, k=K)
b25_df = evaluate(GOLDEN, retrieve_bm25, k=K)

summary = pd.DataFrame({
    "metric": ["recall@k", "precision@k", "mrr", "ndcg@k"],
    "dense": [dense_df[m].mean() for m in ["recall@k", "precision@k", "mrr", "ndcg@k"]],
    "bm25": [b25_df[m].mean() for m in ["recall@k", "precision@k", "mrr", "ndcg@k"]],
})

print(summary)

def run_settings(chunk_size, overlap, kk):
    diff = build_chunks(DOCS, chunk_size, overlap)
    embs = embedder.encode(
        diff["chunk"].tolist(),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
    def retr(q, k):
        qv = embedder.encode(
            [q],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")
        scores, ids = idx.search(qv, k)
        return [(float(scores[0][i]), diff.iloc[ids[0][i]].to_dict()) for i in range(k)]
    dfres = evaluate(GOLDEN, retr, kk)
    return dfres[["recall@k", "precision@k", "mrr", "ndcg@k"]].mean().to_dict()

"""
grid = []
for cs in [50, 200]:
    for ov in [0, 80]:
        for kk in [3, 5]:
            met = run_settings(cs, ov,kk)
            grid.append({"chunk_size": cs, "overlap": ov, "k": kk, **met})

grid_df = pd.DataFrame(grid).sort_values(["k", "recall@k"], ascending=[True, False])
print(grid_df.head(10))

sub = grid_df[grid_df["k"] == 5].groupby("chunk_size")["recall@k"].mean()
plt.figure()
sub.plot(kind="bar")
plt.title("Recall@5 vs Chunk Size")
plt.xlabel("Chunk Size")
plt.ylabel("Recall@5")
plt.tight_layout()
plt.show()
"""

def log_row(path: str, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

log_row("lab10_logs.csv", {
    "timestamp": datetime.now().isoformat(),
    "model_name": MODEL_NAME,
    "index_type": "faiss_flat_ip",
    "k": K,
    "chunk_size": CHUNK_SIZE,
    "overlap": OVERLAP,
    "build_time_sec": build_s,
    "num_chunks": len(df),
    "recall@k_dense": float(summary.loc[summary["metric"]=="recall@k", "dense"].values[0]),
    "recall@k_bm25": float(summary.loc[summary.metric=="recall@k", "bm25"].values[0]),
})






























# pip install pypdf
import os, time, glob, re, faiss, pandas as pd, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv
from typing import Any, Dict, Optional, List, Tuple

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
tokenizer = model = None

SYSTEM_RULES = (
    "You are a factual assistant. Answer ONLY using the provided context snippets. "
    "If missing, reply 'Nie wiem - brak informacji w źródłach.' Include citations like [1], [2]."
)

DOCS = [
    {'id':'d1','text':'Embeddings are vector representations of text used for semantic search.'},
    {'id':'d2','text':'BM25 is a bag-of-words retrieval algorithm based on term frequency.'},
    {'id':'d3','text':'RAG combines retrieval and generation to ground LLM outputs.'},
    {'id':'d4','text':'Cross-encoders score query+doc pairs with a deeper transformer for reranking.'},
    {'id':'d5','text':'Embedding models like all-MiniLM produce compact vectors.'},
]

client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
print("Using Gemini LLM:", GEMINI_MODEL)

def chat_once_gemini(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.3,
    top_p: float = 0.95,
    top_k: int = 40,
    max_output_tokens: int = 1024,
) -> Dict[str, Any]:
    config = genai.types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
        stop_sequences=["<END>"],
    )

    t0 = time.time()
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=config,
    )
    dt = round(time.time() - t0, 3)

    usage = getattr(resp, "usage_metadata", None)
    usage_dict: Optional[Dict[str, Any]] = None
    if usage is not None:
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "completion_tokens": getattr(usage, "candidates_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
            "tool_use_prompt_tokens": getattr(usage, "tool_use_prompt_token_count", None),
            "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
        }

    return {
        "text": getattr(resp, "text", str(resp)),
        "latency_s": dt,
        "usage": usage_dict
    }

def llm_call(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.3,
    top_p: float = 0.95,
    top_k: int = 40,
    max_output_tokens: int = 1024,
) -> Dict[str, Any]:
    return chat_once_gemini(
        prompt,
        system=system,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens
    )

def load_pdf(path):
    rows = []; r=PdfReader(path)
    for i, p in enumerate(r.pages, start=1):
        try:
            text = p.extract_text() or ""
        except Exception:
            text = ""
        rows.append({
            "source": os.path.basename(path),
            "page": i,
            "text": text
        })
    return rows

def load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{
        "source": os.path.basename(path),
        "page": 1,
        "text": text
    }]

def load_md(path):
    return load_txt(path)

def load_corpus(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    rows = []
    for fp in glob.glob(os.path.join(data_dir, "*")):
        l = fp.lower()
        if l.endswith(".pdf"): rows += load_pdf(fp)
        elif l.endswith(".txt"): rows += load_txt(fp)
        elif l.endswith(".md"): rows += load_md(fp)

    if not rows:
        rows = [{
            "source": "demo.md",
            "page": 1,
            "text": "RAG łączy retrieval z generacją."
        }, {
            "source": "demo.md",
            "page": 2,
            "text": "Embeddingi to wektory semantyczne, podobieństwo kosinusowe."
        }]

    return pd.DataFrame(rows)

def simple_chunk(text, chunk_size=500, overlap=50):
    out = []; i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        out.append((i, j, text[i:j]))
        if j == len(text): break
        i = max(0, j - overlap)
    return out

def make_chunks(df, chunk_chars=800, overlap=120):
    rows = []
    for _, r in df.iterrows():
        for k, (a, b, txt) in enumerate(simple_chunk(r["text"], chunk_chars, overlap)):
            if txt.strip():
                rows.append({
                    "source": r["source"],
                    "page": r["page"],
                    "chunk_id": k + 1,
                    "chunk": txt.strip(),
                    "start": a,
                    "end": b
                })
    return pd.DataFrame(rows)

docs_df = load_corpus('./data')
chunks_df = make_chunks(docs_df, 800, 120)
print(f"Loaded {len(docs_df)} documents, {len(chunks_df)} chunks")

EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformer(EMB_MODEL)

def embed_texts(texts, batch_size=32):
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    return embeddings
embs = embed_texts(chunks_df["chunk"].tolist())
index = faiss.IndexFlatIP(embs.shape[1]); index.add(embs)
print(f"FAISS index: {index.ntotal} vectors of dimension {embs.shape[1]}")

def tok(x): return re.findall(r"[a-ząćęłńóśźż0-9]+", x.lower())
bm25_corpus = [tok(c) for c in chunks_df["chunk"].tolist()]
bm25 = BM25Okapi(bm25_corpus)
print("BM25 index created.")


def retrieve_dense(query, k=5):
    qv = embed_texts([query], batch_size=1)
    scores, idxs = index.search(qv, k)
    return [(float(scores[0][i]), chunks_df.iloc[idxs[0][i]].to_dict()) for i in range(k)]

def pack_context(hits, max_per_source=2, max_chars=2000):
    per = {}; ordered = []
    for _, rec in hits:
        key = (rec["source"], rec["page"])
        per.setdefault(key, 0)
        if per[key] < max_per_source:
            per[key] += 1
            ordered.append(rec)
    cites = []; parts = []
    for i, rec in  enumerate(ordered, start=1):
        cites.append({
            "n": i,
            "source": rec["source"],
            "page": rec["page"],
            "chunk_id": rec["chunk_id"]
        })
        parts.append(f"[{i}] {rec['chunk']}")
    ctx = "\n\n".join(parts)
    return (ctx[:max_chars], cites)

def answer_with_api(question, hits):
    ctx, cites = pack_context(hits)
    prompt = "Question: " + question + "\n\nContext:\n" + ctx + "\n\nAnswer in Polish with citations [n]."
    return llm_call(prompt, system=SYSTEM_RULES, max_output_tokens=512, temperature=0.0), cites

print(answer_with_api("Jakie ery wyróżniamy?", retrieve_dense('ery', k=5)))
print(answer_with_api("Co to jest RAG?", retrieve_dense('RAG', k=5)))

# -------------- Od tego miejsca reraking z cross-encoderem --------------

texts = [d['text'] for d in DOCS]
embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
bm25_corpus = [tok(t) for t in texts]
bm25 = BM25Okapi(bm25_corpus)

def retrieve_rerank(query, k=5):
    qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idxs = idx.search(qv, k)
    return [(float(scores[0][i]), DOCS[idxs[0][i]]) for i in range(min(k, len(idxs[0])))]

USE_CROSS_ENCODER = False
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    USE_CROSS_ENCODER = True
except Exception:
    CROSS_ENCODER = None
    USE_CROSS_ENCODER = False

def rerank(query: str, candidates: List[Tuple[float, Dict[str, Any]]]) -> List[Tuple[float, Dict[str, Any]]]:
    if USE_CROSS_ENCODER and CROSS_ENCODER is not None:
        pairs = [[query, c[1]['text']] for c in candidates]
        scores = CROSS_ENCODER.predict(pairs)
        scored = [(float(s), c[1]) for s, c in zip(scores, candidates)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0]
    out = []
    for score, doc in candidates:
        idx_doc = next((i for i, d in enumerate(DOCS) if d['id'] == doc['id']), None)
        if idx_doc is not None:
            s = score
        else:
            s = float(np.dot(qv, embs[idx_doc]))
        out.append((s, doc))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

query = "What are embeddings used for?"
print('BM25 top-3 results:')
bm = bm25.get_scores(tok(query));
bm_idx = np.argsort(bm)[::-1][:3]
for i in bm_idx:
    print(f"Score: {bm[i]:.4f}, Texts: {texts[i]}")

print('\nDense retrieval top-3 results:')
cand = retrieve_rerank(query, k=3)
for s, d in cand:
    print(f"Score: {s:.4f}, Texts: {d['text']}")

print('\nReranked results:')
rr = rerank(query, cand)
for s, d in rr:
    print(f"Score: {s:.4f}, Texts: {d['text']}")

GOLD = [
    ('what are embeddings used for?', {'relevant_ids':['d1','d5']}),
    ('how does RAG work?', {'relevant_ids':['d3']}),
]

def recall_at_k(ranked_docs, relevant_ids, k=3):
    topk = [d['id'] for _, d in ranked_docs[:k]]
    return len(set(topk) & set(relevant_ids)) / len(relevant_ids)

def evaluate_rerank():
    rows = []
    for q, meta in GOLD:
        baseline = retrieve_rerank(q, k=5)
        baseline_recall = recall_at_k(baseline, meta['relevant_ids'], k=3)
        reranked = rerank(q, baseline)
        rerank_recall = recall_at_k(reranked, meta['relevant_ids'], k=3)
        rows.append({
            "query": q,
            "baseline_recall@3": baseline_recall,
            "rerank_recall@3": rerank_recall
        })
    return pd.DataFrame(rows)

print(evaluate_rerank())
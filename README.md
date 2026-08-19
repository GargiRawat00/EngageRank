# EngageRank

Personalized news recommendation system using semantic retrieval,
multi-interest ranking, diversity-aware reranking, cold-start handling,
and grounded RAG briefings.

## Architecture

User History
↓
SentenceTransformer Embeddings
↓
FAISS Candidate Retrieval
↓
Top-3 Historical Similarity Ranking
↓
Top-20 Shortlist
↓
MMR Diversity Reranking
↓
Top Recommendations
↓
Groq RAG Briefing
↓
FastAPI
↓
React Frontend

## Dataset

Microsoft MIND-small

## Retrieval

- SentenceTransformer: all-MiniLM-L6-v2
- 384-dimensional normalized article embeddings
- FAISS IndexFlatIP
- filters already-clicked articles

## Ranking

- candidate compared against all historical clicks
- top-3 similarity aggregation
- MMR reranking with lambda = 0.9

## Cold Start

- training-set click popularity fallback
- personalized ranking used once history is available

## Evaluation

Baseline mean-history:
MRR: 0.3543
NDCG@5: 0.3374
NDCG@10: 0.3969
Recall@5: 0.4754
Recall@10: 0.6450

Top-3 history model:
MRR: 0.3546
NDCG@5: 0.3376
NDCG@10: 0.3980
Recall@5: 0.4759
Recall@10: 0.6480

Diversity experiment on 4,866 validation impressions:
Top-3 ILD@10: 0.9040
Top-3 + MMR ILD@10: 0.9127

## Experiments

- Mean-history baseline
- Recent-click weighting → degraded metrics
- Max historical similarity → degraded metrics
- Top-3 historical aggregation → small improvement
- Timestamped session weighting → degraded metrics
- MMR → improved diversity with small relevance tradeoff

## API

GET /demo-users
GET /demo-user/{user_id}
POST /recommend
POST /briefing

## Running

Backend:
source venv/Scripts/activate
uvicorn api.main:app --reload

Frontend:
cd frontend
npm install
npm run dev

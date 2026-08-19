# EngageRank Architecture

EngageRank is a personalized news recommendation system that combines semantic retrieval, multi-interest ranking, diversity-aware reranking, cold-start handling, and a grounded RAG briefing layer.

The system is designed as a multi-stage recommendation pipeline instead of using a single model for everything.

---

## 1. High-Level Architecture

```text
User / Demo User
        ↓
User Click History
        ↓
SentenceTransformer Embeddings
        ↓
User Interest Representation
        ↓
FAISS Candidate Retrieval
        ↓
Top-3 Historical Similarity Ranking
        ↓
Top-20 Candidate Shortlist
        ↓
MMR Diversity Reranking
        ↓
Final Top Recommendations
        ↓
FastAPI
        ↓
React Frontend
        ↓
Optional RAG News Briefing
```

---

## 2. Dataset

EngageRank uses the Microsoft MIND-small news recommendation dataset.

The main files used are:

```text
data/raw/train/news.tsv
data/raw/train/behaviors.tsv

data/raw/valid/news.tsv
data/raw/valid/behaviors.tsv
```

The `news.tsv` file contains information about news articles such as:

- News ID
- Category
- Subcategory
- Title
- Abstract
- URL
- Title entities
- Abstract entities

The `behaviors.tsv` file contains user interaction information such as:

- Impression ID
- User ID
- Timestamp
- Previously clicked article history
- Candidate impressions
- Click labels

Example:

```text
History:
N100 N200 N300

Current impressions:
N400-0 N500-1 N600-0
```

Here:

```text
1 = clicked
0 = not clicked
```

---

## 3. Data Preprocessing

The preprocessing stage loads the news and behavior files and converts them into a format that can be used by the recommendation pipeline.

The system combines training and validation news articles and removes duplicate articles using their `news_id`.

Each article is represented using information from its title and abstract.

Conceptually:

```text
News Article
     ↓
Title + Abstract
     ↓
Text Representation
```

The processed data is later used to generate article embeddings.

---

## 4. Article Embeddings

EngageRank uses the SentenceTransformer model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Each article is converted into a dense semantic vector.

The model generates:

```text
384-dimensional embeddings
```

Example:

```text
Article
   ↓
SentenceTransformer
   ↓
[0.12, -0.31, 0.08, ...]
```

The final article embedding matrix contains approximately:

```text
65,238 articles × 384 dimensions
```

The embeddings are normalized.

Because of this normalization, the inner product between two vectors behaves like cosine similarity.

The generated files are stored inside:

```text
data/processed/article_embeddings.npy
data/processed/article_ids.csv
```

---

## 5. FAISS Index

Searching all articles individually for every user would be inefficient.

Therefore, EngageRank uses FAISS for fast semantic retrieval.

The system creates a FAISS index using:

```text
faiss.IndexFlatIP
```

`IP` stands for Inner Product.

Because article embeddings are normalized:

```text
Inner Product ≈ Cosine Similarity
```

The FAISS index contains all article embeddings.

It is stored as:

```text
data/processed/article_index.faiss
```

FAISS is used only for candidate retrieval.

It does not perform the final personalized ranking.

---

## 6. User Representation

For users with historical clicks, EngageRank builds an initial representation of their interests.

Suppose a user previously clicked:

```text
N100
N200
N300
N400
```

The system retrieves the embeddings of these articles:

```text
N100 → embedding
N200 → embedding
N300 → embedding
N400 → embedding
```

The embeddings are averaged to form an initial user vector.

```text
Clicked Article Embeddings
           ↓
          Mean
           ↓
      User Vector
```

The user vector is normalized before being sent to FAISS.

---

## 7. Candidate Retrieval

The user vector is searched against the FAISS index.

Conceptually:

```text
User Vector
    ↓
FAISS Index
    ↓
Semantically Similar Articles
```

Instead of running detailed ranking over all 65,000+ articles, FAISS retrieves approximately:

```text
Top 100 candidate articles
```

Articles that the user has already clicked are removed from the candidate set.

Therefore:

```text
65,238 articles
      ↓
FAISS retrieval
      ↓
~100 candidates
```

This significantly reduces the amount of work required by the personalized ranking stage.

---

## 8. Personalized Ranking

The FAISS stage uses an averaged user representation.

However, averaging the complete history can hide individual user interests.

For example, a user may be interested in:

```text
Technology
Sports
Politics
Finance
```

A single average vector may combine all these interests into one representation.

To handle this, EngageRank performs candidate-aware ranking.

For every candidate article:

```text
Candidate Article
       ↓
Compare against every historical article
       ↓
Cosine similarity scores
```

For example:

```text
Candidate vs History

0.18
0.42
0.71
0.64
0.09
0.31
```

The system selects the strongest three similarities:

```text
0.71
0.64
0.42
```

The final candidate relevance score becomes:

```text
relevance_score =
mean(top 3 historical similarities)
```

In this example:

```text
(0.71 + 0.64 + 0.42) / 3
```

This allows a candidate to match one of the user's major interests without requiring it to match the average of the entire history.

---

## 9. Ranking Experiments

Multiple approaches were evaluated before selecting the final ranking strategy.

### 9.1 Mean-History Baseline

The baseline model averaged all historical article embeddings into one user vector.

Validation results:

```text
MRR       = 0.3543
NDCG@5    = 0.3374
NDCG@10   = 0.3969
Recall@5  = 0.4754
Recall@10 = 0.6450
```

---

### 9.2 Recent-History Weighting

A short-term interest vector was created using the latest few items in the supplied history.

The model combined:

```text
Long-Term User Representation
            +
Short-Term User Representation
```

This approach reduced ranking performance.

Therefore it was not included in the final system.

---

### 9.3 Maximum Historical Similarity

Each candidate was scored using only its highest similarity with one historical article.

Conceptually:

```text
score =
max(candidate similarity with history)
```

This also reduced ranking quality.

A single accidental similarity could dominate the score.

Therefore this approach was rejected.

---

### 9.4 Top-3 Historical Similarity

The strongest three similarities were averaged.

Conceptually:

```text
Candidate
    ↓
Similarity against user history
    ↓
Top 3 similarities
    ↓
Average
    ↓
Relevance Score
```

Full validation results:

```text
MRR       = 0.3546
NDCG@5    = 0.3376
NDCG@10   = 0.3980
Recall@5  = 0.4759
Recall@10 = 0.6480
```

This produced a small improvement over the mean-history baseline and became the final relevance ranking method.

---

## 10. Timestamp-Aware Session Experiment

The validation behavior data contains timestamps for user impression events.

An additional experiment reconstructed earlier validation clicks for a user and treated them as short-term session interests.

The model combined:

```text
Long-Term History
       +
Previous Session Clicks
```

Different session weights were tested:

```text
0.00
0.25
0.50
0.75
```

Performance decreased as the short-term session weight increased.

Therefore timestamp-based session weighting was rejected from the final recommendation pipeline.

This was kept as an experiment rather than being forced into the production system.

---

## 11. Diversity Problem

Ranking only by relevance can create repetitive recommendation lists.

For example:

```text
NVIDIA launches new GPU
NVIDIA GPU benchmark results
NVIDIA GPU pricing announced
NVIDIA GPU review
NVIDIA GPU performance analysis
```

These articles may all be relevant, but the final recommendation list lacks variety.

EngageRank addresses this using Maximal Marginal Relevance.

---

## 12. MMR Diversity Reranking

MMR stands for:

```text
Maximal Marginal Relevance
```

It balances:

```text
Relevance
    +
Diversity
```

Conceptually:

```text
MMR Score =
λ × relevance
-
(1 - λ) × redundancy
```

EngageRank uses:

```text
λ = 0.9
```

This means the system gives stronger importance to relevance while still penalizing highly repetitive recommendations.

Approximately:

```text
90% relevance
10% diversity
```

The final flow becomes:

```text
100 FAISS Candidates
        ↓
Top-3 Similarity Ranking
        ↓
Top 20 Candidates
        ↓
MMR Reranking
        ↓
Final Top 10
```

MMR compares a candidate with articles already selected in the output list.

If a candidate is too similar to an already selected article, its MMR score decreases.

---

## 13. Diversity Evaluation

EngageRank measures recommendation diversity using:

```text
ILD@10
```

ILD stands for:

```text
Intra-List Diversity
```

It measures the average semantic distance between articles in a recommendation list.

Experiment results:

```text
Top-3 Ranking

MRR       = 0.3644
NDCG@5    = 0.3468
NDCG@10   = 0.4054
Recall@5  = 0.4842
Recall@10 = 0.6500
ILD@10    = 0.9040
```

After MMR:

```text
Top-3 + MMR

MRR       = 0.3634
NDCG@5    = 0.3451
NDCG@10   = 0.4036
Recall@5  = 0.4816
Recall@10 = 0.6482
ILD@10    = 0.9127
```

The system therefore accepts a small relevance reduction in exchange for better recommendation diversity.

---

## 14. Cold-Start Problem

A new user has no historical clicks.

Therefore the system cannot create:

```text
User History
     ↓
User Embedding
```

Without a user representation, personalized FAISS retrieval cannot operate normally.

EngageRank handles this using a popularity-based fallback.

---

## 15. Cold-Start Recommendation

Popularity is calculated only from clicks in the training behavior data.

Conceptually:

```text
Training Behaviors
       ↓
Count Positive Clicks
       ↓
Article Popularity
```

The most popular articles become initial candidate recommendations.

MMR is then applied so the resulting popular list does not become excessively repetitive.

The decision flow is:

```text
              User
               ↓
        Has click history?
          /          \
        Yes           No
        ↓             ↓
     FAISS         Popularity
        ↓             ↓
 Personalized      Cold Start
    Ranking        Ranking
        ↓             ↓
        └──────┬──────┘
               ↓
              MMR
               ↓
      Final Recommendations
```

Training data is used for popularity calculation so validation click labels are not leaked into cold-start ranking.

---

## 16. Final Recommendation Pipeline

For an existing user:

```text
User History
     ↓
Article Embeddings
     ↓
Mean User Vector
     ↓
FAISS Retrieval
     ↓
Top 100 Candidates
     ↓
Remove Previously Clicked Articles
     ↓
Candidate vs History Similarities
     ↓
Top-3 Similarity Aggregation
     ↓
Relevance Ranking
     ↓
Top 20 Candidates
     ↓
MMR Diversity Reranking
     ↓
Top 10 Recommendations
```

For a new user:

```text
No History
    ↓
Training Click Popularity
    ↓
Popular Candidates
    ↓
Top 20
    ↓
MMR
    ↓
Top 10 Recommendations
```

---

## 17. RAG Briefing Layer

RAG is an additional layer on top of the recommendation system.

It does not decide which articles should be recommended.

The recommendation engine first generates the final personalized articles.

Then the RAG pipeline receives those articles.

```text
Recommendation Engine
        ↓
Top Recommended Articles
        ↓
Title + Category + Abstract
        ↓
Prompt Context
        ↓
Groq LLM
        ↓
Personalized News Briefing
```

The prompt instructs the LLM to use only information present in:

```text
Article Title
Article Category
Article Abstract
```

The model is instructed not to use outside information or invent missing facts.

The current RAG layer uses:

```text
Groq API
openai/gpt-oss-20b
```

The result is a concise briefing based on the recommended articles.

---

## 18. Why RAG Is Separate From Ranking

The recommendation engine and the LLM solve different problems.

The recommendation system answers:

```text
Which articles should this user see?
```

The RAG layer answers:

```text
How can these recommended articles be summarized for the user?
```

Therefore the architecture keeps them separate.

```text
Recommendation
    ↓
Retrieval + Ranking + Reranking

Generation
    ↓
RAG + LLM
```

This makes the recommendation quality independent of the LLM.

---

## 19. FastAPI Backend

The recommendation and briefing pipelines are exposed through FastAPI.

Main API file:

```text
api/main.py
```

Available endpoints include:

```text
GET /
```

Checks whether the API is running.

```text
GET /demo-users
```

Returns demo users with available histories.

```text
GET /demo-user/{user_id}
```

Returns the stored history for a selected demo user.

```text
POST /recommend
```

Generates personalized or cold-start recommendations.

```text
POST /briefing
```

Generates an AI briefing based on the recommended articles.

---

## 20. Recommendation API Flow

```text
React Frontend
      ↓
POST /recommend
      ↓
FastAPI
      ↓
recommend(history)
      ↓
Has History?
   /       \
 Yes        No
  ↓          ↓
FAISS    Popularity
  ↓          ↓
Ranking      |
  ↓          |
  └────┬─────┘
       ↓
      MMR
       ↓
Top Recommendations
       ↓
JSON Response
       ↓
React UI
```

A recommendation response contains information such as:

```text
news_id
title
category
score
recommendation mode
```

The recommendation mode can be:

```text
personalized
```

or:

```text
cold_start
```

---

## 21. Briefing API Flow

The briefing endpoint uses the same recommendation engine.

```text
React Frontend
      ↓
POST /briefing
      ↓
FastAPI
      ↓
Recommendation Engine
      ↓
Top 5 Recommendations
      ↓
Retrieve Title + Abstract + Category
      ↓
Build Grounded Prompt
      ↓
Groq LLM
      ↓
News Briefing
      ↓
JSON Response
      ↓
React UI
```

This ensures that the AI briefing describes articles that were actually selected by EngageRank.

---

## 22. React Frontend

The frontend is built using:

```text
React
Vite
```

Users can select predefined demo users.

For a known user:

```text
Demo User
    ↓
Historical Clicks
    ↓
Personalized Recommendations
```

The interface displays:

```text
Recommendation Rank
Article Category
Article Title
Recommendation Score
Recommendation Mode
```

The user can also generate an AI personalized briefing from the selected recommendations.

---

## 23. New User Frontend Flow

The frontend also includes a new-user option.

```text
New User
   ↓
Empty History
   ↓
POST /recommend
   ↓
Cold-Start Pipeline
   ↓
Popularity + MMR
   ↓
Trending Recommendations
```

This demonstrates that EngageRank supports both:

```text
Known Users
New Users
```

---

## 24. Evaluation Metrics

The recommendation models are evaluated using multiple metrics.

### MRR

```text
Mean Reciprocal Rank
```

Measures how early the first relevant article appears.

Higher is better.

---

### NDCG@K

```text
Normalized Discounted Cumulative Gain
```

Measures ranking quality by rewarding relevant articles placed near the top.

The project evaluates:

```text
NDCG@5
NDCG@10
```

---

### Recall@K

Measures how many relevant articles were retrieved within the first K recommendations.

The project evaluates:

```text
Recall@5
Recall@10
```

---

### ILD@10

```text
Intra-List Diversity
```

Measures semantic diversity among the top 10 recommendations.

Higher ILD means the recommendation list contains less redundant content.

---

## 25. Technology Stack

### Machine Learning

```text
Python
PyTorch
SentenceTransformers
NumPy
Pandas
```

### Embedding Model

```text
all-MiniLM-L6-v2
```

### Retrieval

```text
FAISS
IndexFlatIP
```

### Ranking

```text
Cosine Similarity
Top-3 Historical Similarity Aggregation
```

### Diversity

```text
Maximal Marginal Relevance
```

### Evaluation

```text
MRR
NDCG@5
NDCG@10
Recall@5
Recall@10
ILD@10
```

### Backend

```text
FastAPI
Uvicorn
Pydantic
```

### Generative AI

```text
Groq API
GPT-OSS
RAG
```

### Frontend

```text
React
Vite
JavaScript
CSS
```

---

## 26. Repository Architecture

```text
EngageRank/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── valid/
│   │
│   └── processed/
│       ├── article_embeddings.npy
│       ├── article_ids.csv
│       └── article_index.faiss
│
├── docs/
│   └── architecture.md
│
├── frontend/
│   └── src/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   │
│   ├── data/
│   │   └── preprocess.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   │
│   ├── models/
│   │   ├── popularity.py
│   │   ├── embeddings.py
│   │   └── cold_start.py
│   │
│   ├── ranking/
│   │   ├── diversity.py
│   │   └── recommender.py
│   │
│   ├── retrieval/
│   │   ├── faiss_index.py
│   │   ├── search.py
│   │   └── user_retrieval.py
│   │
│   └── rag/
│       ├── briefing.py
│       └── generator.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 27. Complete System Architecture

```text
                         MIND Dataset
                              ↓
                      Data Preprocessing
                              ↓
                    Article Title + Abstract
                              ↓
                      SentenceTransformer
                              ↓
                     384-D Embeddings
                              ↓
                         FAISS Index
                              ↓
                    ┌─────────┴─────────┐
                    │                   │
                 Known User          New User
                    │                   │
              Click History             │
                    ↓                   │
               User Vector              │
                    ↓                   │
             FAISS Retrieval       Training Click
                    ↓               Popularity
              ~100 Candidates            │
                    ↓                   │
         Remove Previously Clicked       │
                    ↓                   │
         Candidate-History Similarity    │
                    ↓                   │
           Top-3 Similarity Mean         │
                    ↓                   │
             Relevance Ranking           │
                    ↓                   ↓
                  Top 20 Candidates
                         ↓
                    MMR Reranking
                         ↓
                Top 10 Recommendations
                         ↓
                 ┌───────┴────────┐
                 │                │
              FastAPI       RAG Context Builder
                 │                │
                 │          Top Recommended
                 │             Articles
                 │                ↓
                 │       Title + Category +
                 │             Abstract
                 │                ↓
                 │             Groq LLM
                 │                ↓
                 │       Personalized Briefing
                 │                │
                 └───────┬────────┘
                         ↓
                   React Frontend
                         ↓
              User Recommendations
                  + AI Briefing
```

---

## 28. Key Design Decisions

EngageRank separates the recommendation problem into independent stages.

```text
Retrieval
    ↓
Ranking
    ↓
Reranking
    ↓
Generation
```

Each component has a specific responsibility.

### FAISS

```text
Efficiently reduces the article search space.
```

### Top-3 Historical Similarity

```text
Captures multiple user interests instead of relying only on one averaged user representation.
```

### MMR

```text
Reduces redundant recommendations while preserving relevance.
```

### Cold-Start Popularity

```text
Allows recommendations when no user history exists.
```

### RAG

```text
Transforms recommended articles into a readable personalized briefing without controlling the recommendation ranking itself.
```

---

## 29. Final Design Principle

The central design principle of EngageRank is:

```text
Retrieve broadly
      ↓
Rank personally
      ↓
Diversify carefully
      ↓
Generate explanations separately
```

This modular architecture makes each stage independently testable and allows retrieval, ranking, diversity, and generation methods to be improved without redesigning the complete system.

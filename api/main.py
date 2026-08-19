from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ranking.recommender import recommend, behaviors
from src.rag.generator import generate_briefing


app = FastAPI(
    title="EngageRank API",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    history: list[str]
    top_k: int = 10


class BriefingRequest(BaseModel):
    history: list[str]
    top_k: int = 5


@app.get("/")
def root():
    return {
        "message": "EngageRank API is running"
    }


# -----------------------------------
# Get a few valid demo users
# -----------------------------------

@app.get("/demo-users")
def get_demo_users():

    # only users who have some history
    demo_rows = behaviors[
        behaviors["history"].apply(
            lambda x: len(x) > 0
        )
    ]

    # keep one row per user
    demo_rows = demo_rows.drop_duplicates(
        subset=["user_id"]
    )

    # choose first 5 demo users
    demo_rows = demo_rows.head(5)

    users = []

    for _, row in demo_rows.iterrows():

        users.append(
            {
                "user_id": row["user_id"],
                "history_size": len(
                    row["history"]
                )
            }
        )

    return users


# -----------------------------------
# Get selected user's history
# -----------------------------------

@app.get("/demo-user/{user_id}")
def get_demo_user(user_id: str):

    user_rows = behaviors[
        behaviors["user_id"] == user_id
    ]

    # keep only rows with history
    user_rows = user_rows[
        user_rows["history"].apply(
            lambda x: len(x) > 0
        )
    ]

    if len(user_rows) == 0:
        return {
            "user_id": user_id,
            "history": [],
            "history_size": 0
        }

    history = user_rows.iloc[0]["history"]

    return {
        "user_id": user_id,
        "history": history,
        "history_size": len(history)
    }


# -----------------------------------
# Recommendation endpoint
# -----------------------------------

@app.post("/recommend")
def get_recommendations(
    request: RecommendRequest
):

    recommendations = recommend(
        request.history,
        final_k=request.top_k
    )

    return {
        "mode": (
            recommendations[0]["mode"]
            if recommendations
            else "none"
        ),

        "count": len(
            recommendations
        ),

        "recommendations": recommendations
    }


# -----------------------------------
# RAG briefing endpoint
# -----------------------------------

@app.post("/briefing")
def get_briefing(
    request: BriefingRequest
):

    briefing = generate_briefing(
        request.history,
        top_k=request.top_k
    )

    mode = (
        "personalized"
        if len(request.history) > 0
        else "cold_start"
    )

    return {
        "mode": mode,
        "briefing": briefing
    }
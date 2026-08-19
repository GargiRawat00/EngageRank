from fastapi import FastAPI
from pydantic import BaseModel

from src.ranking.recommender import recommend
from src.rag.generator import generate_briefing


app = FastAPI(
    title="EngageRank API",
    version="1.0"
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
        "count": len(recommendations),
        "recommendations": recommendations
    }


@app.post("/briefing")
def get_briefing(
    request: BriefingRequest
):
    briefing = generate_briefing(
        request.history,
        top_k=request.top_k
    )

    return {
        "briefing": briefing
    }
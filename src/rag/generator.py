import os

from dotenv import load_dotenv
from groq import Groq

from src.rag.briefing import build_briefing_context


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def build_prompt(history, top_k=5):

    articles = build_briefing_context(
        history,
        top_k=top_k
    )

    context_text = ""

    for i, article in enumerate(
        articles,
        start=1
    ):
        context_text += f"""
Article {i}
Title: {article["title"]}
Category: {article["category"]}
Abstract: {article["abstract"]}
"""

    prompt = f"""
Create a concise personalized news briefing from the articles below.

STRICT RULES:
- Use only facts explicitly present in the title, category, or abstract.
- Do not use outside knowledge.
- Do not infer missing details.
- Do not invent facts.
- Do not assume numbers, identities, causes, consequences, or relationships.
- Do not combine separate facts into a new factual claim unless the relationship is explicitly stated.
- Do not add league names, sport types, locations, organizations, or classifications unless explicitly present.
- Summarize ALL provided articles.
- Write at most 2-3 concise sentences for each article.
- If an article has very little information, keep its summary short.
- Use the original article title, or a shortened version of it, as the heading.
- Return plain text only.
- Do not use Markdown symbols such as **, *, #, or bullet markers.
- Never stop after summarizing only one article.

ARTICLES:
{context_text}

Write the complete briefing for all {len(articles)} articles.
"""

    return prompt


def generate_briefing(
    history,
    top_k=5
):

    prompt = build_prompt(
        history,
        top_k=top_k
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        reasoning_effort="low",

        temperature=0.2,

        max_completion_tokens=1500,

        stream=False
    )

    briefing = (
        response
        .choices[0]
        .message
        .content
    )

    return briefing


if __name__ == "__main__":

    history = [
        "N55189",
        "N42782",
        "N34694",
        "N45794",
        "N18445"
    ]

    briefing = generate_briefing(
        history,
        top_k=5
    )

    print(
        "\nPERSONALIZED BRIEFING\n"
    )

    print(briefing)
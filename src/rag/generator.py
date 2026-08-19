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
You are generating a personalized news briefing.

Use ONLY the information explicitly present in the provided articles.

STRICT RULES:
- Do not invent any facts.
- Do not infer missing information.
- Do not add background knowledge.
- Do not assume numbers, identities, causes, consequences, or relationships.
- Never convert vague or plural wording into a specific number unless that number is explicitly stated.
- Do not add league names, sport types, locations, organizations, or classifications unless they are explicitly present in the title, category, or abstract.
- When multiple facts appear in the title and abstract, do not combine them into a new factual statement unless that relationship is explicitly stated.
- If an article provides very little information, write a shorter summary instead of filling in gaps.
- Summarize every provided article.
- Keep each article summary concise.
- Preserve important names, numbers, and facts exactly as supported by the context.
- Do not mention anything that cannot be directly supported by the title, category, or abstract.
- Use the article title, or a shortened version of it, as the heading.
- Do not create new factual connections between separate statements.

ARTICLES:
{context_text}

Generate a concise personalized news briefing.
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
                "role": "system",
                "content": (
                    "You are a strictly grounded personalized news briefing assistant. "
                    "Use only facts explicitly stated in the supplied article titles, "
                    "categories, and abstracts. Do not infer, assume, classify, combine, "
                    "or add missing details. Do not use outside knowledge. "
                    "If the context is limited, keep the summary limited."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        # deterministic / less creative
        temperature=0.0,

        # enough space to summarize all retrieved articles
        max_tokens=1000
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


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

    print("\nPERSONALIZED BRIEFING\n")
    print(briefing)
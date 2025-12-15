import os
import pandas as pd
import numpy as np
import re
import argparse
import json

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from openai import OpenAI

# ============================================================
# Client
# ============================================================

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ============================================================
# Helper Functions
# ============================================================

def clean_text(text):
    """Very light cleaning for short survey comments."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())


def get_embeddings(texts, model="text-embedding-3-large"):
    """Return embeddings for a list of strings."""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]


def llm_label_cluster(comments):
    """Generate a short theme label using structured output."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a People Analytics expert."},
            {
                "role": "user",
                "content": (
                    "Given the following employee survey comments, "
                    "return a concise 2–4 word theme label.\n\n"
                    f"Comments:\n{comments}"
                )
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "cluster_theme",
                "schema": {
                    "type": "object",
                    "properties": {
                        "theme": {
                            "type": "string",
                            "description": "2–4 word theme label"
                        }
                    },
                    "required": ["theme"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )

    return json.loads(response.choices[0].message.content)["theme"]


def llm_summarize_cluster(comments):
    """Summarize sentiment, themes, and actions for a cluster (structured)."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a People Analytics expert."},
            {
                "role": "user",
                "content": (
                    "Summarize the following employee survey comments.\n\n"
                    "Return:\n"
                    "- 2–3 key themes\n"
                    "- Overall sentiment\n"
                    "- One recommended leadership action\n\n"
                    f"Comments:\n{comments}"
                )
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "cluster_summary",
                "schema": {
                    "type": "object",
                    "properties": {
                        "overall_sentiment": {
                            "type": "string",
                            "enum": ["Positive", "Neutral", "Negative"]
                        },
                        "themes": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "summary": {
                            "type": "string"
                        },
                        "recommended_action": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "overall_sentiment",
                        "themes",
                        "summary",
                        "recommended_action"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )

    return json.loads(response.choices[0].message.content)


def llm_global_summary(cluster_summaries):
    """Create a structured executive-level summary."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a People Analytics manager."},
            {
                "role": "user",
                "content": (
                    "Given the following cluster summaries, "
                    "produce an executive-level overview.\n\n"
                    f"{cluster_summaries}"
                )
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "executive_summary",
                "schema": {
                    "type": "object",
                    "properties": {
                        "overall_sentiment": {
                            "type": "string",
                            "enum": ["Positive", "Neutral", "Negative"]
                        },
                        "strengths": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "areas_of_concern": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "recommended_actions": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "narrative_summary": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "overall_sentiment",
                        "strengths",
                        "areas_of_concern",
                        "recommended_actions",
                        "narrative_summary"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )

    return json.loads(response.choices[0].message.content)

# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(input_csv, n_clusters=6):
    print("\n🚀 Loading data...")
    df = pd.read_csv(input_csv)
    df["clean_comment"] = df["open_comment"].apply(clean_text)

    print("🔍 Generating embeddings...")
    embeddings = get_embeddings(df["clean_comment"].tolist())
    X = np.array(embeddings)

    print(f"📊 Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)

    closest_indices, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, X
    )

    cluster_results = []

    print("\n📝 Labeling and summarizing clusters with LLM…")
    for cluster_id in range(n_clusters):
        cluster_df = df[df["cluster"] == cluster_id]
        comments = cluster_df["clean_comment"].tolist()

        comment_block = "\n".join(comments[:30])

        theme = llm_label_cluster(comment_block)
        summary_obj = llm_summarize_cluster(comment_block)
        representative = df.loc[closest_indices[cluster_id], "open_comment"]

        cluster_results.append({
            "cluster": cluster_id,
            "theme": theme,
            "overall_sentiment": summary_obj["overall_sentiment"],
            "themes": ", ".join(summary_obj["themes"]),
            "summary": summary_obj["summary"],
            "recommended_action": summary_obj["recommended_action"],
            "representative_comment": representative,
            "n_comments": len(comments)
        })

        print(f"\nCluster {cluster_id}: {theme}")
        print(summary_obj["summary"])

    results_df = pd.DataFrame(cluster_results)
    results_df.to_csv("cluster_summaries.csv", index=False)

    print("\n📘 Generating global executive summary…")
    global_summary = llm_global_summary(
        "\n\n".join(results_df["summary"])
    )

    with open("executive_summary.json", "w") as f:
        json.dump(global_summary, f, indent=2)

    print("\n✨ Done!")
    print("\nExecutive Summary:\n")
    print(global_summary["narrative_summary"])


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Engagement Survey Summarization Pipeline (Structured LLM Output)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/content/llm-engagement-summarizer/synthetic_engagement_survey.csv",
        help="Path to survey CSV file."
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=6,
        help="Number of clusters to produce."
    )

    args = parser.parse_args()
    run_pipeline(args.input, n_clusters=args.clusters)

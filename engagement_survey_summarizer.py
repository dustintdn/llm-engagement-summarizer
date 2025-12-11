import pandas as pd
import numpy as np
import re
import argparse

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# ---- Replace with your provider (OpenAI, Anthropic, etc.) ----
from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")


# ============================================================
# Helper Functions
# ============================================================

def clean_text(text):
    """Very light cleaning for short survey comments."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    return text


def get_embeddings(texts, model="text-embedding-3-large"):
    """Return embeddings for a list of strings."""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]


def llm_label_cluster(comments):
    """Generate a theme label for a cluster using the LLM."""
    prompt = f"""
    You are a People Analytics expert. 
    Below are employee survey comments that belong to one cluster.
    Provide a short theme label (2-4 words) that best represents them.

    Comments:
    {comments}

    Theme label:
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20
    )
    return response.choices[0].message["content"].strip()


def llm_summarize_cluster(comments):
    """Summarize themes and sentiment for a cluster."""
    prompt = f"""
    Summarize the following employee survey comments into:
    - 2-3 key themes
    - Overall sentiment (positive / neutral / negative)
    - A short recommended action for leadership

    Comments:
    {comments}

    Summary:
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message["content"].strip()


def llm_global_summary(cluster_summaries):
    """Create a final executive-level summary."""
    prompt = f"""
    You are a People Analytics manager.
    Below are summaries from multiple clusters of employee feedback.

    Create a 5-7 sentence executive summary of overall employee sentiment.
    Focus on:
    - Strengths
    - Areas of concern
    - Notable themes
    - Recommended actions

    Cluster summaries:
    {cluster_summaries}

    Executive Summary:
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250
    )
    return response.choices[0].message["content"].strip()


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

    # Find representative comments per cluster
    closest_indices, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, X
    )

    # Collect summaries
    cluster_results = []

    print("\n📝 Labeling and summarizing clusters with LLM…")
    for cluster_id in range(n_clusters):
        cluster_df = df[df["cluster"] == cluster_id]
        comments = cluster_df["clean_comment"].tolist()

        comment_block = "\n".join(comments[:30])  # limit size for cost/speed

        theme = llm_label_cluster(comment_block)
        summary = llm_summarize_cluster(comment_block)
        representative = df.loc[closest_indices[cluster_id], "open_comment"]

        cluster_results.append({
            "cluster": cluster_id,
            "theme": theme,
            "summary": summary,
            "representative_comment": representative,
            "n_comments": len(comments)
        })

        print(f"\nCluster {cluster_id}: {theme}")
        print(summary)

    results_df = pd.DataFrame(cluster_results)
    results_df.to_csv("cluster_summaries.csv", index=False)

    print("\n📘 Generating global executive summary…")
    global_summary = llm_global_summary("\n\n".join(results_df["summary"]))

    with open("executive_summary.txt", "w") as f:
        f.write(global_summary)

    print("\n✨ Done!")
    print("\nExecutive Summary:\n")
    print(global_summary)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Engagement Survey Summarization Pipeline")
    parser.add_argument("--input", type=str, default="synthetic_engagement_survey.csv",
                        help="Path to survey CSV file.")
    parser.add_argument("--clusters", type=int, default=6,
                        help="Number of clusters to produce.")
    args = parser.parse_args()

    run_pipeline(args.input, n_clusters=args.clusters)

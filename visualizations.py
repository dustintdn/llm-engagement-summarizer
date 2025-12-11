import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Helper Function: extract sentiment from cluster summary text
# ------------------------------------------------------------
def extract_sentiment(text):
    """
    Very simple sentiment extractor based on LLM output structure.
    Looks for 'positive', 'negative', or 'neutral' in summary text.
    """
    text_lower = text.lower()

    if "negative" in text_lower:
        return "Negative"
    if "positive" in text_lower:
        return "Positive"
    if "neutral" in text_lower:
        return "Neutral"

    # fallback if LLM phrasing differs
    if "concern" in text_lower:
        return "Negative"
    if "strength" in text_lower:
        return "Positive"

    return "Other"


# ------------------------------------------------------------
# Main Visualization Pipeline
# ------------------------------------------------------------
def run_visualizations(cluster_csv="cluster_summaries.csv",
                       survey_csv="synthetic_engagement_survey.csv"):

    print("📥 Loading data...")
    clusters = pd.read_csv(cluster_csv)
    survey = pd.read_csv(survey_csv)

    # Clean data
    clusters["sentiment"] = clusters["summary"].apply(extract_sentiment)
    clusters["theme"] = clusters["theme"].astype(str)

    # ============================================================
    # 1. Sentiment Distribution (by number of comments)
    # ============================================================

    print("📊 Creating sentiment distribution chart...")

    # Weight sentiment by number of comments per cluster
    sentiment_counts = clusters.groupby("sentiment")["n_comments"].sum().reset_index()

    plt.figure(figsize=(7, 5))
    sns.barplot(data=sentiment_counts, x="sentiment", y="n_comments")
    plt.title("Sentiment Distribution Across Survey Responses")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Comments")
    plt.tight_layout()

    plt.savefig("sentiment_distribution.png", dpi=150)
    plt.show()

    # ============================================================
    # 2. Theme Frequency Visualization
    # ============================================================

    print("🎨 Creating theme frequency chart...")

    theme_counts = clusters.groupby("theme")["n_comments"].sum().reset_index()
    theme_counts = theme_counts.sort_values("n_comments", ascending=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=theme_counts, x="n_comments", y="theme")
    plt.title("Theme Frequency (based on cluster sizes)")
    plt.xlabel("Number of Comments")
    plt.ylabel("Theme")
    plt.tight_layout()

    plt.savefig("theme_frequency.png", dpi=150)
    plt.show()

    # ============================================================
    # 3. Save Combined Visualization Data (Optional)
    # ============================================================

    output_data = clusters[["cluster", "theme", "sentiment", "n_comments"]]
    output_data.to_csv("visualization_data.csv", index=False)

    print("\n✨ Visualizations complete!")
    print("Saved files:")
    print(" - sentiment_distribution.png")
    print(" - theme_frequency.png")
    print(" - visualization_data.csv")


# ------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------

if __name__ == "__main__":
    run_visualizations()

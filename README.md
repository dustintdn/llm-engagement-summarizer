# 📊 Engagement Survey Summarizer

**Using LLMs to analyze qualitative employee feedback**

This project demonstrates an end-to-end pipeline for analyzing synthetic employee engagement survey comments using classical NLP, sentiment analysis, and LLM-powered theme summarization.

It is designed specifically as a **lightweight project** to explore:

* Utilzing NLP methods for employee experience data
* Applying **LLMs** to extract insights from qualitative text*
* Producing **summaries + visualizations** suitable for HR stakeholders

> ⚠️ **Note:** All data in this project is fully synthetic and generated in-notebook.
> No real employee data is used.

---

## 🚀 Project Overview

This project simulates an HR engagement survey consisting of open-ended employee comments across several topics (Leadership, Compensation, Culture, etc.).

The workflow:

1. **Generate synthetic survey responses**
2. **Perform sentiment analysis** (using TextBlob)
3. **Group comments by topic** by embedding survey comments and applying clustering
4. **Use an LLM to summarize themes**
5. **Visualize sentiment and theme distribution**

---

## 📁 Repository Structure

```
engagement-survey-summarizer/
│
├── engagement_survey_summarizer.py        # Pipeline script
├── visualizations.py                      # Visualization code (sentiment + themes)
│
├── sample_outputs/
│   ├── theme_summaries.csv
│   └── theme_summaries.json
│
└── README.md
```

---

## 📘 How It Works

### 1. **Synthetic Data Generation**

Creates 300 employee comments across 5 HR themes:

* Leadership
* Culture
* Career Growth
* Compensation
* Work-Life Balance

### 2. **Sentiment Analysis**

A polarity score is assigned to each comment
(`-1 = negative`, `+1 = positive`).

### 3. **LLM Theme Summaries**

For each theme, an LLM generates:

* Key themes
* Common concerns
* Positive highlights

### 4. **Visualizations**

The repo includes quick HR-friendly charts:

* Sentiment distribution
* Comment count by theme
* Average sentiment by theme

---

## 🧪 Example Output Snippet

```
Theme: Career Growth
--------------------
• Employees feel unclear about upward mobility
• Requests for more mentorship and skill development
• Positive sentiment toward manager support but desire for structure
• HR Action: define clear promotion paths, launch internal mobility programs
```

---
## 🔑 Environment Variables

Set your OpenAI key for the LLM summarization step:

```
export OPENAI_API_KEY=your_key_here
```

Or place it in a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

---

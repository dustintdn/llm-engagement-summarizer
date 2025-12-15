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
4. **Use an LLM to summarize themes and structured outputs to enforce sentiment labels and themes (JSON format)**
5. **Visualize sentiment and theme distribution**

---

## 📁 Repository Structure

```
engagement-survey-summarizer/
│
├── synthetic_engagement_survey.csv        # Employee-experience survey data
├── engagement_survey_summarizer.py        # Pipeline script
├── visualizations.py                      # Visualization code (sentiment + themes)
│
├── sample_outputs/
│   ├── cluster_summaries.csv
│   └── executive_summary.json
│
└── README.md
```

---

## 📘 How It Works

### 1. **Synthetic Data Generation**

Creates 100 employee comments across randomized HR themes.

### 2. **LLM Theme Summaries**

For each theme, an LLM generates:

* Key themes
* Summary description
* Recommended action for people decision-makers

A narrative summary that synthesizes all key themes into an overall report with recommendations and highlights.

### 3. **Visualizations**

The repo includes quick HR-friendly charts:

* Sentiment distribution
* Comment count by theme

---
## 🧪 Example Inputs/Outputs

### Input Data: unstructured employee comments
```
Columns:
employee_id,department,tenure_years,engagement_score,open_comment
--------------------
E001,Engineering,2.3,4,"I appreciate the flexibility and my team, but sometimes communication from leadership feels unclear."
E002,Marketing,1.1,3,"Workload has increased a lot recently and it's becoming hard to maintain balance."
E003,Sales,4.8,5,"My manager is fantastic and gives helpful feedback that supports my growth."
```
### Output Data: labeled theme and summary highlights
```
Columns:
cluster,theme,overall_sentiment,themes,summary,recommended_action,representative_comment,n_comments
--------------------
0,Empowerment and Collaboration,Positive,"Collaboration and Teamwork, Empowerment and Innovation, Alignment with Company Mission","Employees feel that the team collaborates well, leadership empowers them, and they are aligned with the company's mission. There is a strong sense of purpose and encouragement for innovation within the workplace.",Continue to promote and celebrate team collaboration and innovative efforts among employees.,Our team collaborates extremely well and leadership is empowering.,8
1,Supportive Team Culture,Positive,"Team Collaboration, Supportive Work Environment, Workload Management","Overall, employees express a strong appreciation for the supportive and collaborative culture within the engineering team, alongside a recognition of some challenges related to cross-team collaboration and workload pressures due to unrealistic deadlines.",Enhance cross-team collaboration initiatives and assess workload expectations to ensure they are realistic and manageable.,Good work-life balance and supportive teammates.,25
```
### Output Data: narrative/global summary of all themes
```
{
  "overall_sentiment": "Neutral",
  "strengths": [
    "Strong sense of purpose and alignment with the company's mission",
    "High levels of collaboration within the engineering team",
    "Empowering leadership that supports innovation",
    "Flexibility with work-from-home options",
    "Improvements in team morale"
  ],
  "areas_of_concern": [
    "Heightened stress and burnout due to increased workloads",
    "Understaffing and unrealistic deadlines",
    "Outdated tools and processes",
    "Unclear communication from leadership",
    "Limited career advancement opportunities",
    "Need for clearer expectations and priorities",
    "Insufficient documentation, onboarding processes, and training"
  ],
  "recommended_actions": [
    "Evaluate workload distributions and staffing levels to mitigate burnout",
    "Review and update tools and processes to enhance efficiency",
    "Implement regular check-ins to improve communication from leadership",
    "Clarify career advancement paths and provide constructive feedback mechanisms",
    "Enhance documentation, onboarding, and training programs for better clarity"
  ],
  "narrative_summary": "Employees generally appreciate the supportive and collaborative culture within the engineering team, highlighting effective teamwork and leadership empowerment. However, they face challenges such as heightened stress from increased workloads and unrealistic deadlines, particularly due to understaffing and outdated tools. Despite positive aspects such as work-from-home flexibility and improved morale, there are significant concerns regarding communication clarity from leadership and opportunities for professional growth. Addressing these issues through targeted actions can further strengthen the positive culture while alleviating current pressures."
}
```

---
## 🔑 Sample Code

Set your OpenAI key for the LLM summarization step:
```
import os
os.environ["OPENAI_API_KEY"] = your_key_here
```

Clone repo and execute .py scripts:
```
!git clone https://github.com/dustintdn/llm-engagement-summarizer.git
!python /llm-engagement-summarizer/engagement_survey_summarizer.py
!python /llm-engagement-summarizer/visualizations.py
```
---

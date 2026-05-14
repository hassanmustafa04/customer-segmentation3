# customer-segmentation3
#  Customer Segmentation with Machine Learning

An end-to-end customer segmentation project using unsupervised machine learning — with an interactive Streamlit dashboard to explore results visually.

---

## Overview

This project analyzes customer data and groups them into meaningful segments using clustering algorithms. It helps businesses understand their customer base and tailor strategies per segment.

**Two clusters identified:**

| Cluster | Label | Size |
|---|---|---|
| Cluster 0 | Balanced / Mixed-Value Segment | ~77.1% of customers |
| Cluster 1 | High-Value Loyalists + Engaged Subscribers | ~22.9% of customers |

---

##  Features

- **3 clustering algorithms** — K-Means, DBSCAN, and Agglomerative Hierarchical Clustering
- **Automatic model evaluation** — Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
- **Interactive Streamlit dashboard** with 4 pages:
  - 📊 Overview — KPIs, PCA scatter plot, cluster size donut chart
  - 👤 Cluster Profiles — radar charts, feature breakdowns, segment descriptions
  - 🔲 Compare Clusters — heatmap of feature averages across clusters
  - 🔍 Individual Explorer — filter, search, and download customer data by segment

---

##  Project Structure

```
customer_segmentation/
├── app.py                  # Streamlit dashboard
├── main.py                 # Core clustering pipeline
├── requirements.txt        # Dependencies
├── data/                   # Customer dataset
├── notebooks/              # Jupyter exploration notebook
├── outputs/                # Generated charts and plots
├── scripts/                # Helper scripts
└── src/
    ├── data_loader.py
    ├── preprocessor.py
    ├── clustering.py
    ├── evaluator.py
    ├── visualizer.py
    └── segment_report.py
```

---

##  Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/customer-segmentation.git
cd customer-segmentation
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## ▶️How to Run

**Run the clustering pipeline (terminal output):**
```bash
python main.py
```

**Launch the interactive dashboard:**
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## Dataset Features

| Category | Features |
|---|---|
| Demographics | Age, Gender, Income, Education Level |
| Behavioral | Purchase Frequency, Avg Order Value, Days Since Last Purchase |
| Engagement | Email Open Rate, Loyalty Points, Support Tickets |

---

##  Key Findings

**Cluster 0 — Balanced / Mixed-Value (~77.1%)**
- Close to population averages across all signals
- Average income: $51,873 | Purchase frequency: 2.9x | Loyalty points: 2,697

**Cluster 1 — High-Value Loyalists (~22.9%)**
- Frequent buyers with strong order values and very recent activity
- Average income: $73,681 | Purchase frequency: 8.6x | Loyalty points: 8,405
- High email engagement (79% open rate) and low support tickets

---

## Tech Stack

- **Python** — pandas, numpy, scikit-learn, scipy
- **Visualization** — matplotlib, seaborn, plotly
- **Dashboard** — Streamlit, streamlit-option-menu
- **Notebook** — Jupyter

---

##  Author

Hassan Haj Hasan
Built with Python & Cursor AI

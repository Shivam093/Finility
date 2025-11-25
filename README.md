# Finility: News Sentiment, VIX & S&P 500 Returns (2017–2020)
This project analyzes whether daily financial news sentiment can help explain or predict movements in the S&P 500 index.  
The study combines ~50,000 headlines from major news outlets with market volatility (VIX) and stock returns.

---

# Data sources
Below is a summary of all datasets used in the project.

| # | Name / Description | Source URL | Type | Fields | Format | Python Access | Estimated Size |
|---|--------------------|------------|------|--------|--------|---------------|----------------|
| 1 | Financial News Headlines (CNBC, Guardian, Reuters) | https://www.kaggle.com/datasets/notlucasp/financial-news-headlines | File (manual download) | date, headline, description | CSV | yes | ~50,000 rows |
| 2 | S&P 500 Index (^GSPC) | https://finance.yahoo.com | API via `yfinance` | Date, Open, High, Low, Close, Volume | CSV | yes | ~1,000 rows |
| 3 | VIX Volatility Index (^VIX) | https://finance.yahoo.com | API via `yfinance` | Date, Close | CSV | yes | ~1,000 rows |

All datasets are aligned for the period **2017–2020**.

---

# Results
**Summary of Findings**
- Daily news sentiment (same-day and lagged) does **not** significantly predict daily S&P 500 returns.  
- VIX (market volatility) **does** significantly explain returns. Higher volatility corresponds to more negative expected returns.  
- Correlation analysis shows:  
  - **Sentiment ↔ Return:** very weak positive correlation  
  - **VIX ↔ Return:** weak negative correlation  
  - **Sentiment ↔ VIX:** moderate negative correlation (fearful markets → negative headlines)  
- Regression **R² ≈ 0.027**, meaning daily stock movements are mostly noise, which is typical for financial time series.

All plots, tables, and regression outputs are in:
`results.ipynb`

This notebook generates:
- Summary statistics  
- Correlation matrix & heatmap  
- Sentiment vs return scatter  
- VIX vs return scatter  
- 30-day rolling sentiment trend  
- Regression results  

---

# Installation
1. **Set up environment variables**  
   Create a `.env` file in `src/` based on `.env.example`.  
   No API keys are required unless you want to re-download Kaggle datasets.  
   **Yahoo Finance data requires no API key.**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

# Running analysis
To reproduce the full pipeline, run the following scripts (from project root):

### Step 1 — Load & combine news
```bash
python src/news_data.py
```

### Step 2 — Train sentiment classifier
```bash
python src/sentiment_model.py
```

### Step 3 — Apply sentiment model & generate daily sentiment
```bash
python src/apply_sentiment.py
```

### Step 4 — Merge sentiment with market data
```bash
python src/merge_series.py
```

This produces the final merged dataset:
`data/merged_sentiment_market.csv`

### Final analysis
All final results, plots, and regression models are produced in:
`results.ipynb`

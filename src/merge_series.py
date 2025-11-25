import pandas as pd

# 1) Daily sentiment
daily_sent = pd.read_csv("data/daily_sentiment_2017_2020.csv")
daily_sent["date"] = pd.to_datetime(daily_sent["date"])
daily_sent = daily_sent.set_index("date")

# 2) S&P 500
sp500 = pd.read_csv("data/sp500_2017_2020.csv")
sp500 = sp500.rename(columns={"Price": "Date"})
sp500 = sp500.iloc[2:].copy()        # remove "Ticker" and "Date,,,,"
sp500["Date"] = pd.to_datetime(sp500["Date"])
sp500 = sp500.set_index("Date")
sp500["return"] = sp500["Close"].astype(float).pct_change()

# 3) VIX
vix = pd.read_csv("data/vix_2017_2020.csv")
vix = vix.rename(columns={"Price": "Date"})
vix = vix.iloc[2:].copy()
vix["Date"] = pd.to_datetime(vix["Date"])
vix = vix.set_index("Date")
vix = vix.rename(columns={"Close": "vix"})
vix["vix"] = vix["vix"].astype(float)

# 4) Merge sentiment + returns + VIX
merged = sp500[["return"]].merge(
    daily_sent, left_index=True, right_index=True, how="left"
)

merged = merged.merge(
    vix[["vix"]], left_index=True, right_index=True
)

# Keep only rows where daily_sentiment is available
filtered = merged.loc["2017-12-18":"2020-07-17"].dropna(subset=["daily_sentiment"])

print(filtered.head())
print(filtered.tail())
print(filtered.isna().sum())

# Save final dataset
filtered.to_csv("data/merged_sentiment_market.csv")
print("Saved filtered dataset → data/merged_sentiment_market.csv")

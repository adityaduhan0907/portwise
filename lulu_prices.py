import yfinance as yf
from datetime import datetime, timedelta

ticker = "LULU"
end_date = datetime.today()
start_date = end_date - timedelta(days=30)

stock = yf.Ticker(ticker)
df = stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

if df.empty:
    print("No data returned. Check your internet connection or ticker symbol.")
else:
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = df.index.strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  Lululemon (LULU) — Past 30 Days Price Data")
    print(f"  {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"{'='*60}\n")

    print(f"{'Date':<14} {'Open':>8} {'High':>8} {'Low':>8} {'Close':>8} {'Volume':>12}")
    print("-" * 60)
    for date, row in df.iterrows():
        print(f"{date:<14} {row['Open']:>8.2f} {row['High']:>8.2f} {row['Low']:>8.2f} {row['Close']:>8.2f} {int(row['Volume']):>12,}")

    overall_high = df["High"].max()
    overall_low = df["Low"].min()
    high_date = df["High"].idxmax()
    low_date = df["Low"].idxmin()
    first_close = df["Close"].iloc[0]
    last_close = df["Close"].iloc[-1]
    pct_change = ((last_close - first_close) / first_close) * 100

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Period High : ${overall_high:.2f}  (on {high_date})")
    print(f"  Period Low  : ${overall_low:.2f}  (on {low_date})")
    print(f"  Latest Close: ${last_close:.2f}")
    print(f"  30-Day Move : {pct_change:+.2f}%")
    print(f"{'='*60}\n")

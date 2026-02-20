"""Generate sample stock data for testing."""

import pandas as pd
import numpy as np
import os

print("Generating sample data...")

# Create directories
os.makedirs('data/raw', exist_ok=True)

# Generate 50 stocks
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 
           'V', 'WMT', 'JNJ', 'PG', 'MA', 'HD', 'DIS', 'BAC', 'ADBE', 'NFLX',
           'CRM', 'CSCO', 'PFE', 'KO', 'PEP', 'INTC', 'CMCSA', 'VZ', 'T',
           'MRK', 'ABT', 'CVX', 'TMO', 'ABBV', 'ACN', 'NKE', 'COST', 'UNP',
           'DHR', 'LIN', 'PM', 'NEE', 'UPS', 'HON', 'QCOM', 'IBM', 'TXN',
           'LOW', 'MDT', 'RTX', 'BMY', 'SBUX']

# Generate prices
data = {}
for i, ticker in enumerate(tickers):
    np.random.seed(42 + i)
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    data[ticker] = prices

df = pd.DataFrame(data, index=dates)

# Generate cointegrated pairs
pairs = {}
for i in range(5):
    np.random.seed(100 + i)
    returns = np.random.multivariate_normal(
        [0.0005, 0.0005],
        [[0.0004, 0.85*0.0004], [0.85*0.0004, 0.0004]],
        len(dates)
    )
    pairs[f'PAIR{i+1}_A'] = 100 * np.exp(np.cumsum(returns[:, 0]))
    pairs[f'PAIR{i+1}_B'] = 100 * np.exp(np.cumsum(returns[:, 1]))

pairs_df = pd.DataFrame(pairs, index=dates)

# Save
df.to_parquet('data/raw/sp500_sample.parquet')
pairs_df.to_parquet('data/raw/cointegrated_pairs.parquet')

print(f"✅ Created sp500_sample.parquet: {df.shape}")
print(f"✅ Created cointegrated_pairs.parquet: {pairs_df.shape}")
print("✅ Sample data ready!")

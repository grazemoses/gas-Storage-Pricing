# STEP 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA
from typing import Union

# STEP 2: Load the CSV data
df = pd.read_csv(r"C:\Users\96657\Downloads\Nat_Gas.csv")
df['Dates'] = pd.to_datetime(df['Dates'])
df = df.sort_values("Dates").set_index("Dates")

# STEP 3: Visualize raw time series
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y='Prices', marker='o')
plt.title("Natural Gas Prices (Oct 2020 – Sep 2024)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 4: Visualize seasonality by month
df['Month'] = df.index.month_name()
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Month', y='Prices', order=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'])
plt.title("Seasonal Pattern of Natural Gas Prices by Month")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 5: Forecast 12 months using STL + ARIMA
df.drop(columns='Month', inplace=True)
stlf = STLForecast(df['Prices'], ARIMA, model_kwargs={"order": (1, 1, 1)}, period=12)
model = stlf.fit()
forecast = model.forecast(steps=12)

# Combine with original series
full_series = pd.concat([df['Prices'], forecast])

# Plot forecast
plt.figure(figsize=(12, 6))
sns.lineplot(data=full_series, marker='o')
plt.axvline(x=df.index[-1], color='red', linestyle='--', label='Forecast Start')
plt.title("Natural Gas Prices with 12-Month Forecast")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 6: Price estimation function
def estimate_gas_price(query_date: Union[str, pd.Timestamp]) -> float:
    if isinstance(query_date, str):
        query_date = pd.to_datetime(query_date)

    series = full_series.copy().sort_index()
    series = series.reindex(series.index.union([query_date]))
    series = series.interpolate(method='time')
    
    if query_date < series.index.min() or query_date > series.index.max():
        raise ValueError(f"Date out of range: {query_date.date()}")

    return round(series.loc[query_date], 2)

# STEP 7: Example usage
test_dates = ["2023-07-15", "2024-12-01", "2025-06-20"]
for date in test_dates:
    print(f"Estimated price on {date}: ${estimate_gas_price(date)}")

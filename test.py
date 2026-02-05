import torch
import yfinance as yf   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

toyota = yf.Ticker("7203.T")
df_1h = toyota.history(period="730d", interval="1h")
df_5m = toyota.history(period="60d", interval="5m")
print(df_1h.head())
print(df_1h.tail())
print(df_5m.head())
print(df_5m.tail())


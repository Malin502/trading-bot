import torch
import yfinance as yf   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

topix_100 = yf.Ticker("1306.T")
df_1h = topix_100.history(period="730d", interval="1h")
df_5m = topix_100.history(period="60d", interval="5m")
print(df_1h.tail(20)) #20個前まで
print(df_5m.tail())


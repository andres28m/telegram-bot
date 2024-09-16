# fetch_data.py
import ccxt
import pandas as pd

def fetch_data(symbol='BTC/USDT', timeframe='1d', limit=1000):
    exchange = ccxt.binance()
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Obtener datos de BTC/USDT y guardar en CSV
df = fetch_data()
df.to_csv('btc_usdt_data.csv')
print("Datos guardados en btc_usdt_data.csv")

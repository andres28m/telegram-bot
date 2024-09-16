# telegram_bot.py
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext
import ccxt
import pandas as pd
import pandas_ta as ta
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Token de tu bot de Telegram
TELEGRAM_BOT_TOKEN = '7504751119:AAGps4QYaCqZoLXIR2KhShYanKNcpv9q1NI'

# Configuración de parámetros para el análisis
LEVERAGE = 10  # Apalancamiento
TIMEFRAME = '1d'  # Tiempo en la operación

# Cargar el modelo y el escalador
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Función para manejar el comando /start
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text('¡Hola! Usa /help para obtener ayuda.')

# Función para manejar el comando /help
async def help_command(update: Update, context: CallbackContext):
    await update.message.reply_text('Aquí están los comandos disponibles:\n/start - Inicia el bot\n/help - Muestra esta ayuda\n/analyze <símbolo> - Analiza la criptomoneda')

# Función para analizar criptomonedas
async def analyze_crypto(update: Update, context: CallbackContext):
    if len(context.args) == 0:
        await update.message.reply_text('Por favor, proporciona el símbolo de la criptomoneda.')
        return

    symbol = context.args[0].upper()
    try:
        # Configurar el intercambio y obtener datos
        exchange = ccxt.binance()
        data = exchange.fetch_ohlcv(symbol + '/USDT', timeframe=TIMEFRAME, limit=1000)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Análisis técnico
        df.set_index('timestamp', inplace=True)
        df['SMA'] = ta.sma(df['close'], length=14)
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['EMA'] = ta.ema(df['close'], length=14)
        df['MACD'] = ta.macd(df['close'])['MACD']
        df['Bollinger_High'] = ta.bbands(df['close'])['BBL_14']
        df['Bollinger_Low'] = ta.bbands(df['close'])['BBU_14']
        df.dropna(inplace=True)

        # Preparar datos para el modelo de ML
        latest = df.iloc[-1]
        features = np.array([latest[['SMA', 'RSI', 'EMA', 'MACD', 'Bollinger_High', 'Bollinger_Low']]])
        features = scaler.transform(features)
        prediction = model.predict(features)[0]

        # Determinar estado del mercado
        if latest['RSI'] > 70:
            market_state = "Sobrecomprado (bajista)"
        elif latest['RSI'] < 30:
            market_state = "Sobrevendido (alcista)"
        else:
            market_state = "Neutral"

        # Cálculo de precios de entrada, TP y SL
        entry_price = latest['close']
        tp_price = entry_price * (1 + 0.01 * LEVERAGE)  # TP al 1% de la entrada ajustado por apalancamiento
        sl_price = entry_price * (1 - 0.01 * LEVERAGE)  # SL al 1% de la entrada ajustado por apalancamiento

        # Tiempo en la operación
        time_in_operation = TIMEFRAME

        # Crear el mensaje
        message = (
            f"Símbolo: {symbol}\n"
            f"Último Precio: {latest['close']}\n"
            f"SMA (14): {latest['SMA']}\n"
            f"RSI (14): {latest['RSI']}\n"
            f"EMA (14): {latest['EMA']}\n"
            f"MACD: {latest['MACD']}\n"
            f"Bollinger High: {latest['Bollinger_High']}\n"
            f"Bollinger Low: {latest['Bollinger_Low']}\n"
            f"Estado del Mercado: {market_state}\n"
            f"Predicción de Machine Learning: {'Sube' if prediction == 1 else 'Baja'}\n"
            f"Precio de Entrada: {entry_price}\n"
            f"Take Profit (TP): {tp_price}\n"
            f"Stop Loss (SL): {sl_price}\n"
            f"Apalancamiento: {LEVERAGE}x\n"
            f"Tiempo en la Operación: {time_in_operation}"
        )

        await update.message.reply_text(message)
    except Exception as e:
        await update.message.reply_text(f'Error: {str(e)}')

# Función principal para configurar el bot
def main():
    # Crear la aplicación
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Añadir los manejadores de comandos
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("analyze", analyze_crypto))
    
    # Iniciar la aplicación
    application.run_polling()

if __name__ == '__main__':
    main()



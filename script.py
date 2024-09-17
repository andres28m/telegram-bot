import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Configuración del registro/logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Función que responderá al comando /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('¡Hola! Soy tu bot de Telegram.')

# Función principal que inicializa el bot
async def main():
    # Reemplaza 'TOKEN' con el token de tu bot de Telegram
    application = ApplicationBuilder().token('TOKEN').build()

    # Añade un manejador para el comando /start
    application.add_handler(CommandHandler("start", start))

    # Ejecuta el bot
    await application.start()
    await application.idle()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())


import os
import asyncio
import logging
import uvicorn
import fcntl
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables at startup
load_dotenv()

from src.server.telegram_bot import TelegramBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SingleInstanceException(Exception):
    pass

def obtain_lock():
    """Ensure only one instance runs at a time"""
    lock_file = open("/tmp/telegram_bot.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_file
    except (IOError, BlockingIOError):
        raise SingleInstanceException("Another instance is already running")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup/shutdown with FastAPI"""
    try:
        # Ensure single instance
        lock_file = obtain_lock()
        logger.info("Successfully obtained lock file")

        # Initialize and start Telegram bot
        bot = TelegramBot()
        await bot.start()
        logger.info("Telegram bot started and ready for interactions")
        
        yield
        
        # Clean shutdown
        logger.info("Shutting down Telegram bot...")
        await bot.stop()
        
    except SingleInstanceException as e:
        logger.error(f"Failed to start: {str(e)}")
        raise
    finally:
        try:
            if 'lock_file' in locals():
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()
                os.unlink("/tmp/telegram_bot.lock")
                logger.info("Lock file cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up lock file: {str(e)}")

# Create FastAPI app for health checks
app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        loop="asyncio"
    )
    
    server = uvicorn.Server(config)
    asyncio.run(server.serve())

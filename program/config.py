from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
folder_path = "data"
threadCount = os.environ["NUMEXPR_MAX_THREADS"] = "16"

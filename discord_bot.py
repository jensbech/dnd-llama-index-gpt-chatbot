from discord.ext import commands
import discord
from bot_functions import *
from bot_events import on_message
from init_chatbot import initialize_chatbot
from setup_query_engines import setup_query_engines
from config import DISCORD_TOKEN

message_queues = {}
context_memory = {}
max_pairs = 0

llm_predictor_chatgpt, service_context, vector_indices, index_summaries = initialize_chatbot()
router_query_engine = setup_query_engines(llm_predictor_chatgpt, service_context, vector_indices, index_summaries)

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True
intents.typing = True
intents.presences = False
bot = commands.Bot(command_prefix=lambda _, __: [], intents=intents)
bot.remove_command("help")

@bot.event
async def on_message(message):
    await on_message(bot, message, router_query_engine, context_memory, max_pairs)

bot.run(DISCORD_TOKEN)

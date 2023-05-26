import json
from llama_index import (
    GPTTreeIndex,
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    StorageContext,
)
from langchain.chat_models import ChatOpenAI
from llama_index.indices.composability import ComposableGraph
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from discord.ext import commands
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.llm_selectors import LLMSingleSelector
from config import OPENAI_API_KEY, DISCORD_TOKEN, folder_path, threadCount
from console_logging import enable_logging
import discord
from pathlib import Path
from llama_index import download_loader

context_memory = {}
max_pairs = 2

MarkdownReader = download_loader("MarkdownReader")
loader = MarkdownReader()

enable_logging()

with open("file_index.json") as file:
    file_index = json.load(file)

llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-3.5-turbo",
)
llm_predictor_chatgpt = LLMPredictor(llm)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor_chatgpt, chunk_size_limit=1024
)

vector_indices = {}
index_summaries = {}

data_dir = Path(folder_path)
for md_file in data_dir.glob("**/*.md"):
    print(md_file)

documents = {}
for md_file in Path(folder_path).glob("**/*.md"):
    file_name = md_file.stem
    metadata = file_index.get(file_name)
    if metadata is not None:
        documents[file_name] = loader.load_data(file=md_file)

        storage_context = StorageContext.from_defaults()

        vector_indices[file_name] = GPTVectorStoreIndex.from_documents(
            documents[file_name],
            service_context=service_context,
            storage_context=storage_context,
        )
        vector_indices[file_name].set_index_id(file_name)

        storage_context.persist(persist_dir=f"./storage/{file_name}")

        index_summaries[file_name] = (
            "This index contains information about " + metadata["description"]
        )
        print(index_summaries[file_name])


graph = ComposableGraph.from_indices(
    GPTTreeIndex,
    [index for _, index in vector_indices.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50,
)

decompose_transform = DecomposeQueryTransform(llm_predictor_chatgpt, verbose=True)

custom_query_engines = {}
for index in vector_indices.values():
    query_engine = index.as_query_engine(service_context=service_context)
    query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_extra_info={"index_summary": index.index_struct.summary},
    )
    custom_query_engines[index.index_id] = query_engine

custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
    response_mode="tree_summarize",
    service_context=service_context,
    verbose=True,
)

graph_query_engine = graph.as_query_engine(custom_query_engines=custom_query_engines)

query_engine_tools = []

for index_summary in index_summaries:
    index = vector_indices[index_summary]
    summary = index_summaries[index_summary]

    query_engine = index.as_query_engine(service_context=service_context)
    vector_tool = QueryEngineTool.from_defaults(query_engine, description=summary)
    query_engine_tools.append(vector_tool)

graph_description = "This tool contains information about a fictional Dungeons and Dragons 5E universe called Kazar, including characters, locations, events, and lore."
graph_tool = QueryEngineTool.from_defaults(
    graph_query_engine, description=graph_description
)
query_engine_tools.append(graph_tool)

router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(
        service_context=service_context,
    ),
    query_engine_tools=query_engine_tools,
)

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True
intents.typing = True
intents.presences = False
bot = commands.Bot(command_prefix=lambda _, __: [], intents=intents)

bot.remove_command("help")


async def ask(message, question: str):
    context = context_memory.get(message.author.id)
    if context:
        history = "\n".join(
            [f"M: {pair['user_message']}\nA: {pair['bot_reply']}" for pair in context]
        )
        question = f"CONVERSATION MEMORY :\n{history}\nYOU MAY CONSULT THE MEMORY WHEN ANSWERING. NEVER ANSWER IN THE SAME WAY YOU'VE DONE BEFORE. HERE IS THE NEXT QUESTION FOR YOU TO ANSWER:\nM: {question}"

    question = f"You are an eccentric and audacious theorist, coming up with wild alternative theories to the source material you answer questions from. You answer every question with a wild alternative theory than the original source material. This is your question: {question}"
    try:
        response = router_query_engine.query(question)
        responseString = response.response

        if message.author.id not in context_memory:
            context_memory[message.author.id] = []
        context_memory[message.author.id].append(
            {"user_message": message.content, "bot_reply": responseString}
        )

        print("CURRENT CONTEXT MEMORY: ", context_memory[message.author.id])
        if len(context_memory[message.author.id]) > max_pairs:
            context_memory[message.author.id] = context_memory[message.author.id][
                -max_pairs:
            ]

        responseString = (
            responseString[3:] if responseString.startswith("A: ") else responseString
        )

        await message.reply(responseString)
    except ValueError as e:
        print(f"Caught an error: {e}")
        default_response = (
            "I'm sorry, there is no answer to that question in my knowledge base..."
        )
        print("Responding with: " + default_response)
        await message.reply(default_response)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} - {bot.user.id}")


@bot.event
async def on_message(message):
    if message.author.bot:
        return
    if bot.user in message.mentions:
        question = message.content.replace(f"<@!{bot.user.id}>", "").strip()
        print("Answering question: ", question)
        async with message.channel.typing():
            await ask(message, question)
    await bot.process_commands(message)


bot.run(DISCORD_TOKEN)

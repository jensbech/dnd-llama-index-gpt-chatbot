import json
from llama_index import (
    GPTKeywordTableIndex,
    GPTListIndex,
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
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
from llama_index.selectors.llm_selectors import LLMSingleSelector, LLMMultiSelector
from config import OPENAI_API_KEY, DISCORD_TOKEN, folder_path, threadCount
from config import DISCORD_TOKEN
from console_logging import enable_logging
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import discord
from discord.ext import commands

enable_logging()

with open("file_index.json") as f:
    file_index = json.load(f)

llm_predictor_chatgpt = LLMPredictor(
    llm=ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",
    )
)
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor_chatgpt, chunk_size_limit=1024
)

documents = {}
vector_indices = {}
index_summaries = {}
for file_name, metadata in file_index.items():
    documents[file_name] = SimpleDirectoryReader(
        input_files=[f"{folder_path}/{file_name}.txt"]
    ).load_data()
    storage_context = StorageContext.from_defaults()

    vector_indices[file_name] = GPTVectorStoreIndex.from_documents(
        documents[file_name],
        service_context=service_context,
        storage_context=storage_context,
    )
    vector_indices[file_name].set_index_id(file_name)

    storage_context.persist(persist_dir=f"./storage/{file_name}")

    index_summaries[file_name] = metadata["description"]


graph = ComposableGraph.from_indices(
    GPTListIndex,
    [index for _, index in vector_indices.items()],
    [summary for _, summary in index_summaries.items()],
    max_keywords_per_chunk=50,
)

root_index = graph.get_index(graph.root_id)
root_index.set_index_id("compare_contrast")

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

graph_description = "This tool contains articles about a fictional Dungeons and Dragons 5E universe called Kazar, including characters, locations, events and lore."
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
intents.typing = True
intents.presences = False
bot = commands.Bot(command_prefix=lambda _, __: [], intents=intents)

bot.remove_command("help")


async def ask(message, question: str):
    try:
        response = router_query_engine.query(question)
        responseString = response.response
        await message.reply(responseString)
    except Exception as e:
        await message.reply(
            "Regrettably, I cannot offer an answer to your question since it does not appear to be relevant to the Lore of Kazar. If you suspect an error, kindly notify my Creator..."
        )


@bot.event
async def on_message(message):
    if message.author.bot:
        return
    if bot.user in message.mentions:
        question = message.content.replace(f"<@!{bot.user.id}>", "").strip()
        prepromptquestinon = question
        await ask(message, prepromptquestinon)

    await bot.process_commands(message)


bot.run(DISCORD_TOKEN)

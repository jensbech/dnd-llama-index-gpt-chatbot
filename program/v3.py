from config import OPENAI_API_KEY, DISCORD_TOKEN, folder_path, threadCount
from llama_index import (
    download_loader,
    GPTVectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from pathlib import Path
from llama_index import (
    GPTListIndex,
    LLMPredictor,
    ServiceContext,
    load_graph_from_storage,
)
from langchain import OpenAI
from llama_index.indices.composability import ComposableGraph
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from llama_index.langchain_helpers.agents import (
    LlamaToolkit,
    create_llama_chat_agent,
    IndexToolConfig,
)
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine

files = ["characters"]  # Specify your .md files here
MarkdownReader = download_loader("MarkdownReader", refresh_cache=True)

loader = MarkdownReader()
doc_set = {}
all_docs = []
for file in files:
    file_docs = loader.load_data(file=Path(f"./data/{file}.md"))
    # insert file metadata into each doc
    for d in file_docs:
        d.extra_info = {"file": file}
    doc_set[file] = file_docs
    all_docs.extend(file_docs)

# initialize simple vector indices + global vector index
service_context = ServiceContext.from_defaults(chunk_size_limit=512)
index_set = {}
for file in files:
    storage_context = StorageContext.from_defaults()
    cur_index = GPTVectorStoreIndex.from_documents(
        doc_set[file],
        service_context=service_context,
        storage_context=storage_context,
    )
    index_set[file] = cur_index
    storage_context.persist(persist_dir=f"./storage/{file}")

# Load indices from disk
index_set = {}
for file in files:
    storage_context = StorageContext.from_defaults(persist_dir=f"./storage/{file}")
    cur_index = load_index_from_storage(storage_context=storage_context)
    index_set[file] = cur_index

# describe each index to help traversal of composed graph
index_summaries = [f"{file} index" for file in files]

# define an LLMPredictor set number of output tokens
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
storage_context = StorageContext.from_defaults()

# define a list index over the vector indices
# allows us to synthesize information across each index
graph = ComposableGraph.from_indices(
    GPTListIndex,
    [index_set[file] for file in files],
    index_summaries=index_summaries,
    service_context=service_context,
    storage_context=storage_context,
)
root_id = graph.root_id

# [optional] save to disk
storage_context.persist(persist_dir=f"./storage/root")

# [optional] load from disk, so you don't need to build graph from scratch
graph = load_graph_from_storage(
    root_id=root_id,
    service_context=service_context,
    storage_context=storage_context,
)

# define a decompose transform
decompose_transform = DecomposeQueryTransform(llm_predictor, verbose=True)

# define custom retrievers
custom_query_engines = {}
for index in index_set.values():
    query_engine = index.as_query_engine()
    query_engine = TransformQueryEngine(
        query_engine,
        query_transform=decompose_transform,
        transform_extra_info={"index_summary": index.index_struct.summary},
    )
    custom_query_engines[index.index_id] = query_engine
custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(
    response_mode="tree_summarize",
    verbose=True,
)

# tool config
graph_config = IndexToolConfig(
    query_engine=custom_query_engines[graph.root_id],
    name=f"Graph Index",
    description="useful for when you want to answer queries that require analyzing multiple Markdown files.",
    tool_kwargs={"return_direct": True},
)

# define toolkit
index_configs = []
for file in files:
    query_engine = custom_query_engines[index_set[file].index_id]
    tool_config = IndexToolConfig(
        query_engine=query_engine,
        name=f"Vector Index {file}",
        description=f"useful for when you want to answer queries about the {file} Markdown file",
        tool_kwargs={"return_direct": True},
    )
    index_configs.append(tool_config)

toolkit = LlamaToolkit(
    index_configs=index_configs + [graph_config],
)

memory = ConversationBufferMemory(memory_key="chat_history")
llm = OpenAI(temperature=0)
agent_chain = create_llama_chat_agent(toolkit, llm, memory=memory, verbose=True)

agent_chain.run(input="what is my name?")

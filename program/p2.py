import json
from langchain import OpenAI
from llama_index import (
    GPTKeywordTableIndex,
    GPTListIndex,
    GPTVectorStoreIndex,
    PromptHelper,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
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
from config import DISCORD_TOKEN
from console_logging import enable_logging
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTVectorStoreIndex
from llama_index.query_engine import RetrieverQueryEngine

enable_logging()

documents = SimpleDirectoryReader("data").load_data()
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents=documents)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
llm_predictor = LLMPredictor(llm=llm)

prompt_helper = PromptHelper(max_input_size=4096, num_output=256, max_chunk_overlap=20)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper
)

index = GPTVectorStoreIndex.from_documents(
    documents=documents, service_context=service_context
)

retriever = index.as_retriever(
    retriever_mode="embedding",
    similarity_tasdop_k=2,
)

query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever, response_mode="compact"
)

response = query_engine.query("summarize expedition 1 in great detail")
print(response)

import json
from llama_index import (
    GPTTreeIndex,
    GPTVectorStoreIndex,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    download_loader,
)
from llama_index.indices.composability import ComposableGraph
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.llm_selectors import LLMSingleSelector
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from config import folder_path

def initialize_chatbot():
    # Prepare data loader
    MarkdownReader = download_loader("MarkdownReader")
    loader = MarkdownReader()

    # Load index data
    with open("file_index.json") as file:
        file_index = json.load(file)

    # Prepare language model
    llm = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo")
    llm_predictor_chatgpt = LLMPredictor(llm)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt, chunk_size_limit=1024)

    # Load documents and build indices
    vector_indices = {}
    index_summaries = {}
    documents = {}

    for md_file in Path(folder_path).glob("**/*.md"):
        file_name = md_file.stem
        metadata = file_index.get(file_name)
        if metadata is not None:
            documents[file_name] = loader.load_data(file=md_file)
            storage_context = StorageContext.from_defaults()
            vector_indices[file_name] = GPTVectorStoreIndex.from_documents(
                documents[file_name], service_context=service_context, storage_context=storage_context
            )
            vector_indices[file_name].set_index_id(file_name)
            storage_context.persist(persist_dir=f"./storage/{file_name}")
            index_summaries[file_name] = "This index contains information about " + metadata["description"]

    return llm_predictor_chatgpt, service_context, vector_indices, index_summaries

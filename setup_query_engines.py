from llama_index import GPTTreeIndex
from llama_index.indices.composability import ComposableGraph
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.llm_selectors import LLMSingleSelector
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform

def setup_query_engines(llm_predictor_chatgpt, service_context, vector_indices, index_summaries):
    # Set up query engines and tools
    decompose_transform = DecomposeQueryTransform(llm_predictor_chatgpt, verbose=True)
    graph = ComposableGraph.from_indices(
        GPTTreeIndex,
        [index for _, index in vector_indices.items()],
        [summary for _, summary in index_summaries.items()],
        max_keywords_per_chunk=50,
    )

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

    graph_description = "This tool contains information about a fictional Dungeons and Dragons 5E universe called Kazar, including characters, locations, events and lore."
    graph_tool = QueryEngineTool.from_defaults(graph_query_engine, description=graph_description)
    query_engine_tools.append(graph_tool)

    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(service_context=service_context),
        query_engine_tools=query_engine_tools,
    )

    return router_query_engine

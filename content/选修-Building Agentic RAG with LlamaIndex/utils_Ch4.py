from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from typing import List, Optional

def get_doc_tools(
    file_path: str,
    name: str,
) -> str:
    """从文档中获取向量问询和摘要问询工具"""

    # 加载文件
    # 文档（document）-节点（nodes）-向量索引（vector_index）
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    embed_model = OpenAIEmbedding(model=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL, api_key="sk-eGmYx2KCAjrvgrTpcoeYSc6GfPlrAJmhzarFDCbjtcJnO5Ha", api_base='https://api.aiproxy.io/v1')
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
    
    # 向量索引（vector_index）-查询引擎（query_engine）
    def vector_query(
        query: str, 
        page_numbers: Optional[List[str]] = None
    ) -> str:
        """
        用于回答关于 metagpt.pdf 论文相关问题。

        如果对这篇论文有具体问题则会有用。
        请将 page_numbers 这一参数设为 None（默认就是），除非你想在某一特定页搜索。  
    
        参数：
            query (str): 所嵌入的字符串问询
            page_numbers (Optional[List[str]]): 通过页数筛选。如果想在所有页就行向量搜索则将其设为 None，否则设定特定的页数。
        """
    
        # 如果 page_numbers 不是一个空列表，那么就是列表；否则，就是 []
        page_numbers = page_numbers or []
        # e.g. 如果 page_numbers = ['1', '2']
        # metadata_dicts 结果为 [{'key': 'page_label', 'value': '1'}, {'key': 'page_label', 'value': '2'}]
        # e.g. 如果 page_numbers = []
        # metadata_dicts 结果为 []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        # 向量索引（vector_index）-查询引擎（query_engine）
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response
        
    # 向量查询工具（vector_query_tool）：文档（document）-节点（nodes）-向量索引（vector_index）-查询引擎（query_engine）
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query
    )
    
    # 文档（document）-节点（nodes）-摘要索引（summary_index）
    summary_index = SummaryIndex(nodes)
    # 摘要索引（summary_index）-摘要查询引擎（summary_query_engine）
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    # 摘要工具（summary_tool）：文档（document）-节点（nodes）-摘要索引（summary_index）-摘要查询引擎（summary_query_engine）
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            "Use ONLY IF you want to get a holistic summary of MetaGPT. "
            "Do NOT use if you have specific questions over MetaGPT."
        ),
    )

    return vector_query_tool, summary_tool
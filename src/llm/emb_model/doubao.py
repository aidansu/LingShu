import logging
import os
from typing import List

from langchain_core.embeddings import Embeddings
from volcenginesdkarkruntime import Ark

logger = logging.getLogger(__name__)


DOUBAO_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"


class DoubaoEmbeddings(Embeddings):
    """使用火山引擎官方 SDK 实现的 Doubao Embeddings"""
    
    def __init__(
        self, 
        model_name: str, 
        api_url: str | None = None, 
        api_key: str | None = None,
        encoding_format: str = "float",
        **kwargs
    ):
        """
        初始化 Doubao Embeddings
        
        Args:
            model_name: 模型名称，如 "doubao-embedding-text-240715"
            api_url: API 基础 URL，默认为 "https://ark.cn-beijing.volces.com/api/v3"
            api_key: API Key，如果未提供则从环境变量 ARK_API_KEY 读取
            encoding_format: 编码格式，默认为 "float"
        """
        self.model_name = model_name
        self.encoding_format = encoding_format
        
        # 如果未提供 api_key，从环境变量读取 ARK_API_KEY
        if api_key is None:
            api_key = os.environ.get("ARK_API_KEY")
        
        if not api_key:
            raise ValueError("api_key 是必需的，请提供 api_key 或设置环境变量 ARK_API_KEY")
        
        # 如果未提供 api_url，使用默认值
        if api_url is None:
            api_url = DOUBAO_API_BASE
        
        # 初始化 Ark 客户端
        self.client = Ark(
            api_key=api_key,
            base_url=api_url
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将文档列表转换为向量列表
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表，每个向量是一个浮点数列表
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
                encoding_format=self.encoding_format
            )
            
            # 提取向量数据
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            logger.error(f'Doubao Embeddings 调用失败: {e}')
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本转换为向量
        
        Args:
            text: 查询文本
            
        Returns:
            向量（浮点数列表）
        """
        return self.embed_documents([text])[0]

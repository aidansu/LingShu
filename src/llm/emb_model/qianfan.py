import logging
import os
from typing import List

import requests

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


QIANFAN_API_BASE = "https://qianfan.baidubce.com/v2/embeddings"


class QianfanEmbeddings(Embeddings):
    """使用百度千帆 API 实现的 Embeddings"""
    
    def __init__(
        self, 
        model_name: str, 
        api_key: str | None = None,
        **kwargs
    ):
        """
        初始化 Qianfan Embeddings
        
        Args:
            model_name: 模型名称，如 "embedding-v1"
            api_key: API Key（Bearer token），如果未提供则从环境变量 QIANFAN_API_KEY 读取
        """
        self.model_name = model_name
        
        # 如果未提供 api_key，从环境变量读取
        if api_key is None:
            api_key = os.environ.get("QIANFAN_API_KEY")
        
        if not api_key:
            raise ValueError("api_key 是必需的，请提供 api_key 或设置环境变量 QIANFAN_API_KEY")
    
        self.api_key = api_key
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将文档列表转换为向量列表
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表，每个向量是一个浮点数列表
        """
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            payload = {
                "model": self.model_name,
                "input": texts
            }
            
            response = requests.post(QIANFAN_API_BASE, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 检查是否有错误
            if "error_code" in result:
                error_msg = result.get("error_msg", "未知错误")
                raise ValueError(f"千帆 Embeddings API 调用失败: {error_msg}")
            
            # 提取向量数据
            data = result.get("data", [])
            if not data:
                raise ValueError("千帆 Embeddings API 返回数据为空")
            
            embeddings = []
            for item in data:
                embedding = item.get("embedding", [])
                if not embedding:
                    raise ValueError("千帆 Embeddings API 返回的 embedding 为空")
                embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            logger.error(f'千帆 Embeddings 调用失败: {e}')
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


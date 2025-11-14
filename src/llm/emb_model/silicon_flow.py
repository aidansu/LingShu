import logging
from typing import List

import requests

from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings

logger = logging.getLogger(__name__)


SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1/embeddings"


class SiliconFlowEmbeddings(HuggingFaceInferenceAPIEmbeddings):
    def __init__(self, model_name: str, api_url: str | None = None, api_key: str | None = None, **kwargs):
        # 如果 api_url 是 None，使用默认的 SILICONFLOW_API_BASE
        if api_url is None:
            api_url = SILICONFLOW_API_BASE
        # api_key 是父类的必需字段，即使为 None 也需要传递
        super().__init__(model_name=model_name, api_url=api_url, api_key=api_key or "", **kwargs)
        # 如果提供了 api_key，更新 headers
        if api_key:
            if not hasattr(self, '_headers') or self._headers is None:
                self._headers = {}
            self._headers["Authorization"] = f"Bearer {api_key}"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batched_embeddings = []

        try:
            headers = self._headers.copy() if self._headers else {}
            # 使用父类的 api_key 字段，如果存在且 headers 中没有 Authorization，则添加
            if self.api_key and "Authorization" not in headers:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.post(
                self._api_url,
                headers=headers,
                json={
                    "input": texts,
                    "model": self.model_name,
                },
            )
            response.raise_for_status()
            response = response.json()
            batched_embeddings.extend(r["embedding"] for r in response["data"])
        except Exception as e:
            logger.error('Exception: %s', e)
            raise

        return batched_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        将单个查询文本转换为向量

        Args:
            text: 查询文本

        Returns:
            向量（浮点数列表）
        """
        return self.embed_documents([text])[0]

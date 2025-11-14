
import requests
from typing import List, Sequence, Any, Dict, Optional, Union

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import Field


def _parse_results(results_json: Any) -> List[Dict[str, Any]]:
    """解析来自 API 的 JSON 响应。"""
    if isinstance(results_json, dict) and "results" in results_json:
        return results_json["results"]
    if isinstance(results_json, list):
        return results_json
    raise ValueError("无法从 API 响应中解析出结果列表。")


class LocalReranker(BaseDocumentCompressor):
    """
    一个功能完整的本地文档重排器，支持 API Key 认证和完整的异步操作。

    它通过调用一个本地 API 来实现重排逻辑，并实现了 BaseDocumentCompressor 的
    同步 `compress_documents` 和异步 `acompress_documents` 方法。
    """
    model: Optional[str] = None
    """Model to use for reranking."""

    top_n: Optional[int] = 3
    """Number of documents to return."""

    # --- 本地服务连接参数 ---
    api_url: str = Field(description="本地 Rerank 服务的 API 地址。")
    api_key: Optional[str] = Field(default=None, description="[可选] 用于认证的 API Key。")

    class Config:
        """Pydantic V1 配置，允许任意类型。"""
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

    def _prepare_headers(self) -> Dict[str, str]:
        """根据是否存在 api_key 准备请求头。"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        top_n: Optional[int] = -1,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            top_n : The number of results to return. If None returns all results.
                Defaults to self.top_n.
        """  # noqa: E501

        if len(documents) == 0:  # to avoid empty api call
            return []
        headers = self._prepare_headers()
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n
        payload = {"model": self.model, "query": query, "documents": docs, "top_n": top_n}

        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=60000
            )
            response.raise_for_status()
            rerank_results = _parse_results(response.json())
            # print(rerank_results)
        except Exception as e:
            # print(f"[Sync] 调用本地 Rerank API 时出错: {e}")
            return []  # 发生错误时返回空列表

        result_dicts = []
        for res in rerank_results:
            result_dicts.append(
                {
                    "index": res["index"],
                    "relevance_score": res["relevance_score"]
                }
            )
        return result_dicts

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        top_n: Optional[int] = -1,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using DashScope's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            top_n: The number of results to return. If None returns all results.
                Defaults to self.top_n.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        compressed = []
        for res in self.rerank(documents, query, top_n=len(documents)):
            doc = documents[res["index"]]
            compressed.append(doc)
        return compressed

"""Qianfan chat models wrapper."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import httpx

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

QIANFAN_API_BASE = "https://qianfan.baidubce.com/v2"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2


class QianfanAPIError(Exception):
    """千帆API调用异常基类."""
    pass


class QianfanAPIRequestError(QianfanAPIError):
    """千帆API请求错误."""
    pass


class QianfanAPIResponseError(QianfanAPIError):
    """千帆API响应解析错误."""
    pass


@contextmanager
def connect_sse(client: Any, method: str, url: str, **kwargs: Any) -> Iterator:
    """Context manager for connecting to an SSE stream.

    Args:
        client: The HTTP client.
        method: The HTTP method.
        url: The URL.
        kwargs: Additional keyword arguments.

    Yields:
        The event source.
    """
    from httpx_sse import EventSource

    with client.stream(method, url, **kwargs) as response:
        yield EventSource(response)


@asynccontextmanager
async def aconnect_sse(
    client: Any, method: str, url: str, **kwargs: Any
) -> AsyncIterator:
    """Async context manager for connecting to an SSE stream.

    Args:
        client: The HTTP client.
        method: The HTTP method.
        url: The URL.
        kwargs: Additional keyword arguments.

    Yields:
        The event source.
    """
    from httpx_sse import EventSource

    async with client.stream(method, url, **kwargs) as response:
        yield EventSource(response)


def _convert_dict_to_message(_dict: Dict[str, Any]) -> BaseMessage:
    role = _dict.get("role")
    content = _dict.get("content", "")
    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        additional_kwargs = {}
        tool_calls = _dict.get("tool_calls", None)
        if tool_calls is not None:
            additional_kwargs["tool_calls"] = tool_calls
        elif "reasoning_content" in _dict:
            additional_kwargs = {"reasoning_content": _dict["reasoning_content"]}
        else:
            additional_kwargs = {}
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    if role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=content,
            tool_call_id=_dict.get("tool_call_id"),
            additional_kwargs=additional_kwargs,
        )

    return ChatMessage(role=role, content=content)  # type: ignore[arg-type]


def _convert_message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
            "name": message.name or message.additional_kwargs.get("name"),
        }
    else:
        raise TypeError(f"Got unknown type '{message.__class__.__name__}'.")
    return message_dict


def _convert_delta_to_message_chunk(
    _dict: Dict[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content", "") or ""
    additional_kwargs = {}
    tool_calls = _dict.get("tool_calls", None)
    if tool_calls is not None:
        additional_kwargs["tool_calls"] = tool_calls
    elif "reasoning_content" in _dict:
        additional_kwargs = {"reasoning_content": _dict["reasoning_content"]}

    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    if role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    return default_class(content=content)  # type: ignore[call-arg]


def _truncate_params(payload: Dict[str, Any]) -> None:
    """Truncate temperature and top_p parameters between [0.01, 0.99].
    """
    temperature = payload.get("temperature")
    top_p = payload.get("top_p")
    if temperature is not None:
        payload["temperature"] = max(0.01, min(0.99, temperature))
    if top_p is not None:
        payload["top_p"] = max(0.01, min(0.99, top_p))


"""千帆聊天模型封装类.

该类封装了百度千帆大模型的API调用，支持同步/异步调用和流式输出。

使用示例:
    ```python
    from src.llm.chat_model.qianfan import ChatQianfan
    
    # 初始化模型
    llm = ChatQianfan(
        model="deepseek-v3.1-250821",
        api_key="your-api-key",
        temperature=0.7,
        enable_thinking=True
    )
    
    # 同步调用
    response = llm.invoke([HumanMessage(content="你好")])
    
    # 异步调用
    response = await llm.ainvoke([HumanMessage(content="你好")])
    ```

Attributes:
    qianfan_api_key: API密钥，可从环境变量QIANFAN_API_KEY获取
    qianfan_api_base: API基础URL，默认为官方地址
    model_name: 模型名称
    temperature: 采样温度，控制输出的随机性
    enable_thinking: 是否启用思考模式
    request_timeout: 请求超时时间（秒）
    max_retries: 最大重试次数
"""


class ChatQianfan(BaseChatModel):

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"qianfan_api_key": "QIANFAN_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "qianfan"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.qianfan_api_base:
            attributes["qianfan_api_base"] = self.qianfan_api_base

        return attributes

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "qianfan-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Qianfan API."""
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        # 根据文档，千帆API可能支持thinking参数，格式可能为 {"type": "enabled"/"disabled"}
        # 如果文档中有其他格式，需要根据实际文档调整
        if self.enable_thinking is not None:
            params["thinking"] = {"type": "enabled" if self.enable_thinking else "disabled"}
        return params

    # client:
    qianfan_api_key: Optional[str] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `QIANFAN_API_KEY` if not provided."""
    qianfan_api_base: Optional[str] = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service
        emulator.
    """
    model_name: str = Field(default="deepseek-v3.1-250821", alias="model")
    """Model name to use."""
    streaming: bool = False
    """Whether to stream the response."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    stop: Optional[Union[List[str], str]] = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""
    enable_thinking: Optional[bool] = Field(default=None)
    """是否启用思考模式: True (启用), False (禁用), None (不设置，由模型决定)"""
    temperature: float = 0.7
    """What sampling temperature to use."""
    top_p: float = 0.7
    """Top-p sampling parameter."""
    frequency_penalty: Optional[float] = None
    """Frequency penalty parameter."""
    presence_penalty: Optional[float] = None
    """Presence penalty parameter."""
    n: int = 1
    """Number of completions to generate."""
    request_timeout: float = Field(default=DEFAULT_TIMEOUT)
    """请求超时时间（秒）."""
    max_retries: int = Field(default=DEFAULT_MAX_RETRIES)
    """最大重试次数."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict[str, Any]) -> Any:
        # 优先从传入的参数获取，然后从环境变量获取
        api_key = values.get("qianfan_api_key") or values.get("api_key")
        if not api_key:
            api_key = os.environ.get("QIANFAN_API_KEY")
        
        if not api_key:
            raise ValueError(
                "qianfan_api_key 是必需的，请提供 api_key 或设置环境变量 QIANFAN_API_KEY"
            )
        
        values["qianfan_api_key"] = api_key
        values["qianfan_api_base"] = values.get("qianfan_api_base") or values.get("base_url") or QIANFAN_API_BASE

        return values

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # 保存 API 密钥和基础 URL，用于 HTTP 请求
        self._api_key = self.qianfan_api_key
        # 规范化 base_url：移除尾部斜杠，确保URL构建一致
        self._api_base = (self.qianfan_api_base or QIANFAN_API_BASE).rstrip('/')

    def _get_url(self) -> str:
        """获取API请求URL."""
        api_base = self._api_base or QIANFAN_API_BASE
        return f"{api_base}/chat/completions"

    def _get_headers(self, stream: bool = False) -> Dict[str, str]:
        """获取API请求headers.
        
        Args:
            stream: 是否为流式请求
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        return headers

    def _extract_chunk_data(
        self, chunk: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, Any], Optional[Dict[str, Any]], str]]:
        """从chunk中提取数据.
        
        Args:
            chunk: API返回的chunk数据
            
        Returns:
            包含(delta, usage, model_name)的元组，如果chunk无效则返回None
        """
        if "choices" not in chunk or len(chunk["choices"]) == 0:
            return None
        
        choice = chunk["choices"][0]
        delta = choice.get("delta", {})
        usage = chunk.get("usage")
        model_name = chunk.get("model", "")
        
        return delta, usage, model_name

    def _process_stream_chunk(
        self,
        chunk: Dict[str, Any],
        default_chunk_class: Type[BaseMessageChunk],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Optional[ChatGenerationChunk]:
        """处理流式响应的单个chunk（同步版本）.
        
        Args:
            chunk: API返回的chunk数据
            default_chunk_class: 默认的消息chunk类型
            run_manager: 回调管理器（可选）
            
        Returns:
            ChatGenerationChunk对象，如果chunk无效则返回None
        """
        chunk_data = self._extract_chunk_data(chunk)
        if chunk_data is None:
            return None
        
        delta, usage, model_name = chunk_data
        chunk_msg = _convert_delta_to_message_chunk(delta, default_chunk_class)
        finish_reason = delta.get("finish_reason")

        generation_info = (
            {
                "finish_reason": finish_reason,
                "token_usage": usage,
                "model_name": model_name,
            }
            if finish_reason is not None
            else None
        )
        chunk_msg = ChatGenerationChunk(
            message=chunk_msg, generation_info=generation_info
        )
        if run_manager:
            run_manager.on_llm_new_token(chunk_msg.text, chunk=chunk_msg)
        return chunk_msg

    async def _process_stream_chunk_async(
        self,
        chunk: Dict[str, Any],
        default_chunk_class: Type[BaseMessageChunk],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> Optional[ChatGenerationChunk]:
        """处理流式响应的单个chunk（异步版本）.
        
        Args:
            chunk: API返回的chunk数据
            default_chunk_class: 默认的消息chunk类型
            run_manager: 异步回调管理器（可选）
            
        Returns:
            ChatGenerationChunk对象，如果chunk无效则返回None
        """
        chunk_data = self._extract_chunk_data(chunk)
        if chunk_data is None:
            return None
        
        delta, usage, model_name = chunk_data
        chunk_msg = _convert_delta_to_message_chunk(delta, default_chunk_class)
        finish_reason = delta.get("finish_reason")

        generation_info = (
            {
                "finish_reason": finish_reason,
                "token_usage": usage,
                "model_name": model_name,
            }
            if finish_reason is not None
            else None
        )
        chunk_msg = ChatGenerationChunk(
            message=chunk_msg, generation_info=generation_info
        )
        if run_manager:
            await run_manager.on_llm_new_token(chunk_msg.text, chunk=chunk_msg)
        return chunk_msg

    def _create_message_dicts(
            self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _normalize_response(self, response: Union[dict, BaseModel]) -> Dict[str, Any]:
        """规范化API响应为字典格式.
        
        Args:
            response: API响应对象
            
        Returns:
            规范化后的字典
        """
        if isinstance(response, dict):
            return response
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        if hasattr(response, 'dict'):
            return response.dict()
        return {}

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        """从API响应创建ChatResult对象.
        
        Args:
            response: API响应对象或字典
            
        Returns:
            ChatResult对象
            
        Raises:
            QianfanAPIResponseError: 当响应格式无效时
        """
        response_dict = self._normalize_response(response)
        
        if "choices" not in response_dict:
            raise QianfanAPIResponseError("API响应中缺少'choices'字段")
        
        generations = []
        for res in response_dict.get("choices", []):
            message = _convert_dict_to_message(res.get("message", {}))
            generation_info = dict(finish_reason=res.get("finish_reason"))
            generations.append(
                ChatGeneration(message=message, generation_info=generation_info)
            )
        
        token_usage = response_dict.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            stream: Optional[bool] = None,
            **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response."""
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {
            **params,
            **kwargs,
            "messages": message_dicts,
            "stream": False,
        }
        _truncate_params(payload)
        
        url = self._get_url()
        headers = self._get_headers(stream=False)
        
        try:
            with httpx.Client(headers=headers, timeout=self.request_timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return self._create_chat_result(response.json())
        except httpx.HTTPStatusError as e:
            error_msg = f"Qianfan API 返回错误状态码: {e.response.status_code}"
            if e.response.text:
                error_msg += f", 响应内容: {e.response.text[:200]}"
            logger.error(error_msg)
            raise QianfanAPIRequestError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Qianfan API 请求失败: {str(e)}"
            logger.error(error_msg)
            raise QianfanAPIRequestError(error_msg) from e
        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Qianfan API 响应解析失败: {str(e)}"
            logger.error(error_msg)
            raise QianfanAPIResponseError(error_msg) from e
        except Exception as e:
            error_msg = f"Qianfan API 调用失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise QianfanAPIError(error_msg) from e

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the chat response in chunks."""
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {**params, **kwargs, "messages": message_dicts, "stream": True}
        _truncate_params(payload)

        url = self._get_url()
        headers = self._get_headers(stream=True)

        try:
            with httpx.Client(headers=headers, timeout=self.request_timeout) as client:
                with connect_sse(
                    client, "POST", url, json=payload
                ) as event_source:
                    for sse in event_source.iter_sse():
                        if not sse.data or sse.data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(sse.data)
                        except json.JSONDecodeError:
                            logger.warning(f"跳过无效的JSON chunk: {sse.data[:100]}")
                            continue
                        
                        chunk_msg = self._process_stream_chunk(
                            chunk, AIMessageChunk, run_manager
                        )
                        if chunk_msg is None:
                            continue
                        
                        yield chunk_msg

                        if chunk_msg.generation_info and chunk_msg.generation_info.get("finish_reason"):
                            break
        except httpx.HTTPStatusError as e:
            error_msg = f"Qianfan API 流式调用返回错误状态码: {e.response.status_code}"
            logger.error(error_msg)
            raise QianfanAPIRequestError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Qianfan API 流式请求失败: {str(e)}"
            logger.error(error_msg)
            raise QianfanAPIRequestError(error_msg) from e
        except Exception as e:
            error_msg = f"Qianfan API 流式调用失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise QianfanAPIError(error_msg) from e

    async def _agenerate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            stream: Optional[bool] = None,
            **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {
            **params,
            **kwargs,
            "messages": message_dicts,
            "stream": False,
        }
        _truncate_params(payload)
        
        url = self._get_url()
        headers = self._get_headers(stream=False)
        
        try:
            async with httpx.AsyncClient(headers=headers, timeout=self.request_timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return self._create_chat_result(response.json())
        except httpx.HTTPStatusError as e:
            error_msg = f"Qianfan API 异步调用返回错误状态码: {e.response.status_code}"
            if e.response.text:
                error_msg += f", 响应内容: {e.response.text[:200]}"
            logger.error(error_msg)
            raise QianfanAPIRequestError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Qianfan API 异步请求失败: {str(e)}"
            logger.error(error_msg)
            raise QianfanAPIRequestError(error_msg) from e
        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Qianfan API 异步响应解析失败: {str(e)}"
            logger.error(error_msg)
            raise QianfanAPIResponseError(error_msg) from e
        except Exception as e:
            error_msg = f"Qianfan API 异步调用失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise QianfanAPIError(error_msg) from e

    async def _astream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {**params, **kwargs, "messages": message_dicts, "stream": True}
        _truncate_params(payload)

        url = self._get_url()
        headers = self._get_headers(stream=True)

        try:
            async with httpx.AsyncClient(headers=headers, timeout=self.request_timeout) as client:
                async with aconnect_sse(
                    client, "POST", url, json=payload
                ) as event_source:
                    async for sse in event_source.aiter_sse():
                        if not sse.data or sse.data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(sse.data)
                        except json.JSONDecodeError:
                            logger.warning(f"跳过无效的JSON chunk: {sse.data[:100]}")
                            continue
                        
                        chunk_msg = await self._process_stream_chunk_async(
                            chunk, AIMessageChunk, run_manager
                        )
                        if chunk_msg is None:
                            continue
                        
                        yield chunk_msg

                        if chunk_msg.generation_info and chunk_msg.generation_info.get("finish_reason"):
                            break
        except httpx.HTTPStatusError as e:
            error_msg = f"Qianfan API 异步流式调用返回错误状态码: {e.response.status_code}"
            logger.error(error_msg)
            raise QianfanAPIRequestError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Qianfan API 异步流式请求失败: {str(e)}"
            logger.error(error_msg)
            raise QianfanAPIRequestError(error_msg) from e
        except Exception as e:
            error_msg = f"Qianfan API 异步流式调用失败: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise QianfanAPIError(error_msg) from e


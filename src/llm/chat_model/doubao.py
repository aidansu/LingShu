"""Doubao chat models wrapper."""

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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

DOUBAO_API_BASE = "https://ark.cn-beijing.volces.com/api/v3"


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)


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


"""
A wrapper around Doubao's Chat API.

To use, you should have the ``volcenginesdkarkruntime`` python package installed, and the
environment variable `ARK_API_KEY` set with your API key.
"""


class ChatDouBao(BaseChatModel):

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"ark_api_key": "ARK_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "doubao"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.ark_api_base:
            attributes["ark_api_base"] = self.ark_api_base

        return attributes

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "doubao-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Doubao API."""
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.thinking_type is not None:
            params["thinking"] = {"type": self.thinking_type}
        return params

    # client:
    ark_api_key: Optional[str] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `ARK_API_KEY` if not provided."""
    ark_api_base: Optional[str] = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service
        emulator.
    """
    model_name: str = Field(default="doubao-seed-1-6-251015", alias="model")
    """Model name to use."""
    streaming: bool = False
    """Whether to stream the response."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    stop: Optional[Union[List[str], str]] = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""
    thinking_type: Optional[str] = Field(default=None)
    """思考模式类型: "disabled" (不使用深度思考能力), "enabled" (使用深度思考能力), "auto" (模型自行判断是否使用深度思考能力)"""
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

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @field_validator("thinking_type")
    @classmethod
    def validate_thinking_type(cls, v: Optional[str]) -> Optional[str]:
        """验证 thinking_type 的值"""
        if v is not None and v not in ("disabled", "enabled", "auto"):
            raise ValueError(
                f'thinking_type 必须是 "disabled", "enabled" 或 "auto" 之一，当前值: {v}'
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict[str, Any]) -> Any:
        # 优先从传入的参数获取，然后从环境变量获取
        api_key = values.get("ark_api_key") or values.get("api_key")
        if not api_key:
            api_key = os.environ.get("ARK_API_KEY")
        
        if not api_key:
            raise ValueError(
                "ark_api_key 是必需的，请提供 api_key 或设置环境变量 ARK_API_KEY"
            )
        
        values["ark_api_key"] = api_key
        values["ark_api_base"] = values.get("ark_api_base") or values.get("base_url") or DOUBAO_API_BASE

        return values

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # 保存 API 密钥和基础 URL，用于 HTTP 请求
        self._api_key = self.ark_api_key
        # 确保 base_url 有默认值
        self._api_base = self.ark_api_base or DOUBAO_API_BASE
        # 确保 URL 以 / 结尾，以便正确拼接路径
        if self._api_base and not self._api_base.endswith('/'):
            self._api_base = self._api_base.rstrip('/')

    def _create_message_dicts(
            self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.dict() if hasattr(response, 'dict') else response.model_dump() if hasattr(response, 'model_dump') else {}
        
        for res in response.get("choices", []):
            message = _convert_dict_to_message(res.get("message", {}))
            generation_info = dict(finish_reason=res.get("finish_reason"))
            generations.append(
                ChatGeneration(message=message, generation_info=generation_info)
            )
        token_usage = response.get("usage", {})
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
        
        import httpx
        
        # 确保 base_url 有值
        api_base = self._api_base or DOUBAO_API_BASE
        url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            with httpx.Client(headers=headers, timeout=60) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return self._create_chat_result(response.json())
        except Exception as e:
            logger.error(f"Doubao API 调用失败: {e}")
            raise

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

        default_chunk_class = AIMessageChunk
        
        import httpx
        
        # 确保 base_url 有值
        api_base = self._api_base or DOUBAO_API_BASE
        url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        try:
            with httpx.Client(headers=headers, timeout=60) as client:
                with connect_sse(
                    client, "POST", url, json=payload
                ) as event_source:
                    for sse in event_source.iter_sse():
                        if not sse.data or sse.data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(sse.data)
                        except json.JSONDecodeError:
                            continue
                        
                        if "choices" not in chunk or len(chunk["choices"]) == 0:
                            continue
                        
                        choice = chunk["choices"][0]
                        usage = chunk.get("usage", None)
                        model_name = chunk.get("model", "")
                        
                        delta = choice.get("delta", {})
                        chunk_msg = _convert_delta_to_message_chunk(delta, default_chunk_class)
                        finish_reason = choice.get("finish_reason", None)

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
                        yield chunk_msg

                        if finish_reason is not None:
                            break
        except Exception as e:
            logger.error(f"Doubao API 流式调用失败: {e}")
            raise

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
        
        import httpx
        
        # 确保 base_url 有值
        api_base = self._api_base or DOUBAO_API_BASE
        url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            async with httpx.AsyncClient(headers=headers, timeout=60) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return self._create_chat_result(response.json())
        except Exception as e:
            logger.error(f"Doubao API 异步调用失败: {e}")
            raise

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

        default_chunk_class = AIMessageChunk
        
        import httpx
        
        # 确保 base_url 有值
        api_base = self._api_base or DOUBAO_API_BASE
        url = f"{api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        try:
            async with httpx.AsyncClient(headers=headers, timeout=60) as client:
                async with aconnect_sse(
                    client, "POST", url, json=payload
                ) as event_source:
                    async for sse in event_source.aiter_sse():
                        if not sse.data or sse.data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(sse.data)
                        except json.JSONDecodeError:
                            continue
                        
                        if "choices" not in chunk or len(chunk["choices"]) == 0:
                            continue
                        
                        choice = chunk["choices"][0]
                        usage = chunk.get("usage", None)
                        model_name = chunk.get("model", "")
                        
                        delta = choice.get("delta", {})
                        chunk_msg = _convert_delta_to_message_chunk(delta, default_chunk_class)
                        finish_reason = choice.get("finish_reason", None)

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
                        yield chunk_msg

                        if finish_reason is not None:
                            break
        except Exception as e:
            logger.error(f"Doubao API 异步流式调用失败: {e}")
            raise


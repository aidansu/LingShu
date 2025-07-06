"""SiliconFlow chat models wrapper."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
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
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

API_TOKEN_TTL_SECONDS = 3 * 60
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1/chat/completions"

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
A wrapper around SiliconFlow's Chat API.

To use, you should have the ``siliconflow`` python package installed, and the
environment variable ``SILICON_FLOW_API_KEY`` set with your API key.
"""


class ChatSiliconFlow(BaseChatModel):

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"siliconflow_api_key": "SILICONFLOW_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "siliconflow"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.siliconflow_api_base:
            attributes["siliconflow_api_base"] = self.siliconflow_api_base

        return attributes

    @property
    def _llm_type(self) -> str:
        """Return the type of chat model."""
        return "siliconflow-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "enable_thinking": self.enable_thinking,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params


    # client:
    siliconflow_api_key: Optional[str] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `SILICON_FLOW_API_KEY` if not provided."""
    siliconflow_api_base: Optional[str] = Field(default=None, alias="api_base")
    """Base URL path for API requests, leave blank if not using a proxy or service
        emulator.
    """
    model_name: str = Field(default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", alias="model")
    """Model name to use."""
    streaming: bool = False
    """Seed for generation"""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    stop: Optional[Union[List[str], str]] = Field(default=None, alias="stop_sequences")
    """Default stop sequences."""
    enable_thinking: bool = True
    thinking_budget: int = 4096
    temperature: float = 0.7
    """What sampling temperature to use."""
    top_p: float = 0.7
    top_k: int = 50
    frequency_penalty: Optional[float] = None
    n: int = 1
    response_format: Dict[str, Any] = None



    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict[str, Any]) -> Any:
        values["siliconflow_api_key"] = get_from_dict_or_env(
            values, ["siliconflow_api_key", "api_key"], "SILICONFLOW_API_KEY"
        )
        values["siliconflow_api_base"] = get_from_dict_or_env(
            values, "siliconflow_api_base", "SILICONFLOW_API_BASE", default=SILICONFLOW_API_BASE
        )

        return values

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
            response = response.dict()
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
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

        if self.siliconflow_api_key is None:
            raise ValueError("Did not find siliconflow_api_key.")
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {
            **params,
            **kwargs,
            "messages": message_dicts,
            "stream": False,
        }
        _truncate_params(payload)
        headers = {
            "Authorization": f"Bearer {self.siliconflow_api_key}",
            "Accept": "application/json",
        }
        import httpx

        with httpx.Client(headers=headers, timeout=60) as client:
            response = client.post(self.siliconflow_api_base, json=payload)  # type: ignore[arg-type]
            response.raise_for_status()
        return self._create_chat_result(response.json())

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the chat response in chunks."""
        if self.siliconflow_api_key is None:
            raise ValueError("Did not find siliconflow_api_key.")
        if self.siliconflow_api_base is None:
            raise ValueError("Did not find siliconflow_api_base.")
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {**params, **kwargs, "messages": message_dicts, "stream": True}
        _truncate_params(payload)
        headers = {
            "Authorization": f"Bearer {self.siliconflow_api_key}",
            "Accept": "application/json",
        }

        default_chunk_class = AIMessageChunk
        import httpx

        with httpx.Client(headers=headers, timeout=60) as client:
            with connect_sse(
                    client, "POST", self.siliconflow_api_base, json=payload
            ) as event_source:
                for sse in event_source.iter_sse():
                    chunk = json.loads(sse.data)
                    if len(chunk["choices"]) == 0:
                        continue
                    choice = chunk["choices"][0]
                    usage = chunk.get("usage", None)
                    model_name = chunk.get("model", "")
                    chunk = _convert_delta_to_message_chunk(
                        choice["delta"], default_chunk_class
                    )
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
                    chunk = ChatGenerationChunk(
                        message=chunk, generation_info=generation_info
                    )
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    yield chunk

                    if finish_reason is not None:
                        break

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

        if self.siliconflow_api_key is None:
            raise ValueError("Did not find siliconflow_api_key.")
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {
            **params,
            **kwargs,
            "messages": message_dicts,
            "stream": False,
        }
        _truncate_params(payload)
        headers = {
            "Authorization": f"Bearer {self.siliconflow_api_key}",
            "Accept": "application/json",
        }
        import httpx

        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            response = await client.post(self.siliconflow_api_base, json=payload)  # type: ignore[arg-type]
            response.raise_for_status()
        return self._create_chat_result(response.json())

    async def _astream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.siliconflow_api_key is None:
            raise ValueError("Did not find siliconflow_api_key.")
        if self.siliconflow_api_base is None:
            raise ValueError("Did not find siliconflow_api_base.")
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {**params, **kwargs, "messages": message_dicts, "stream": True}
        _truncate_params(payload)
        headers = {
            "Authorization": f"Bearer {self.siliconflow_api_key}",
            "Accept": "application/json",
        }

        default_chunk_class = AIMessageChunk
        import httpx

        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            async with aconnect_sse(
                    client, "POST", self.siliconflow_api_base, json=payload
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    chunk = json.loads(sse.data)
                    if len(chunk["choices"]) == 0:
                        continue
                    choice = chunk["choices"][0]
                    usage = chunk.get("usage", None)
                    model_name = chunk.get("model", "")
                    chunk = _convert_delta_to_message_chunk(
                        choice["delta"], default_chunk_class
                    )
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
                    chunk = ChatGenerationChunk(
                        message=chunk, generation_info=generation_info
                    )
                    if run_manager:
                        await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    yield chunk

                    if finish_reason is not None:
                        break

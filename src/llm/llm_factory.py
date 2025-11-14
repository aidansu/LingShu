from langchain_core.output_parsers import BaseTransformOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Any, Dict
import os
import logging

from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatZhipuAI, ChatTongyi, MoonshotChat
from langchain_community.document_compressors import DashScopeRerank
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings

from src.configs.config import services
from src.llm.chat_model.local_chat_open_ai import LocalChatOpenAI
from src.llm.chat_model.silicon_flow import ChatSiliconFlow
from src.llm.chat_model.doubao import ChatDouBao
from src.llm.chat_model.qianfan import ChatQianfan
from src.llm.emb_model.local_embeddings import LocalEmbeddings
from src.llm.emb_model.silicon_flow import SiliconFlowEmbeddings
from src.llm.emb_model.doubao import DoubaoEmbeddings
from src.llm.emb_model.qianfan import QianfanEmbeddings
from src.llm.rerank_model.local_rerank import LocalReranker

logger = logging.getLogger(__name__)


class LLMInitializationError(Exception):
    pass


def get_api_key(provider_name: str) -> str:
    api_key_env = f"{provider_name.upper()}_API_KEY"
    api_key = os.getenv(api_key_env)
    return api_key


def create_common_params(provider_name: str) -> Dict[str, str]:
    api_key = get_api_key(provider_name)
    return {"api_key": api_key}


def init_chat_llm(model: str = None, thinking: bool = False, stream: bool = False, temperature: float = None) -> Any:

    client_initializers = {
        "local": lambda provider, model, params, temperature: LocalChatOpenAI(
            model=model,
            base_url=provider.get("base_url"),
            extra_body={"enable_thinking": thinking},
            max_retries=2,  # 设置最大重试次数
            request_timeout=500.0,  # 设置请求超时时间
            temperature=temperature,
            **params
        ),
        "openai": lambda provider, model, params, temperature: init_chat_model(
            model=model,
            model_provider="openai",
            base_url=provider.get("base_url"),
            extra_body={"enable_thinking": thinking},
            temperature=temperature,
            **params
        ),
        "deepseek": lambda provider, model, params, temperature: init_chat_model(
            model=model,
            model_provider="deepseek",
            base_url=provider.get("base_url"),
            temperature=temperature,
            **params
        ),
        "dashscope": lambda provider, model, params, temperature: ChatTongyi(
            model=model,
            base_url=provider.get("base_url"),
            model_kwargs={"enable_thinking": True if stream and thinking else False},
            temperature=temperature,
            **params
        ),
        "moonshot": lambda provider, model, params, temperature: MoonshotChat(
            model=model,
            base_url=provider.get("base_url"),
            temperature=temperature,
            **params
        ),
        "zhipuai": lambda provider, model, params, temperature: ChatZhipuAI(
            model=model,
            base_url=provider.get("base_url"),
            temperature=temperature,
            model_kwargs={"thinking": {"type": "enabled" if thinking else "disabled"}},
            **params
        ),
        "ark": lambda provider, model, params, temperature: ChatDouBao(
            model=model,
            base_url=provider.get("base_url"),
            temperature=temperature,
            thinking_type="enabled" if thinking else "disabled",
            **params
        ),
        "qianfan": lambda provider, model, params, temperature: ChatQianfan(
            model=model,
            base_url=provider.get("base_url"),
            enable_thinking=thinking if thinking else None,
            temperature=temperature,
            **params
        ),
        "siliconflow": lambda provider, model, params, temperature: ChatSiliconFlow(
            model=model,
            base_url=provider.get("base_url"),
            enable_thinking=thinking,
            temperature=temperature,
            **params
        ),

    }

    for provider_name, provider in services.llm_models.items():
        models = provider.get("models", {})
        for model_id, model_config in models.items():
            # 如果没有传入模型名称，则使用第一个为默认模型
            model = model or model_config.get("model_name", [])[0]
            if model in model_config.get("model_name", []):
                client_type = provider.get("client")
                common_params = create_common_params(provider_name)
                
                # 如果没有传入 temperature，则使用配置中的默认值
                temp = temperature if temperature is not None else model_config.get("temperature", 0.7)

                initializer = client_initializers.get(client_type)
                if initializer:
                    return initializer(provider, model, common_params, temp)

    raise LLMInitializationError(f"未配置模型: {model}")


def init_embedding_model(model: str = None) -> Any:
    client_initializers = {
        "local": lambda provider, model: LocalEmbeddings(
            model_name=model,
            api_url=provider.get("api_url"),
            api_key=provider.get("api_key") or ""
        ),
        "dashscope": lambda provider, model: DashScopeEmbeddings(
            model=model,
            dashscope_api_key=provider.get("api_key") or get_api_key("dashscope")
        ),
        "zhipuai": lambda provider, model: ZhipuAIEmbeddings(
            model=model,
            api_key=provider.get("api_key") or get_api_key("zhipuai")
        ),
        "ark": lambda provider, model: DoubaoEmbeddings(
            model_name=model,
            api_key=provider.get("api_key") or get_api_key("ark")
        ),
        "qianfan": lambda provider, model: QianfanEmbeddings(
            model_name=model,
            api_key=provider.get("api_key") or get_api_key("qianfan")
        ),
        "siliconflow": lambda provider, model: SiliconFlowEmbeddings(
            model_name=model,
            api_key=provider.get("api_key") or get_api_key("siliconflow")
        )
    }

    for provider_name, provider in services.emb_models.items():
        provider_model = provider.get("model")
        # 处理单个字符串或列表的情况
        if isinstance(provider_model, list):
            # 如果是列表，检查传入的模型是否在列表中
            if model is None:
                # 如果没有传入模型名称，则使用列表中的第一个为默认模型
                model = provider_model[0]
            elif model not in provider_model:
                # 如果传入的模型不在列表中，跳过这个provider
                continue
        else:
            # 如果是单个字符串，保持原有逻辑
            if model is None or model == provider_model:
                model = model or provider_model
            else:
                # 如果传入的模型不匹配，跳过这个provider
                continue
        
        initializer = client_initializers.get(provider_name)
        if initializer:
            try:
                return initializer(provider, model)
            except Exception as e:
                logger.error(f"初始化embedding模型失败 {provider_name}: {e}")
                raise LLMInitializationError(f"初始化embedding模型失败: {e}")

    raise LLMInitializationError(f"未配置模型: {model}")


def init_rerank_model(model: str = None) -> Any:
    client_initializers = {
        "local": lambda provider, model: LocalReranker(
            model=model,
            api_url=provider.get("api_url"),
            api_key=provider.get("api_key") or ""
        ),
        "dashscope": lambda _, model: DashScopeRerank(
            model=model,
            dashscope_api_key=get_api_key("dashscope")
        )
    }

    for provider_name, provider in services.rerank_models.items():
        # 如果没有传入模型名称，则使用配置中的模型名称
        model = model or provider.get("model")
        if model == provider.get("model"):
            initializer = client_initializers.get(provider_name)
            if initializer:
                try:
                    return initializer(provider, model)
                except Exception as e:
                    logger.error(f"初始化rerank模型失败 {provider_name}: {e}")
                    raise LLMInitializationError(f"初始化rerank模型失败: {e}")

    raise LLMInitializationError(f"未配置模型: {model}")


async def llm_ainvoke(llm: Any, output_parser: BaseTransformOutputParser, prompt_name: str, temperature: float = None, **kwargs):
    prompt_template = services.prompts.get(prompt_name)
    full_prompt = prompt_template.format(**kwargs)
    prompt = PromptTemplate.from_template(prompt_template)
    
    # 如果传入了 temperature 参数，则动态设置模型的 temperature
    if temperature is not None:
        # 检查模型是否有 temperature 属性
        if hasattr(llm, 'temperature'):
            llm.temperature = temperature
        elif hasattr(llm, 'model_kwargs'):
            llm.model_kwargs['temperature'] = temperature
        elif hasattr(llm, 'extra_body'):
            llm.extra_body['temperature'] = temperature
    
    chain = prompt | llm | output_parser
    result = await chain.ainvoke(kwargs)
    logger.info(f"大模型返回结果：{result}")
    return full_prompt, result


async def llm_astream(llm: Any, output_parser: BaseTransformOutputParser, prompt_name: str, temperature: float = None, **kwargs):
    prompt_template = services.prompts.get(prompt_name)
    prompt = PromptTemplate.from_template(prompt_template)
    
    # 如果传入了 temperature 参数，则动态设置模型的 temperature
    if temperature is not None:
        # 检查模型是否有 temperature 属性
        if hasattr(llm, 'temperature'):
            llm.temperature = temperature
        elif hasattr(llm, 'model_kwargs'):
            llm.model_kwargs['temperature'] = temperature
        elif hasattr(llm, 'extra_body'):
            llm.extra_body['temperature'] = temperature
    
    chain = prompt | llm | output_parser
    async for response in chain.astream(kwargs):
        yield response

from typing import Any, Dict
import os

from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatZhipuAI

from src.configs.config import LLM_CONFIG
from src.llm.model.ChatSiliconFlow import ChatSiliconFlow


class LLMInitializationError(Exception):
    pass


def get_api_key(provider_name: str) -> str:
    api_key_env = f"{provider_name.upper()}_API_KEY"
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise LLMInitializationError(f"未找到环境变量 {api_key_env}")
    return api_key


def create_common_params(provider_name: str) -> Dict[str, str]:
    api_key = get_api_key(provider_name)
    return {"api_key": api_key}


def init_chat_llm(model: str) -> Any:
    model_providers = LLM_CONFIG["model_providers"]

    client_initializers = {
        "openai": lambda provider, model, params: init_chat_model(
            model=model,
            model_provider="openai",
            base_url=provider.get("base_url"),
            # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
            # 使用Qwen3开源版模型时，若未启用流式输出，请添加下面参数，否则会报错
            extra_body={"enable_thinking": False},
            **params
        ),
        "deepseek": lambda provider, model, params: init_chat_model(
            model=model,
            model_provider="deepseek",
            **params
        ),
        "zhipuai": lambda _, model, params: ChatZhipuAI(model=model, **params),
        "siliconflow": lambda _, model, params: ChatSiliconFlow(model=model, **params),
    }

    for provider_name, provider in model_providers.items():
        models = provider.get("models", {})
        for model_id, model_config in models.items():
            if model in model_config.get("model_name", []):
                client_type = provider.get("client")
                common_params = create_common_params(provider_name)

                initializer = client_initializers.get(client_type)
                if initializer:
                    return initializer(provider, model, common_params)

    raise LLMInitializationError(f"未配置模型: {model}")

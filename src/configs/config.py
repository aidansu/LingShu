import os
import logging

import yaml
from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError


class ModelConfig(BaseModel):
    model_name: List[str]
    thinking: bool = False


class ProviderConfig(BaseModel):
    client: str
    base_url: str = None
    models: Dict[str, ModelConfig]


def load_logging_config(config_path: str = "src/configs/logging_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # 检查所有文件处理器的路径，并确保其目录存在
    for handler in config.get('handlers', {}).values():
        if 'filename' in handler:
            filename = handler['filename']
            # 创建日志目录
            log_dir = os.path.dirname(filename)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
    logging.config.dictConfig(config)


def load_llm_config(config_path: str = "src/configs/llm_config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_config(raw_config: dict) -> dict:
    validated = {"model_providers": {}}
    for provider_name, config in raw_config["model_providers"].items():
        try:
            validated["model_providers"][provider_name] = ProviderConfig(**config).dict()
        except ValidationError as e:
            raise ValueError(f"Invalid config for {provider_name}: {str(e)}")
    return validated


LLM_CONFIG = validate_config(load_llm_config())

import yaml
from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError


class ModelConfig(BaseModel):
    model_name: List[str]


class ProviderConfig(BaseModel):
    client: str
    base_url: str = None
    models: Dict[str, ModelConfig]


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

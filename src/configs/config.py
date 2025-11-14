import os
from dataclasses import dataclass
from logging.config import dictConfig
from pathlib import Path
import typing as t
from types import SimpleNamespace

import yaml
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from starlette.middleware import Middleware
from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError
from functools import lru_cache
from pydantic import TypeAdapter

from src.db.postgres import base_entity
from src.db.redis import redis_database
from src.services import mysecrets
from src.utils.config_util import get_yaml
from src.utils.json_serialize_util import json_serialize, json_deserialize

log_directory = Path("log")
log_directory.mkdir(parents=True, exist_ok=True)
load_dotenv(override=True)


class CorsConfig(BaseModel):
    allow_origins: list[str]
    allow_methods: list[str]
    allow_headers: list[str]
    allow_credentials: bool


class HttpConfig(BaseModel):
    cors: CorsConfig


class MilvusConfig(BaseModel):
    host: str
    port: int = 19530
    user: str | None = None
    password: str | None = None
    db: str | None = None
    token: str | None = None
    uri: str | None = None

    def get_connection_params(self) -> Dict[str, Any]:
        """获取Milvus连接参数"""
        params = {}

        # 如果提供了完整的URI，使用URI
        if self.uri:
            params["uri"] = self.uri
        else:
            # 否则使用host和port构建URI
            params["uri"] = f"http://{self.host}:{self.port}"

        # 如果提供了token，使用token
        if self.token:
            params["token"] = self.token
        elif self.user and self.password:
            # 否则使用用户名密码构建token
            params["token"] = f"{self.user}:{self.password}"

        # 如果指定了数据库名
        if self.db:
            params["db_name"] = self.db

        return params


class ElasticsearchConfig(BaseModel):
    host: str
    ports: List[int] = [9200]
    username: str | None = None
    password: str | None = None
    use_ssl: bool = False
    verify_certs: bool = True
    ca_certs: str | None = None
    client_cert: str | None = None
    client_key: str | None = None

    def get_connection_params(self) -> Dict[str, Any]:
        """获取Elasticsearch连接参数"""
        # 构建服务器列表
        hosts = []
        for port in self.ports:
            hosts.append({
                "host": self.host,
                "port": str(port)
            })

        params = {
            "hosts": hosts,
            "use_ssl": self.use_ssl,
            "verify_certs": self.verify_certs
        }

        # 设置认证信息
        if self.username and self.password:
            params["http_auth"] = (self.username, self.password)

        # SSL证书配置
        if self.ca_certs:
            params["ca_certs"] = self.ca_certs
        if self.client_cert:
            params["client_cert"] = self.client_cert
        if self.client_key:
            params["client_key"] = self.client_key

        return params


class ServicesConfig(BaseModel):
    http: HttpConfig
    logging: t.Any
    postgres: base_entity.SqlConfig
    redis: redis_database.RedisConfig
    milvus: MilvusConfig | None = None
    es: ElasticsearchConfig | None = None
    llm_models: Dict[str, Any]
    emb_models: Dict[str, Any]
    rerank_models: Dict[str, Any]


class ModelConfig(BaseModel):
    model_name: List[str]
    thinking: bool = False
    temperature: float = 0.7


class LLMConfig(BaseModel):
    client: str
    base_url: str | None = None
    models: Dict[str, ModelConfig]


class EmbConfig(BaseModel):
    api_url: str | None = None
    api_key: str | None = None
    model: str | List[str]


class RerankConfig(BaseModel):
    api_url: str | None = None
    api_key: str | None = None
    model: str


@dataclass
class Services:
    middleware: list[Middleware]
    postgres: AsyncEngine | None
    redis: redis_database.Redis | None
    milvus: MilvusConfig | None
    es: ElasticsearchConfig | None
    llm_models: Dict[str, Any]
    emb_models: Dict[str, Any]
    rerank_models: Dict[str, Any]
    prompts: dict
    _cfg: ServicesConfig | None = None  # 存储配置引用，用于延迟加载
    
    def ensure_postgres(self) -> AsyncEngine:
        """确保 postgres 连接已创建，如果未创建则创建"""
        if self.postgres is None:
            if not self._cfg:
                raise ValueError("配置未初始化，无法创建 PostgreSQL 连接")
            if has_placeholder(self._cfg.postgres.url):
                raise ValueError(
                    f"PostgreSQL 配置不完整，URL 中包含占位符: {self._cfg.postgres.url}。"
                    "请先配置完整的数据库连接信息。"
                )
            self.postgres = create_async_engine(
                url=self._cfg.postgres.url,
                json_serializer=json_serialize,
                json_deserializer=json_deserialize,
                pool_size=self._cfg.postgres.pool_size,
                max_overflow=self._cfg.postgres.max_overflow,
                pool_recycle=self._cfg.postgres.pool_recycle,
                pool_pre_ping=True,
                echo=False,
            )
        return self.postgres
    
    def ensure_redis(self) -> redis_database.Redis:
        """确保 redis 连接已创建，如果未创建则创建"""
        if self.redis is None:
            if not self._cfg:
                raise ValueError("配置未初始化，无法创建 Redis 连接")
            if has_placeholder(self._cfg.redis.host):
                raise ValueError(
                    f"Redis 配置不完整，host 中包含占位符: {self._cfg.redis.host}。"
                    "请先配置完整的 Redis 连接信息。"
                )
            self.redis = redis_database.Redis(self._cfg.redis)
        return self.redis


def get_project_root() -> Path:
    # 本文件在 src/configs/ 下，项目根目录为 src 的上一级
    return Path(__file__).resolve().parents[2]


def get_yaml_abs_path(rel_path: str) -> Path:
    root = get_project_root()
    return root / rel_path


@lru_cache(maxsize=1)
def load_config():
    secrets_obj = mysecrets.from_config(
        TypeAdapter(mysecrets.SecretsConfig).validate_python({
            "type": os.environ.get("SECRETS_PROVIDER", "env"),
            "vault_url": os.environ.get("KEY_VAULT_URL"),
        })
    )
    context = SimpleNamespace(secrets=secrets_obj)
    return get_yaml("src/configs/config.yaml", context)


@lru_cache(maxsize=1)
def load_prompts():
    yaml_path = get_yaml_abs_path("src/configs/prompts.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        if data is None:
            return {}
        return data["prompts"] if "prompts" in data else data


def has_placeholder(value: str) -> bool:
    """检查字符串中是否包含占位符"""
    if not isinstance(value, str):
        return False
    placeholders = ['<ip>', '<port>', '<username>', '<password>', '<database>']
    return any(placeholder in value for placeholder in placeholders)


def check_config_ready(cfg: ServicesConfig) -> bool:
    """检查配置是否完整，不包含占位符"""
    # 检查 postgres URL
    if has_placeholder(cfg.postgres.url):
        return False
    
    # 检查 redis host
    if has_placeholder(cfg.redis.host):
        return False
    
    # 检查 milvus host（如果配置了）
    if cfg.milvus and has_placeholder(cfg.milvus.host):
        return False
    
    # 检查 es host（如果配置了）
    if cfg.es and has_placeholder(cfg.es.host):
        return False
    
    return True


def from_config(cfg: ServicesConfig, prompts: dict) -> Services:
    dictConfig(cfg.logging)
    middleware = [Middleware(cls=CORSMiddleware, **cfg.http.cors.model_dump())]

    # 检查配置是否完整
    if not check_config_ready(cfg):
        # 如果配置不完整，创建占位符对象，延迟创建连接
        return Services(
            middleware=middleware,
            postgres=None,  # 延迟创建
            redis=None,  # 延迟创建
            milvus=cfg.milvus,
            es=cfg.es,
            llm_models=get_llm_models_config(cfg.llm_models),
            emb_models=get_embedding_models_config(cfg.emb_models),
            rerank_models=get_rerank_models_config(cfg.rerank_models),
            prompts=prompts,
            _cfg=cfg  # 存储配置引用，用于延迟加载
        )

    # 数据库连接配置
    postgres = create_async_engine(
        url=cfg.postgres.url,
        json_serializer=json_serialize,
        json_deserializer=json_deserialize,
        pool_size=cfg.postgres.pool_size,  # 连接池大小
        max_overflow=cfg.postgres.max_overflow,  # 最大溢出连接数
        pool_recycle=cfg.postgres.pool_recycle,  # 连接回收时间
        pool_pre_ping=True,  # 启用连接前ping检查
        echo=False,  # 关闭SQL日志
    )

    llm_models = get_llm_models_config(cfg.llm_models)
    emb_models = get_embedding_models_config(cfg.emb_models)
    rerank_models = get_rerank_models_config(cfg.rerank_models)
    return Services(
        middleware=middleware,
        postgres=postgres,
        redis=redis_database.Redis(cfg.redis),
        milvus=cfg.milvus,
        es=cfg.es,
        llm_models=llm_models,
        emb_models=emb_models,
        rerank_models=rerank_models,
        prompts=prompts,
        _cfg=cfg  # 存储配置引用，用于延迟加载
    )


def get_llm_models_config(raw_config: dict) -> dict:
    llm_config = {}
    for provider_name, config in raw_config.items():
        try:
            llm_config[provider_name] = LLMConfig(**config).dict()
        except ValidationError as e:
            raise ValueError(f"Invalid config for {provider_name}: {str(e)}")
    return llm_config


def get_embedding_models_config(raw_config: dict) -> dict:
    emb_config = {}
    for emb_name, config in raw_config.items():
        try:
            emb_config[emb_name] = EmbConfig(**config).dict()
        except ValidationError as e:
            raise ValueError(f"Invalid config : {str(e)}")
    return emb_config


def get_rerank_models_config(raw_config: dict) -> dict:
    rerank_config = {}
    for rerank_name, config in raw_config.items():
        try:
            rerank_config[rerank_name] = RerankConfig(**config).dict()
        except ValidationError as e:
            raise ValueError(f"Invalid config : {str(e)}")
    return rerank_config


def get_default_config():
    config_name = os.environ.get("CONFIG_NAME", "dev")
    prompts = load_prompts()
    # 最后再解析 config.yaml文件
    cfg = load_config()
    return ServicesConfig.model_validate(cfg[config_name]), prompts


_configs, _prompts = get_default_config()
_services: Services | None = None


def get_services() -> Services:
    """获取 services 实例，延迟加载"""
    global _services
    if _services is None:
        _services = from_config(_configs, _prompts)
    return _services


# 为了向后兼容，提供一个 services 属性访问器
class _ServicesProxy:
    """Services 代理类，提供延迟加载和属性访问"""
    
    def __getattr__(self, name: str):
        services = get_services()
        value = getattr(services, name)
        
        # 如果是 postgres 或 redis，且为 None，尝试延迟创建
        if name == 'postgres' and value is None:
            return services.ensure_postgres()
        if name == 'redis' and value is None:
            return services.ensure_redis()
        
        return value


services = _ServicesProxy()

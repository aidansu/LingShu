
from typing import Any, Dict
from abc import ABC, abstractmethod

from src.common.constants import VECTOR_DATABASE_NAME


class BaseCollection(ABC):
    """
    Milvus Collection 基类
    定义了集合的基本属性和抽象方法
    """

    def __init__(self):
        self.id: int | None = None
        self.text: str | None = None
        self.vector: list[float] | None = None

    @classmethod
    @abstractmethod
    def get_collection_name(cls) -> str:
        """
        获取集合名称，子类必须实现
        """
        pass

    @classmethod
    def get_database_name(cls) -> str:
        """
        获取数据库名称，默认为 chat_bi
        """
        return VECTOR_DATABASE_NAME
    
    @classmethod
    @abstractmethod
    def get_schema_config(cls) -> Dict[str, Any]:
        """
        获取collection的schema配置，子类必须实现
        返回包含字段定义的字典
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_index_config(cls) -> Dict[str, Any]:
        """
        获取collection的索引配置，子类必须实现
        返回包含索引参数的字典
        """
        pass

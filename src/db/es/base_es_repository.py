import json
import logging
from typing import TypeVar, Generic, Any, Type, cast

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Document, Search
from elasticsearch_dsl.connections import connections

from src.db.es.base_doc import BaseDoc
from src.configs.config import services

logger = logging.getLogger(__name__)

ModelType = TypeVar('ModelType', bound=BaseDoc)


# 初始化Elasticsearch连接
def _initialize_es_connection():
    """
    初始化Elasticsearch连接，从配置文件读取连接参数
    """
    if services.es is None:
        raise ValueError("Elasticsearch 配置未找到，请检查 config.yaml 中的 es 配置")
    
    connection_params = services.es.get_connection_params()
    logger.info(f"Creating Elasticsearch connection with params: {connection_params}")
    
    connections.create_connection(**connection_params)

# 初始化连接
_initialize_es_connection()


class BaseEsRepository(Generic[ModelType]):

    @classmethod
    def get_model_class(cls) -> Type[ModelType]:
        """
        获取模型类，需要子类重写
        """
        raise NotImplementedError("子类必须实现 get_model_class 方法")

    @classmethod
    def save(cls, document: Document):
        doc_class = cls.get_model_class()
        doc_dict = document.to_dict()
        doc = doc_class(meta={'id': document.id}, **doc_dict)
        doc.save()

    @classmethod
    def update(cls, updated_doc: Document):
        doc_class = cls.get_model_class()
        doc = doc_class.get(id=updated_doc.id)
        if doc:
            for key, value in updated_doc.to_dict().items():
                setattr(doc, key, value)
            doc.save()

    @classmethod
    def delete_by_id(cls, doc_id: int):
        doc_class = cls.get_model_class()
        doc = doc_class.get(id=doc_id)
        if doc:
            doc.delete()

    @classmethod
    def delete_by_ids(cls, doc_ids: list[int]) -> bool:
        """
        批量删除文档

        Args:
            doc_ids: 主键ID列表

        Returns:
            是否删除成功
        """
        doc_class = cls.get_model_class()
        success_count = 0
        
        for doc_id in doc_ids:
            try:
                doc = doc_class.get(id=doc_id)
                if doc:
                    doc.delete()
                    success_count += 1
                else:
                    logger.warning(f"文档 ID {doc_id} 不存在")
            except Exception as e:
                logger.error(f"删除文档 ID {doc_id} 失败: {str(e)}")
                
        # 如果至少删除了一个文档，就认为成功
        return success_count > 0

    @classmethod
    def match_search(cls, index_name: str, field: str, content: str, size: int = 10):
        try:
            # 获取Elasticsearch客户端，使用类型转换确保类型安全
            es = cast(Elasticsearch, connections.get_connection())
            
            # 刷新索引
            es.indices.refresh(index=index_name)

            # 构建搜索查询
            s = Search(index=index_name).query("match", **{field: content})

            # 设置返回结果数量
            s = s[:size]

            # 打印生成的 DSL 查询语句
            logger.info(f"索引 {index_name} 的 DSL 查询语句: {json.dumps(s.to_dict(), ensure_ascii=False)}")
            
            # 执行搜索
            response = s.execute()
            return response
            
        except Exception as e:
            logger.error(f"执行搜索失败，索引: {index_name}, 错误: {str(e)}")
            return None

    @classmethod
    def match_search_with_filters(cls, index_name: str, match_field: str, content: str, filters: list[tuple[str, Any]],
                                  size: int = 10):
        try:
            # 获取Elasticsearch客户端，使用类型转换确保类型安全
            es = cast(Elasticsearch, connections.get_connection())
            
            # 刷新索引
            es.indices.refresh(index=index_name)

            # 构建基础搜索查询
            s = Search(index=index_name).query("match", **{match_field: content})

            # 设置返回结果数量
            s = s[:size]

            # 添加过滤条件
            for filter_field, filter_value in filters:
                s = s.filter("term", **{filter_field: filter_value})

            # 打印生成的 DSL 查询语句
            logger.info(f"索引 {index_name} 的 DSL 查询语句: {json.dumps(s.to_dict(), ensure_ascii=False)}")
            
            # 执行搜索
            response = s.execute()
            return response
            
        except Exception as e:
            logger.error(f"执行带过滤条件的搜索失败，索引: {index_name}, 错误: {str(e)}")
            return None

    @classmethod
    def bool_search(cls, index_name: str, must_conditions: list[dict] | None = None, 
                    filter_conditions: list[dict] | None = None, should_conditions: list[dict] | None = None,
                    size: int = 10):
        """
        通用bool查询方法

        Args:
            index_name: 索引名称
            must_conditions: must条件列表，每个元素是字典形式的查询条件
            filter_conditions: filter条件列表，每个元素是字典形式的查询条件
            should_conditions: should条件列表，每个元素是字典形式的查询条件
            size: 返回结果数量

        Returns:
            搜索响应结果
        """
        try:
            # 获取Elasticsearch客户端，使用类型转换确保类型安全
            es = cast(Elasticsearch, connections.get_connection())
            
            # 刷新索引
            es.indices.refresh(index=index_name)

            # 构建bool查询
            bool_query = {}
            if must_conditions:
                bool_query["must"] = must_conditions
            if filter_conditions:
                bool_query["filter"] = filter_conditions
            if should_conditions:
                bool_query["should"] = should_conditions
            
            # 构建搜索查询
            s = Search(index=index_name).query("bool", **bool_query)

            # 设置返回结果数量
            s = s[:size]

            # 打印生成的 DSL 查询语句
            logger.info(f"索引 {index_name} 的 DSL 查询语句: {json.dumps(s.to_dict(), ensure_ascii=False)}")
            
            # 执行搜索
            response = s.execute()
            return response
            
        except Exception as e:
            logger.error(f"执行bool查询失败，索引: {index_name}, 错误: {str(e)}")
            return None

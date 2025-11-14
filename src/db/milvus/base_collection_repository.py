
import logging
from typing import TypeVar, Generic, Type, Dict, Any, Union

from pymilvus import MilvusClient

from src.db.milvus.base_collection import BaseCollection
from src.configs.config import services

logger = logging.getLogger(__name__)

ModelType = TypeVar('ModelType', bound=BaseCollection)


class BaseCollectionRepository(Generic[ModelType]):
    """
    Milvus Collection Repository 基类
    提供对Milvus collection的增删改查、向量搜索等操作的封装
    """

    # Milvus 客户端实例（类级别共享）
    _client: MilvusClient | None = None

    @classmethod
    def get_model_class(cls) -> Type[ModelType]:
        """
        获取模型类，需要子类重写
        """
        raise NotImplementedError("子类必须实现 get_model_class 方法")

    @classmethod
    def get_client(cls) -> MilvusClient:
        """
        获取Milvus客户端，如果不存在则创建
        """
        if cls._client is None:
            # 从配置文件读取连接参数
            if services.milvus is None:
                raise ValueError("Milvus 配置未找到，请检查 config.yaml 中的 milvus 配置")
            
            connection_params = services.milvus.get_connection_params()
            cls._client = MilvusClient(**connection_params)
        return cls._client

    @classmethod
    def set_client(cls, client: MilvusClient):
        """
        设置Milvus客户端实例
        """
        cls._client = client

    @classmethod
    def _ensure_database_context(cls) -> MilvusClient:
        """
        确保客户端使用正确的数据库上下文
        
        Returns:
            配置好数据库上下文的Milvus客户端
        """
        client = cls.get_client()
        model_class = cls.get_model_class()
        database_name = model_class.get_database_name()
        
        if database_name:
            client.using_database(database_name)
        
        return client

    @classmethod
    def _get_collection_params(cls, **kwargs) -> Dict[str, Any]:
        """
        获取包含数据库名称的collection参数
        
        Args:
            **kwargs: 其他参数
            
        Returns:
            包含collection_name和database_name的参数字典
        """
        model_class = cls.get_model_class()
        collection_name = model_class.get_collection_name()
        database_name = model_class.get_database_name()
        
        params = {
            "collection_name": collection_name,
            **kwargs
        }
        if database_name:
            params["database_name"] = database_name
            
        return params

    @classmethod
    def ensure_collection_exists(cls) -> bool:
        """
        确保collection存在，如果不存在则创建
        
        Returns:
            是否成功确保collection存在
        """
        try:
            client = cls._ensure_database_context()
            model_class = cls.get_model_class()
            collection_name = model_class.get_collection_name()
            database_name = model_class.get_database_name()

            # 检查collection是否存在
            has_collection_params = {"collection_name": collection_name}
            if database_name:
                has_collection_params["database_name"] = database_name
            if client.has_collection(**has_collection_params):
                return True

            # 创建schema
            schema_config = model_class.get_schema_config()
            
            # 分离字段配置和schema配置
            fields_config = schema_config.pop('fields', [])
            schema = client.create_schema(**schema_config)

            # 添加字段
            for field_config in fields_config:
                schema.add_field(**field_config)

            # 创建collection，使用数据库名称
            create_params = {
                "collection_name": collection_name,
                "schema": schema
            }
            if database_name:
                create_params["database_name"] = database_name
                
            client.create_collection(**create_params)

            # 创建索引
            index_config = model_class.get_index_config()
            if index_config:
                index_params = client.prepare_index_params()
                for index_param in index_config.get('indexes', []):
                    index_params.add_index(**index_param)

                create_index_params = {"collection_name": collection_name, "index_params": index_params}
                if database_name:
                    create_index_params["database_name"] = database_name
                client.create_index(**create_index_params)

            # 加载collection
            load_params = {"collection_name": collection_name}
            if database_name:
                load_params["database_name"] = database_name
            client.load_collection(**load_params)

            logger.info(f"Successfully created and loaded collection: {collection_name} in database: {database_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {str(e)}")
            return False

    @classmethod
    def insert(cls, data: Union[Dict[str, Any], list[Dict[str, Any]]], **kwargs) -> Dict[str, Any]:
        """
        插入数据到collection
        
        Args:
            data: 要插入的数据，可以是单个字典或字典列表
            **kwargs: 其他参数，如 partition_name
            
        Returns:
            插入结果信息
        """
        try:
            client = cls._ensure_database_context()

            # 确保collection存在
            cls.ensure_collection_exists()

            # 如果是单个字典，转换为列表
            if isinstance(data, dict):
                data = [data]

            params = cls._get_collection_params(data=data, **kwargs)
            result = client.insert(**params)

            logger.info(f"Inserted {result.get('insert_count', 0)} records to {params['collection_name']}")
            return result

        except Exception as e:
            logger.error(f"Failed to insert data: {str(e)}")
            raise

    @classmethod
    def upsert(cls, data: Union[Dict[str, Any], list[Dict[str, Any]]], **kwargs) -> Dict[str, Any]:
        """
        更新插入数据到collection
        
        Args:
            data: 要更新插入的数据，可以是单个字典或字典列表
            **kwargs: 其他参数，如 partition_name
            
        Returns:
            更新插入结果信息
        """
        try:
            client = cls._ensure_database_context()

            # 确保collection存在
            cls.ensure_collection_exists()

            # 如果是单个字典，转换为列表
            if isinstance(data, dict):
                data = [data]

            params = cls._get_collection_params(data=data, **kwargs)
            result = client.upsert(**params)

            logger.info(f"Upserted {result.get('upsert_count', 0)} records to {params['collection_name']}")
            return result

        except Exception as e:
            logger.error(f"Failed to upsert data: {str(e)}")
            raise

    @classmethod
    def delete(cls, filter_expr: str, **kwargs) -> Dict[str, Any]:
        """
        根据过滤条件删除数据
        
        Args:
            filter_expr: 过滤表达式，如 "id in [1, 2, 3]"
            **kwargs: 其他参数，如 partition_name
            
        Returns:
            删除结果信息
        """
        try:
            client = cls._ensure_database_context()

            params = cls._get_collection_params(filter=filter_expr, **kwargs)
            result = client.delete(**params)

            logger.info(f"Deleted records from {params['collection_name']} with filter: {filter_expr}")
            return result

        except Exception as e:
            logger.error(f"Failed to delete data: {str(e)}")
            raise

    @classmethod
    def delete_by_ids(cls, ids: list[Any], id_field: str = "id", **kwargs) -> Dict[str, Any]:
        """
        根据ID列表批量删除数据
        
        Args:
            ids: ID列表
            id_field: ID字段名，默认为"id"
            **kwargs: 其他参数，如 partition_name
            
        Returns:
            删除结果信息
        """
        if not ids:
            return {"delete_count": 0}

        # 构建过滤表达式
        if len(ids) == 1:
            filter_expr = f"{id_field} == {repr(ids[0])}"
        else:
            # 格式化ID列表为字符串
            ids_str = [repr(id_val) for id_val in ids]
            filter_expr = f"{id_field} in [{', '.join(ids_str)}]"

        return cls.delete(filter_expr, **kwargs)

    @classmethod
    def query(cls, filter_expr: str = "", output_fields: list[str] | None = None, **kwargs) -> list[Dict[str, Any]]:
        """
        查询数据
        
        Args:
            filter_expr: 过滤表达式，如 "id > 0"
            output_fields: 要返回的字段列表
            **kwargs: 其他参数，如 partition_names, limit, offset
            
        Returns:
            查询结果列表
        """
        try:
            client = cls._ensure_database_context()

            params = cls._get_collection_params(
                filter=filter_expr,
                output_fields=output_fields,
                **kwargs
            )
            result = client.query(**params)

            return result

        except Exception as e:
            logger.error(f"Failed to query data: {str(e)}")
            raise

    @classmethod
    def get_by_id(cls,
                  entity_id: Any,
                  id_field: str = "id",
                  output_fields: list[str] | None = None,
                  **kwargs) -> Dict[str, Any] | None:
        """
        根据ID获取单个实体
        
        Args:
            entity_id: 实体ID
            id_field: ID字段名，默认为"id"
            output_fields: 要返回的字段列表
            **kwargs: 其他参数
            
        Returns:
            实体数据或None
        """
        filter_expr = f"{id_field} == {repr(entity_id)}"
        results = cls.query(filter_expr, output_fields, limit=1, **kwargs)
        return results[0] if results else None

    @classmethod
    def get_by_ids(cls, ids: list[Any],
                   id_field: str = "id",
                   output_fields: list[str] | None = None,
                   **kwargs) -> list[Dict[str, Any]]:
        """
        根据ID列表批量获取实体
        
        Args:
            ids: ID列表
            id_field: ID字段名，默认为"id"
            output_fields: 要返回的字段列表
            **kwargs: 其他参数
            
        Returns:
            实体数据列表
        """
        if not ids:
            return []

        # 构建过滤表达式
        if len(ids) == 1:
            filter_expr = f"{id_field} == {repr(ids[0])}"
        else:
            ids_str = [repr(id_val) for id_val in ids]
            filter_expr = f"{id_field} in [{', '.join(ids_str)}]"

        return cls.query(filter_expr, output_fields, **kwargs)

    @classmethod
    def search(cls,
               data: Union[list[list[float]], list[Dict[str, Any]]],
               anns_field: str | None = "vector",
               limit: int = 10,
               search_params: Dict[str, Any] | None = None,
               filter_expr: str | None = "",
               output_fields: list[str] | None = None,
               **kwargs) -> list[list[Dict[str, Any]]]:
        """
        向量搜索
        
        Args:
            data: 查询向量数据
            anns_field: 向量字段名
            limit: 返回结果数量限制
            search_params: 搜索参数
            filter_expr: 过滤表达式
            output_fields: 要返回的字段列表
            **kwargs: 其他参数，如 partition_names
            
        Returns:
            搜索结果列表
        """
        try:
            client = cls._ensure_database_context()

            # 默认搜索参数
            if search_params is None:
                search_params = {"metric_type": "COSINE", "params": {}}

            params = cls._get_collection_params(
                data=data,
                anns_field=anns_field,
                limit=limit,
                search_params=search_params,
                filter=filter_expr,
                output_fields=output_fields,
                **kwargs
            )
            result = client.search(**params)

            return result

        except Exception as e:
            logger.error(f"Failed to search vectors: {str(e)}")
            raise

    @classmethod
    def hybrid_search(cls,
                      reqs: list[Any],
                      ranker: Any,
                      limit: int = 10,
                      output_fields: list[str] | None = None,
                      **kwargs) -> list[list[Dict[str, Any]]]:
        """
        混合搜索
        
        Args:
            reqs: 搜索请求列表
            ranker: 排序器
            limit: 返回结果数量限制
            output_fields: 要返回的字段列表
            **kwargs: 其他参数
            
        Returns:
            搜索结果列表
        """
        try:
            client = cls._ensure_database_context()

            params = cls._get_collection_params(
                reqs=reqs,
                ranker=ranker,
                limit=limit,
                output_fields=output_fields,
                **kwargs
            )
            result = client.hybrid_search(**params)

            return result

        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {str(e)}")
            raise

    @classmethod
    def count(cls, filter_expr: str = "", **kwargs) -> int:
        """
        统计数据条数
        
        Args:
            filter_expr: 过滤表达式
            **kwargs: 其他参数
            
        Returns:
            数据条数
        """
        try:
            client = cls._ensure_database_context()

            # 使用query获取count
            params = cls._get_collection_params(
                filter=filter_expr,
                output_fields=["count(*)"],
                **kwargs
            )
            result = client.query(**params)

            if result:
                # 解析count结果
                count_str = result[0].get("count(*)", "0")
                return int(count_str)
            return 0

        except Exception as e:
            logger.error(f"Failed to count data: {str(e)}")
            raise

    @classmethod
    def exists(cls, filter_expr: str, **kwargs) -> bool:
        """
        检查是否存在满足条件的数据
        
        Args:
            filter_expr: 过滤表达式
            **kwargs: 其他参数
            
        Returns:
            是否存在数据
        """
        return cls.count(filter_expr, **kwargs) > 0

    @classmethod
    def drop_collection(cls) -> bool:
        """
        删除collection
        
        Returns:
            是否删除成功
        """
        try:
            client = cls._ensure_database_context()
            model_class = cls.get_model_class()
            collection_name = model_class.get_collection_name()
            database_name = model_class.get_database_name()

            has_collection_params = {"collection_name": collection_name}
            if database_name:
                has_collection_params["database_name"] = database_name
                
            if client.has_collection(**has_collection_params):
                drop_params = {"collection_name": collection_name}
                if database_name:
                    drop_params["database_name"] = database_name
                client.drop_collection(**drop_params)
                logger.info(f"Dropped collection: {collection_name} from database: {database_name}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to drop collection: {str(e)}")
            return False

    @classmethod
    def get_collection_stats(cls) -> Dict[str, Any]:
        """
        获取collection统计信息
        
        Returns:
            统计信息字典
        """
        try:
            client = cls._ensure_database_context()
            
            params = cls._get_collection_params()
            return client.get_collection_stats(params['collection_name'])

        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            raise

    @classmethod
    def describe_collection(cls) -> Dict[str, Any]:
        """
        获取collection描述信息
        
        Returns:
            描述信息字典
        """
        try:
            client = cls._ensure_database_context()
            
            params = cls._get_collection_params()
            return client.describe_collection(params['collection_name'])

        except Exception as e:
            logger.error(f"Failed to describe collection: {str(e)}")
            raise

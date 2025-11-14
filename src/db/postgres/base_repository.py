
import logging
from datetime import datetime
from typing import TypeVar, Generic, Type, List

from sonyflake import SonyFlake
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.engine import Result

from src.common.page import PageQuery
from src.common.sys_user import SysUser
from src.db.postgres.base_entity import BaseEntity
from src.common.constants import IS_DELETED_YES, IS_DELETED_NO, TABLE_LOGIC, CREATE_USER, UPDATE_USER, CREATE_TIME, \
    UPDATE_TIME, ID

logger = logging.getLogger(__name__)

ModelType = TypeVar('ModelType', bound=BaseEntity)


class BaseRepository(Generic[ModelType]):
    """通用的 Repository 基类，封装 BaseEntity 的所有通用操作"""

    @classmethod
    def get_model_class(cls) -> Type[ModelType]:
        """
        获取模型类，需要子类重写
        """
        raise NotImplementedError("子类必须实现 get_model_class 方法")

    @classmethod
    async def get_by_id(cls, session: AsyncSession, ident: int):
        """
        通过ID查询实体的类方法版本，可以直接通过子类调用

        Args:
            session: 数据库会话
            ident: 实体ID

        Returns:
            查询到的实体实例（仅返回未删除的记录）
        """
        model_class = cls.get_model_class()
        return await model_class.select_one(session, id=ident, is_deleted=IS_DELETED_NO)

    @classmethod
    async def get_one(cls, session: AsyncSession, **kwargs):
        """
        通过条件查询实体的类方法版本，可以直接通过子类调用

        Args:
            session: 数据库会话
            kwargs: 查询条件

        Returns:
            查询到的实体实例（仅返回未删除的记录）
        """
        model_class = cls.get_model_class()
        kwargs[TABLE_LOGIC] = IS_DELETED_NO
        return await model_class.select_one(session, **kwargs)

    @classmethod
    async def list(cls, session: AsyncSession, **kwargs) -> List[ModelType]:
        """
        通过条件查询实体的类方法版本，可以直接通过子类调用

        Args:
            session: 数据库会话
            kwargs: 查询条件

        Returns:
            查询到的实体实例列表（仅返回未删除的记录）
        """
        model_class = cls.get_model_class()
        kwargs[TABLE_LOGIC] = IS_DELETED_NO
        return await model_class.select_list(session, **kwargs)

    @classmethod
    async def list_by_ids(cls, session: AsyncSession, ids: List[int]):
        """
        通过ID列表查询实体的类方法版本，可以直接通过子类调用

        Args:
            session: 数据库会话
            ids: 实体ID列表

        Returns:
            查询到的实体实例列表（仅返回未删除的记录）
        """
        model_class = cls.get_model_class()
        return await model_class.select_batch_ids(session, ids)

    @classmethod
    async def page(cls, session: AsyncSession, page_query: PageQuery, **kwargs):
        """
        分页查询实体列表的类方法版本，可以直接通过子类调用

        这是一个通用的分页查询方法，会自动过滤已删除的记录。
        如果需要自定义查询逻辑（如搜索、排序等），可以在子类中重写此方法。

        Args:
            session: 数据库会话
            page_query: 分页查询参数，包含：
                - current: 当前页码，从1开始
                - size: 每页记录数
                - keyword: 搜索关键词（可选，本方法不处理，子类可以重写处理）
                - orders: 排序方式（可选，本方法不处理，子类可以重写处理）
            kwargs: 额外的查询条件，如 name="张三", status=1 等

        Returns:
            instances, total_count: 查询到的实体实例列表和总记录数
        """
        model_class = cls.get_model_class()

        # 计算offset
        offset = (page_query.current - 1) * page_query.size

        # 添加逻辑删除过滤条件
        kwargs[TABLE_LOGIC] = IS_DELETED_NO

        # 调用BaseEntity的select_page方法
        return await model_class.select_page(
            session=session,
            offset=offset,
            limit=page_query.size,
            **kwargs
        )

    @classmethod
    async def save(cls, session: AsyncSession, user: SysUser, **kwargs):
        """
        创建实体的类方法版本，可以直接通过子类调用

        Args:
            session: 数据库会话
            user: 操作用户信息
            kwargs: 实体属性

        Returns:
            创建的实体实例
        """
        model_class = cls.get_model_class()
        now = datetime.now()
        # 添加创建用户信息
        if not kwargs.get(ID):
            kwargs[ID] = SonyFlake().next_id()
        kwargs[CREATE_USER] = user.id
        kwargs[CREATE_TIME] = now
        kwargs[UPDATE_USER] = user.id
        kwargs[UPDATE_TIME] = now
        kwargs[TABLE_LOGIC] = IS_DELETED_NO
        return await model_class.insert(session, **kwargs)

    @classmethod
    async def save_batch(cls, session: AsyncSession, user: SysUser, entity_list: List[ModelType]):
        """
        批量创建实体的类方法版本，可以直接通过子类调用

        Args:
            session: 数据库会话
            user: 操作用户信息
            entity_list: 实体列表

        Returns:
            创建的实体实例列表
        """
        # 输入验证
        if not entity_list:
            return []

        if not user:
            raise ValueError("用户信息不能为空")

        try:
            model_class = cls.get_model_class()
            now = datetime.now()
            created_entities = []

            # 为每个实体设置通用字段
            for entity_data in entity_list:
                # 如果实体是字典类型，转换为模型实例
                if isinstance(entity_data, dict):
                    # 生成ID（如果没有提供）
                    if not entity_data.get(ID):
                        entity_data[ID] = SonyFlake().next_id()

                    # 设置通用字段
                    entity_data[CREATE_USER] = user.id
                    entity_data[CREATE_TIME] = now
                    entity_data[UPDATE_USER] = user.id
                    entity_data[UPDATE_TIME] = now
                    entity_data[TABLE_LOGIC] = IS_DELETED_NO

                    # 创建模型实例
                    instance = model_class(**entity_data)
                else:
                    # 如果已经是模型实例，设置字段
                    if not hasattr(entity_data, ID) or not getattr(entity_data, ID):
                        setattr(entity_data, ID, SonyFlake().next_id())

                    setattr(entity_data, CREATE_USER, user.id)
                    setattr(entity_data, CREATE_TIME, now)
                    setattr(entity_data, UPDATE_USER, user.id)
                    setattr(entity_data, UPDATE_TIME, now)
                    setattr(entity_data, TABLE_LOGIC, IS_DELETED_NO)

                    instance = entity_data

                # 添加到会话
                session.add(instance)
                created_entities.append(instance)

            # 刷新会话以将实例持久化到数据库（但不提交事务）
            await session.flush()

            # 刷新所有实例以获取数据库生成的值
            for instance in created_entities:
                await session.refresh(instance)

            return created_entities

        except Exception as e:
            logger.error(f"Error batch saving {cls.__name__}: {e}")
            # 回滚事务
            await session.rollback()
            # 重新抛出异常，让上层处理
            raise

    @classmethod
    async def update_by_id(cls, session: AsyncSession, user: SysUser, ident: int | None = None, **kwargs):
        """
        更新实体的类方法版本，可以直接通过子类调用

        Args:
            session: 数据库会话
            user: 操作用户信息
            ident: 实体ID
            kwargs: 要更新的属性

        Returns:
            更新后的实体实例
        """
        model_class = cls.get_model_class()

        # 如果没有提供ident，尝试从kwargs中获取id
        if ident is None:
            ident = kwargs.pop('id', None)
            if ident is None:
                raise ValueError("必须提供实体ID，可以通过ident参数或kwargs中的id字段提供")

        # 添加更新用户信息
        kwargs[UPDATE_USER] = user.id
        kwargs[UPDATE_TIME] = datetime.now()
        return await model_class.update_by_id(session, ident, **kwargs)

    @classmethod
    async def remove_by_id(cls, session: AsyncSession, user: SysUser, ident: int) -> bool:
        """
        逻辑删除实体的类方法版本，可以直接通过子类调用
        
        Args:
            session: 数据库会话
            user: 操作用户信息
            ident: 实体ID

        Returns:
            是否删除成功
        """
        # 先检查实体是否存在
        entity = await cls.get_by_id(session, ident)
        if entity:
            await cls.update_by_id(session, user, ident, is_deleted=IS_DELETED_YES)
            return True
        return False

    @classmethod
    async def remove_by_ids(cls, session: AsyncSession, user: SysUser, ids: List[int]) -> bool:
        """
        批量逻辑删除

        Args:
            session: 数据库会话
            user: 操作用户信息
            ids: 实体ID列表
        Returns:
            是否删除成功（True: 至少有一条记录被删除，False: 没有记录被删除）
        Raises:
            ValueError: 当ids为空或update_user_id无效时
            SQLAlchemyError: 数据库操作异常
        """
        # 输入验证
        if not ids:
            return False
        
        if not user:
            raise ValueError("update_user_id must be a positive integer")
        
        # 去重并过滤无效ID
        unique_ids = [id for id in set(ids) if id > 0]
        if not unique_ids:
            return False
        
        try:
            model_class = cls.get_model_class()
            
            # 批量更新 is_deleted 字段为 1，同时更新 update_user
            stmt = update(model_class).where(
                model_class.id.in_(unique_ids), 
                model_class.is_deleted == IS_DELETED_NO
            ).values(
                is_deleted=IS_DELETED_YES, 
                update_user=user.id
            )

            result: Result = await session.execute(stmt)

            # 返回是否有记录被更新
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting {cls.__name__} with IDs {ids}: {e}")
            # 回滚事务
            await session.rollback()
            # 重新抛出异常，让上层处理
            raise

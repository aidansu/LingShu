
from typing import Self, List

import typing as t
from datetime import datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy import and_, select, BIGINT, Integer, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func

from src.common.constants import IS_DELETED_NO


class SqlConfig(BaseModel):
    url: str
    connect_args: t.Any = None
    pool_size: int = 1000
    max_overflow: int = 500
    pool_timeout: int = 30
    pool_recycle: int = 1800


class Base(DeclarativeBase):
    pass


class BaseEntity(Base):
    __abstract__ = True
    
    # 添加Pydantic配置，允许从SQLAlchemy模型创建
    model_config = ConfigDict(from_attributes=True)

    id: Mapped[int] = mapped_column(BIGINT, comment='主键ID', primary_key=True, index=True, autoincrement=True)
    create_user: Mapped[int] = mapped_column(BIGINT, comment='创建人', nullable=False)
    create_time: Mapped[datetime] = mapped_column(comment='创建时间', server_default=func.now())
    update_user: Mapped[int] = mapped_column(BIGINT, comment='更新人', nullable=True)
    update_time: Mapped[datetime] = mapped_column(comment='更新时间', server_default=func.now(), onupdate=func.now())
    create_dept: Mapped[int] = mapped_column(BIGINT, comment='创建部门', nullable=False)
    status: Mapped[int] = mapped_column(Integer, comment='状态', nullable=False)
    is_deleted: Mapped[bool] = mapped_column(Integer, comment='是否删除', default=IS_DELETED_NO)

    @classmethod
    async def insert(cls, session: AsyncSession, **kwargs) -> Self:
        """
        插入一条数据

        Args:
            session: 数据库会话
            kwargs: 插入的数据

        Returns:
            插入后的实例
        """
        instance = cls(**kwargs)
        session.add(instance)
        await session.flush()
        await session.refresh(instance)
        return instance

    @classmethod
    async def delete_by_id(cls, session: AsyncSession, ident: int) -> Self | None:
        """
        通过ID删除一条数据

        Args:
            session: 数据库会话
            ident: 删除的ID

        Returns:
            删除的实例
        """
        sql = delete(cls).where(cls.id == ident)
        result = await session.execute(sql)
        return result.rowcount

    @classmethod
    async def delete_batch_ids(cls, session: AsyncSession, ids: List[int]):
        """
        通过ID逻辑删除一条数据

        Args:
            session: 数据库会话
            ids: 删除的ID列表

        Returns:
            删除的实例
        """
        stmt = delete(cls).where(cls.id.in_(ids))
        result = await session.execute(stmt)
        return result.rowcount

    @classmethod
    async def update_by_id(cls, session: AsyncSession, ident: int, **kwargs):
        """
        通过ID更新实例
        
        Args:
            session: 数据库会话
            ident: 实例ID
            kwargs: 要更新的属性
            
        Returns:
            更新后的实例
        """
        instance = await session.get(cls, ident)
        if not instance:
            raise ValueError(f"实例 ID {ident} 不存在")
            
        for attr, value in kwargs.items():
            setattr(instance, attr, value)
        await session.flush()
        await session.refresh(instance)
        return instance

    @classmethod
    async def select_by_id(cls, session: AsyncSession, ident: int) -> Self | None:
        """
        通过ID查询单个实例
        
        Args:
            session: 数据库会话
            ident: 对象ID
            
        Returns:
            查询到的实例，如果不存在则返回None
        """
        return await session.get(cls, ident)

    @classmethod
    async def select_batch_ids(cls, session: AsyncSession, ids: List[int]):
        """
        批量查询指定ID的实例

        Args:
            session: 数据库会话
            ids: 实例ID列表

        Returns:
            查询到的实例列表
        """
        query = select(cls).where(cls.id.in_(ids), cls.is_deleted == IS_DELETED_NO)
        result = await session.execute(query)
        return result.scalars().all()

    @classmethod
    async def select_one(cls, session: AsyncSession, **kwargs):
        """
        查询单个实例

        Args:
            session: 数据库会话
            kwargs: 查询条件

        Returns:
            查询到的实例，如果存在多个则返回第一个，如果不存在则返回None
        """
        query_conditions = []
        for field, value in kwargs.items():
            query_conditions.append(getattr(cls, field) == value)

        query = select(cls).where(and_(*query_conditions))
        result = await session.execute(query)

        instance = result.scalars().first()
        return instance

    @classmethod
    async def select_count(cls, session: AsyncSession, **kwargs):
        """
        查询符合条件的实例数量

        Args:
            session: 数据库会话
            kwargs: 查询条件

        Returns:
            符合条件的实例数量
        """
        query_conditions = []
        for field, value in kwargs.items():
            query_conditions.append(getattr(cls, field) == value)

        result = await session.execute(select(func.count(cls.id)).where(and_(*query_conditions)))
        return result.scalars().all()

    @classmethod
    async def select_list(cls, session: AsyncSession, **kwargs):
        """
        查询列表

        :param session: 数据库会话
        :param kwargs: 查询条件
        :return:
        """
        query_conditions = []
        for field, value in kwargs.items():
            query_conditions.append(getattr(cls, field) == value)

        result = await session.execute(
            select(cls).where(and_(*query_conditions))
        )
        return result.scalars().all()

    @classmethod
    async def select_page(cls, session: AsyncSession, offset: int = 0, limit: int = 100, **kwargs):
        """
        分页查询数据

        Args:
            session: 数据库会话
            offset: 偏移量
            limit: 限制数量
            kwargs: 查询条件

        Returns:
            tuple: (查询结果列表, 总记录数)
        """
        # 构建基础查询，默认过滤已删除的记录
        base_query = select(cls).where(cls.is_deleted == IS_DELETED_NO)
        
        # 添加查询条件
        for attr, value in kwargs.items():
            if hasattr(cls, attr):
                base_query = base_query.where(getattr(cls, attr) == value)
        
        # 构建数据查询（添加分页）
        data_query = base_query.offset(offset).limit(limit)
        
        # 构建计数查询（使用子查询提高性能）
        count_query = select(func.count()).select_from(base_query.subquery())
        
        # 并行执行查询总数和分页数据
        count_result = await session.execute(count_query)
        data_result = await session.execute(data_query)
        
        # 获取结果
        total_count = count_result.scalar_one()
        instances = data_result.scalars().all()
        
        return instances, total_count

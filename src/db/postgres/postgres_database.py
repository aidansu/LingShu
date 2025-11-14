
import typing as t
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from src.configs import config

_AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None


def get_async_session_local() -> async_sessionmaker[AsyncSession]:
    """获取 AsyncSessionLocal，延迟创建"""
    global _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        # 通过代理访问 postgres，会自动触发延迟加载
        postgres_engine = config.services.postgres
        _AsyncSessionLocal = async_sessionmaker(
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            bind=postgres_engine,
            expire_on_commit=False
        )
    return _AsyncSessionLocal


# 为了向后兼容，提供一个属性访问器
class _AsyncSessionLocalProxy:
    """AsyncSessionLocal 代理类，提供延迟加载"""
    
    def __call__(self, *args, **kwargs):
        return get_async_session_local()(*args, **kwargs)
    
    def __getattr__(self, name: str):
        return getattr(get_async_session_local(), name)


AsyncSessionLocal = _AsyncSessionLocalProxy()

Base = declarative_base()


async def get_async_db() -> t.AsyncGenerator[AsyncSession, None]:
    """
    数据库会话依赖，自动处理事务提交和回滚
    
    正常完成时自动提交，异常时自动回滚
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()  # 如果 handler 正常完成，则提交
    except Exception:
        await session.rollback()  # 异常时回滚
        raise
    finally:
        await session.close()

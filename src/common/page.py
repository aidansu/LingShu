from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar('T')


class PageQuery(BaseModel):
    """
    分页查询参数

    Attributes:
        keyword: 搜索关键词
        current: 当前页码，默认为1
        size: 每页大小，默认为10，最大为100
        orders: 排序方式
    """
    keyword: str | None = None
    current: int = 1
    size: int = 10
    orders: list[str] | None = None


class Page(BaseModel, Generic[T]):
    """
    分页列表
    """
    current: int = 1
    size: int = 10
    total: int = 0
    pages: int = 0
    records: list[T] = []

    def __init__(self, current: int = 1, size: int = 10, total: int = 0, records=None):
        if records is None:
            records = []
        super().__init__(current=current, size=size, total=total, records=records)
        self.current = current
        self.size = size

    def setRecords(self, records, total):
        self.records = records
        self.total = total
        if self.size == 0:
            self.pages = 0
        else:
            self.pages = self.total // self.size
            if self.total % self.size != 0:
                self.pages += 1

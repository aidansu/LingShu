
from fastapi import HTTPException

from src.common.result_code import ResultCode


# 业务异常类
class BizException(HTTPException):
    def __init__(self, detail: str, code: int | None = None):
        if code is None:
            code = ResultCode.FAILURE.getCode()
        super().__init__(status_code=code, detail=detail)

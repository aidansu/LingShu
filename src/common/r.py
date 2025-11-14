
from http import HTTPStatus
from typing import Any

from fastapi import status
from pydantic import BaseModel

from src.common.biz_exception import BizException
from src.common.constants import DEFAULT_SUCCESS_MESSAGE, DEFAULT_NULL_MESSAGE, DEFAULT_FAILURE_MESSAGE
from src.common.result_code import ResultCode


class R(BaseModel):
    """
    公共响应类
    """
    status: int = status.HTTP_200_OK
    data: Any | None = None
    msg: str | None = ''

    @staticmethod
    def datas(data: Any, msg: str | None = DEFAULT_SUCCESS_MESSAGE):
        if data is None:
            msg = DEFAULT_NULL_MESSAGE
        return R(status=HTTPStatus.OK.value, data=data, msg=msg)

    @staticmethod
    def success(msg: str | None = DEFAULT_SUCCESS_MESSAGE):
        if msg is None:
            msg = DEFAULT_SUCCESS_MESSAGE
        return R(status=HTTPStatus.OK.value, msg=msg)

    @staticmethod
    def fail(msg: str | None = DEFAULT_FAILURE_MESSAGE):
        return R(status=HTTPStatus.BAD_REQUEST.value, msg=msg)

    @staticmethod
    def error(status: int | None = HTTPStatus.BAD_REQUEST.value, msg: str | None = DEFAULT_FAILURE_MESSAGE):
        return R(status=status, msg=msg)

    @staticmethod
    def failCode(result_code: ResultCode):
        return R(status=result_code.getCode(), msg=result_code.getMsg())

    @staticmethod
    def failBizException(exception: BizException):
        return R(status=exception.status_code, msg=exception.detail)

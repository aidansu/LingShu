
from http import HTTPStatus

from enum import Enum


class ResultCode(Enum):
    SUCCESS = (HTTPStatus.OK.value, "操作成功")
    FAILURE = (HTTPStatus.BAD_REQUEST.value, "业务异常")
    UN_AUTHORIZED = (HTTPStatus.UNAUTHORIZED.value, "请求未授权")
    FORBIDDEN = (HTTPStatus.FORBIDDEN.value, "暂无权限访问")
    TOKEN_ILLEGAL = (HTTPStatus.UNAUTHORIZED, "Token 不合法")
    REFRESH_TOKEN_ILLEGAL = (HTTPStatus.UNAUTHORIZED, "刷新 Token 不合法")
    TOKEN_EXPIRED = (HTTPStatus.UNAUTHORIZED, "Token 已过期")
    NOT_FOUND = (HTTPStatus.NOT_FOUND, "404 没找到请求")
    INTERNAL_SERVER_ERROR = (HTTPStatus.INTERNAL_SERVER_ERROR.value, "服务器异常")
    PARAM_MISS = (HTTPStatus.BAD_REQUEST.value, "缺少必要的请求参数")
    PARAM_TYPE_ERROR = (HTTPStatus.BAD_REQUEST.value, "请求参数类型错误")
    PARAM_BIND_ERROR = (HTTPStatus.BAD_REQUEST.value, "请求参数绑定错误")
    PARAM_VALID_ERROR = (HTTPStatus.BAD_REQUEST.value, "参数校验失败")
    FILE_TYPE_NOT_SUPPORT = (HTTPStatus.BAD_REQUEST.value, "文件类型不支持")
    FILE_SUFFIX_NOT_SUPPORT = (HTTPStatus.BAD_REQUEST.value, "文件后缀不支持")
    FILE_SIZE_LIMIT = (HTTPStatus.BAD_REQUEST.value, "文件大小超过限制")
    FILE_NOT_FOUND = (HTTPStatus.NOT_FOUND.value, "文件未找到")
    FILE_NO_PERMISSION = (HTTPStatus.FORBIDDEN.value, "文件暂无权限查看")

    def __init__(self, code, msg):
        self.code = code
        self.msg = msg

    def getMsg(self):
        return self.msg

    def getCode(self):
        return self.code

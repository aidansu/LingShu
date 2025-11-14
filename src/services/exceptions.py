from fastapi.exceptions import (
    HTTPException,
    RequestValidationError,
    ResponseValidationError,
)
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm.exc import NoResultFound

from src.common.r import R


class SettingNotFound(Exception):
    pass


def no_result_found_handle(req: Request, exc: NoResultFound) -> JSONResponse:
    content = R.error(404, f"Object has not found, exc: {exc}, query_params: {req.query_params}").model_dump()
    return JSONResponse(content=content, status_code=404)


def http_exc_handle(_: Request, exc: HTTPException) -> JSONResponse:
    content = R.error(exc.status_code, exc.detail).model_dump()
    return JSONResponse(content=content, status_code=exc.status_code)


def request_validation_handle(_: Request, exc: RequestValidationError) -> JSONResponse:
    content = R.error(422, f"RequestValidationError, {exc}").model_dump()
    return JSONResponse(content=content, status_code=422)


def response_validation_handle(_: Request, exc: ResponseValidationError) -> JSONResponse:
    content = R.error(500, f"ResponseValidationError, {exc}").model_dump()
    return JSONResponse(content=content, status_code=500)


def response_exception_handle(_: Request, exc: Exception) -> JSONResponse:
    content = R.error(500, f"Internal Server Error").model_dump()
    return JSONResponse(content=content, status_code=500)

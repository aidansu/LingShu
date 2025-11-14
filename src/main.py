from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from src.configs import config
from src.modules.llm_api.llm_api import llm_api
from src.services.exceptions import (
    NoResultFound,
    no_result_found_handle,
    HTTPException,
    http_exc_handle,
    RequestValidationError,
    request_validation_handle,
    ResponseValidationError,
    response_validation_handle, response_exception_handle,
)


def register_exceptions(fastapi_app: FastAPI):
    fastapi_app.add_exception_handler(NoResultFound, no_result_found_handle)
    fastapi_app.add_exception_handler(HTTPException, http_exc_handle)
    fastapi_app.add_exception_handler(RequestValidationError, request_validation_handle)
    fastapi_app.add_exception_handler(ResponseValidationError, response_validation_handle)
    fastapi_app.add_exception_handler(Exception, response_exception_handle)


app = FastAPI(
    title="灵枢",
    description="AI大模型应用开发框架",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    middleware=config.services.middleware
)


register_exceptions(app)
app.include_router(llm_api, tags=["大模型接口"])


import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from src.configs.config import LLM_CONFIG
from src.llm.llm_factory import init_chat_llm

logger = logging.getLogger(__name__)
app = FastAPI()


class ChatReq(BaseModel):
    model: str
    message: str


class ChatResp(BaseModel):
    type: str
    title: str | None = None
    content: str | None = None
    msg: str | None = None


@app.get("/")
async def root():
    return {"message": "Hello LingShu"}


@app.get("/models", response_class=JSONResponse)
async def list_models():
    """获取支持的模型列表"""
    try:
        models_list = []
        # 遍历所有模型提供商
        for provider_name, provider_config in LLM_CONFIG["model_providers"].items():
            # 遍历该提供商下的所有模型配置
            for model_id, model_config in provider_config.get("models", {}).items():
                for model_name in model_config["model_name"]:
                    models_list.append({
                        "provider": provider_name,
                        "model_name": model_name
                    })
        return {"models": models_list}
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"配置文件格式错误，缺失关键字段：{str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取模型列表失败：{str(e)}"
        )


@app.post("/chat-invoke")
async def chat_invoke(request: ChatReq):
    try:
        # 初始化模型
        llm = init_chat_llm(request.model).with_config(temperature=0.6)
        # 调用模型
        response = await llm.ainvoke(request.message)
        # 返回结果
        response_obj = ChatResp(type="text", msg=response.content)
        return response_obj.model_dump(exclude_none=True)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.post("/chat-stream")
async def chat_stream(request: ChatReq):
    try:
        # 初始化模型
        llm = init_chat_llm(request.model)

        # 创建异步生成器来流式响应
        async def generate():
            async for chunk in llm.astream(request.message):
                response_obj = ChatResp(type="text", msg=chunk.content)
                # 假设返回文本数据，按需调整格式
                yield f"data: {response_obj.model_dump_json(exclude_none=True)}\n\n"
            # 添加结束标记
            yield f"data: [DONE]\n\n"

        # 使用StreamingResponse流式传输数据
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

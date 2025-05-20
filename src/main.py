from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from pydantic import BaseModel

from src.configs.config import LLM_CONFIG
from src.llm.llm_factory import init_chat_llm

app = FastAPI()


class ChatRequest(BaseModel):
    model: str
    message: str


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


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 初始化模型
        llm = init_chat_llm(request.model).with_config(temperature=0.6)

        # 调用模型
        response = await llm.ainvoke(request.message)

        return {
            "model": request.model,
            "response": response.content
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


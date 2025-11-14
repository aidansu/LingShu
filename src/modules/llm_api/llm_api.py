import logging
from typing import List, Dict, Any

from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.configs.config import services

from pydantic import BaseModel

from src.llm.llm_factory import init_chat_llm, init_embedding_model

logger = logging.getLogger(__name__)
llm_api = APIRouter()

# 模型类型常量
MODEL_TYPE_CHAT = "chat"
MODEL_TYPE_EMBEDDING = "embedding"
MODEL_TYPE_RERANK = "rerank"
VALID_MODEL_TYPES = {MODEL_TYPE_CHAT, MODEL_TYPE_EMBEDDING, MODEL_TYPE_RERANK}

class ChatReq(BaseModel):
    model: str | None = None
    messages: str
    stream: bool = False
    thinking: bool = False
    temperature: float = 0.6


class ChatResp(BaseModel):
    type: str
    title: str | None = None
    content: str | None = None
    msg: str | None = None
    iconType: int | None = None
    status: int | None = None


class EmbReq(BaseModel):
    input: List[str]
    model: str | None = None


def _create_model_dict(model_id: str, provider_name: str, model_type: str) -> Dict[str, Any]:
    """创建标准化的模型字典
    
    Args:
        model_id: 模型ID/名称
        provider_name: 提供商名称
        model_type: 模型类型（chat/embedding/rerank）
    
    Returns:
        标准化的模型字典
    """
    return {
        "id": model_id,
        "object": "model",
        "owned_by": provider_name,
        "type": model_type
    }


def _get_models_from_config(models_config: Dict[str, Any], model_type: str, thinking: bool = False):
    """从配置中获取模型列表（统一处理 chat、embedding、rerank）
    
    Args:
        models_config: 模型配置字典
        model_type: 模型类型（chat/embedding/rerank）
        thinking: 是否只返回支持思考的模型（仅对 chat 类型有效）
    
    Returns:
        模型列表
    """
    models_list = []
    
    # chat 模型有嵌套的 models 结构
    if model_type == MODEL_TYPE_CHAT:
        for provider_name, provider_config in models_config.items():
            for model_id, model_config in provider_config.get("models", {}).items():
                # 如果要求 thinking，但模型不支持，则跳过
                if thinking and not model_config.get("thinking", False):
                    continue
                
                for model_name in model_config.get("model_name", []):
                    models_list.append(_create_model_dict(model_name, provider_name, model_type))
    else:
        # embedding 和 rerank 模型使用扁平结构
        for provider_name, config in models_config.items():
            provider_model = config.get("model", "")
            
            # 处理单个字符串或列表的情况
            model_names = provider_model if isinstance(provider_model, list) else [provider_model]
            
            for model_name in model_names:
                if model_name:  # 确保模型名称不为空
                    models_list.append(_create_model_dict(model_name, provider_name, model_type))
    
    return models_list


@llm_api.get("/v1/models", response_class=JSONResponse)
async def list_models(
    modelType: str | None = None, 
    thinking: bool = False
):
    """获取支持的模型列表
    
    Args:
        modelType: 模型类型，"chat"、"embedding"、"rerank" 或 None（返回所有类型），默认为 None；
        thinking: 是否只返回支持思考的模型（仅对 chat 类型有效）
    
    Returns:
        包含模型列表的响应
    """
    # 参数验证
    if modelType is not None and modelType not in VALID_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"无效的模型类型: {modelType}。支持的类型: {', '.join(VALID_MODEL_TYPES)}"
        )
    
    # 如果 thinking 参数在非 chat 类型时使用，给出警告
    if thinking and modelType and modelType != MODEL_TYPE_CHAT:
        logger.warning(f"thinking 参数仅在 chat 类型时有效，当前类型: {modelType}")
    
    models_list = []
    
    # 如果未指定类型，返回所有类型的模型
    if modelType is None:
        models_list.extend(_get_models_from_config(services.llm_models, MODEL_TYPE_CHAT, thinking))
        models_list.extend(_get_models_from_config(services.emb_models, MODEL_TYPE_EMBEDDING))
        models_list.extend(_get_models_from_config(services.rerank_models, MODEL_TYPE_RERANK))
    elif modelType == MODEL_TYPE_CHAT:
        models_list.extend(_get_models_from_config(services.llm_models, MODEL_TYPE_CHAT, thinking))
    elif modelType == MODEL_TYPE_EMBEDDING:
        models_list.extend(_get_models_from_config(services.emb_models, MODEL_TYPE_EMBEDDING))
    elif modelType == MODEL_TYPE_RERANK:
        models_list.extend(_get_models_from_config(services.rerank_models, MODEL_TYPE_RERANK))
    
    return {
        "object": "list",
        "data": models_list
    }


@llm_api.post("/v1/chat/completions")
async def chat_completions(request: ChatReq):
    # 初始化模型，直接传递 temperature 参数
    llm = init_chat_llm(
        model=request.model, 
        thinking=request.thinking, 
        stream=request.stream,
        temperature=request.temperature
    )
    # 获取提示模板
    prompt_template = services.prompts.get("chat_completions")
    prompt = PromptTemplate.from_template(prompt_template)

    if request.stream:
        # 对于流式输出，不使用 StrOutputParser 以便获取 thinking 内容
        stream_chain = prompt | llm

        async def generate():
            async for chunk in stream_chain.astream({"messages": request.messages}):
                logger.info(f"== stream：{chunk}")

                # 处理思考内容
                reasoning_content = getattr(chunk, 'additional_kwargs', {}).get('reasoning_content', '')
                if reasoning_content:
                    think_response = ChatResp(type="think", title="思考中...", content=reasoning_content)
                    yield f"data: {think_response.model_dump_json(exclude_none=True)}\n\n"

                # 处理正常文本内容
                if chunk.content:
                    text_response = ChatResp(type="text", msg=chunk.content)
                    # 假设返回文本数据，按需调整格式
                    yield f"data: {text_response.model_dump_json(exclude_none=True)}\n\n"
            # 添加结束标记
            yield f"data: [DONE]\n\n"

        # 使用StreamingResponse流式传输数据
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        # 构建处理链
        chain = prompt | llm | StrOutputParser()
        # 调用模型
        response = await chain.ainvoke({"messages": request.messages})
        logger.info(f"== chat invoke response：{response}")
        # 返回结果
        reasoning_content = getattr(response, 'additional_kwargs', {}).get('reasoning_content', '')
        response_obj = ChatResp(type="text", msg=response, content=reasoning_content if reasoning_content else None)
        return response_obj.model_dump(exclude_none=True)


@llm_api.post("/v1/embeddings", response_class=JSONResponse)
async def embeddings(request: EmbReq):
    embedding_model = init_embedding_model(request.model)
    return embedding_model.embed_documents(request.input)


@llm_api.post("/v1/prompt", response_class=JSONResponse)
async def embeddings(prompt_name: str):
    if prompt_name:
        prompt = services.prompts.get(prompt_name)
        return {prompt_name: prompt}

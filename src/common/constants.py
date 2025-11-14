
# 多模型对话角色
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# 逻辑删除字段名称
ID = 'id'
TABLE_LOGIC = 'is_deleted'
CREATE_USER = 'create_user'
UPDATE_USER = 'update_user'
CREATE_TIME = 'create_time'
UPDATE_TIME = 'update_time'
# 是否删除相关常量
IS_DELETED_NO = 0
IS_DELETED_YES = 1

# 默认为空消息
DEFAULT_NULL_MESSAGE = '暂无数据'
# 默认成功消息
DEFAULT_SUCCESS_MESSAGE = '操作成功'
# 默认失败消息
DEFAULT_FAILURE_MESSAGE = '操作失败'

# 向量数据库名称
VECTOR_DATABASE_NAME = 'ling_shu'


# 流式输出内容类型
STREAM_CONTENT_TYPE_THINK = 'think'
STREAM_CONTENT_TYPE_TEXT = 'text'
STREAM_CONTENT_TYPE_TIPS = 'tips'
STREAM_CONTENT_TYPE_ERROR = 'error'

STREAM_STOP = 'data: [DONE]\n\n'

ERROR_MESSAGE = "服务器繁忙，请稍后再试。"

# 认证相关配置
TOKEN_STATE_KEY_PREFIX = "chat:token::token:state:"
REFRESH_TOKEN_STATE_KEY_PREFIX = "chat:token::refresh_token:state:"
# 用户信息缓存key
USER_INFO_CACHE_PREFIX = "chat:user::user:id:"
USER_NAME_CACHE_PREFIX = "chat:user:name:user:id:"
ROLE_NAME_CACHE_PREFIX = "chat:user::role:id:"

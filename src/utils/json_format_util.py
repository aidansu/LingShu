import json
import re


def extract_json_from_text(text):
    """
    从包含 ```json ... ``` 的文本中提取 JSON 字符串
    返回解析后的 JSON 字典，如果提取或解析失败则返回 None
    """
    # 使用正则表达式匹配 ```json 开头，到 ``` 结束的内容（支持单行或多行）
    pattern = r"```json\s*({.*?})\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        json_str = match.group(1)
        try:
            # 尝试解析 JSON 字符串
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            return None
    else:
        print("未找到匹配的 JSON 内容")
        return None

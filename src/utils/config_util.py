import os
import typing as t
from io import TextIOWrapper
from pathlib import Path
from re import sub, match

import yaml
from yaml import Loader, SafeLoader, load


def _get_direct_match_path(str: str) -> t.Optional[str]:
    m = match(r"\${{ (.+?) }}", str)
    return m.group(1) if m is not None else None


def _get_path(context: t.Any, path: list[str]) -> t.Optional[t.Any]:
    value = context

    for p in path:
        if value is None:
            return None
        value = getattr(value, p, None)

    return value


def _load_yaml(f: TextIOWrapper, context: t.Optional[t.Any]) -> t.Any:
    def string_constructor(loader: Loader, node: t.Any) -> t.Any:
        v = loader.construct_yaml_str(node)  # type: ignore

        direct_path = _get_direct_match_path(v)

        # if the value of a string is exactly a variable
        # then we directly evaluate it against the context
        # this enables us to return more than just strings
        if direct_path is not None:
            return _get_path(context, direct_path.split("."))

        # convert yaml substitute strings to python formatted strings
        # ${{ path.prop }} -> {path.prop}
        v = sub(r"\${{\s*(.*?)\s*}}", r"{\1}", v)

        return v.format_map(context.__dict__)

    this_loader = SafeLoader
    this_loader.add_constructor("tag:yaml.org,2002:str", string_constructor)

    return load(f, Loader=this_loader)


def get_yaml(file_path: str, context: t.Any) -> t.Any:
    # 如果是绝对路径，直接用；否则以项目根目录为基准
    if not os.path.isabs(file_path):
        # 获取当前文件的上上级目录（即项目根目录）
        base_dir = Path(__file__).resolve().parents[2]
        file_path = str(base_dir / file_path)
    with open(file_path, "r", encoding="utf8") as file:
        return _load_yaml(file, context)


def load_yaml(file_path: str) -> t.Dict[str, t.Any]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"指定的提示文件未找到：'{file_path}'")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            # 使用标准的 safe_load，不添加任何自定义构造器
            data = yaml.safe_load(f)
            if data and 'prompts' in data and isinstance(data.get('prompts'), dict):
                return data['prompts']
            return {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"解析 YAML 文件时出错：'{file_path}'. 错误: {e}")

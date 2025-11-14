import json
import typing as t
from datetime import datetime


def json_serialize_value(obj: t.Any) -> t.Any:
    if isinstance(obj, datetime.datetime):
        return {"__date": obj.isoformat()}
    return obj


def json_serialize(obj: t.Any) -> t.Any:
    return json.dumps(obj, default=json_serialize_value)


def json_deserialize_value(obj: t.Any) -> t.Any:
    if "__date" in obj:
        return datetime.datetime.fromisoformat(obj["__date"])
    return obj


def json_deserialize(obj: t.Any) -> t.Any:
    return json.loads(obj, object_hook=json_deserialize_value)

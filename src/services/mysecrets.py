import typing as t
from os import environ
from types import SimpleNamespace

from pydantic import BaseModel


class EnvSecretsConfig(BaseModel):
    type: t.Literal["env"]


SecretsConfig = EnvSecretsConfig


class _Namespace(SimpleNamespace):
    def __getattribute__(self, __name: str) -> t.Any:
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            return None


def from_config(cfg: SecretsConfig) -> t.Any:
    return _Namespace(**dict(environ))

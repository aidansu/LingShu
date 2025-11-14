
from dataclasses import dataclass, asdict


@dataclass
class SysUser:
    id: int
    # username: str
    # roles: str
    # nick_name: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def to_model(cls, dict_obj: dict):
        return cls(
            id=dict_obj.get("id", -1),
            # username=dict_obj.get("username", ""),
            # roles=dict_obj.get('roles', ""),
            # nick_name=dict_obj.get('nick_name', "")
        )

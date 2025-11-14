import datetime
from typing import Optional

from fastapi import HTTPException, status
from jose import jwt, JWTError

from src.common import constants
from src.configs.config import services

SECRET_KEY = 'UZCO1EiXmb39PEmFYONVDIp1Pv5WMYj7'
ALGORITHM = 'HS256'
AUDIENCE = 'audience'
ISSUER = 'issuer'
expires_in: 7200


class JWTUtil:
    @staticmethod
    async def generate_token(user_info, algorithm: Optional[str] = ALGORITHM, expires_in: Optional[int] = 7200):
        """
        生成JWT
        :param user_info: 用户信息
        :param algorithm: 算法，默认为'HS256'
        :param expires_in: 过期时间（秒），默认为300秒
        :return: JWT字符串
        """
        payload = {'exp': datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=expires_in),
                   'user_info': user_info,
                   'audience': AUDIENCE,
                   'issuer': ISSUER
                   }
        token = jwt.encode(payload, SECRET_KEY, algorithm=algorithm)
        await services.redis.set_value(constants.TOKEN_STATE_KEY_PREFIX + token, 'success', expires_in=expires_in)
        return token

    @staticmethod
    async def generate_refresh_token(user_info, algorithm: Optional[str] = ALGORITHM,
                                     expires_in: Optional[int] = 604800):
        """
        生成JWT
        :param user_info: 用户信息
        :param algorithm: 算法，默认为'HS256'
        :param expires_in: 过期时间（秒），默认为7天
        :return: JWT字符串
        """
        payload = {'exp': datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=expires_in),
                   'user_info': user_info,
                   'audience': AUDIENCE,
                   'issuer': ISSUER
                   }
        token = jwt.encode(payload, SECRET_KEY, algorithm=algorithm)
        await services.redis.set_value(constants.REFRESH_TOKEN_STATE_KEY_PREFIX + token, 'success',
                                       expires_in=expires_in)

        return token

    @staticmethod
    async def parse_jwt(json_web_token: str):
        unauthorized_exception = HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            if await JWTUtil.judge_token(json_web_token):
                payload = jwt.decode(json_web_token, SECRET_KEY, algorithms=ALGORITHM)
                if payload['exp'] < datetime.datetime.now(datetime.UTC).timestamp():
                    raise unauthorized_exception
                return payload['user_info'], payload['exp']
            else:
                raise unauthorized_exception
        except JWTError as exc:
            print(exc)
            raise unauthorized_exception

    @staticmethod
    async def judge_token(token: str):
        """判断token是否存在"""
        redis_token = await services.redis.get_value(constants.TOKEN_STATE_KEY_PREFIX + token)
        if not redis_token:
            refresh_token = await services.redis.get_value(constants.REFRESH_TOKEN_STATE_KEY_PREFIX + token)
            if not refresh_token:
                return False
            else:
                return True
        else:
            return True

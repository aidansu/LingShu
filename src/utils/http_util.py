import json

import aiohttp


class HttpUtil:

    @staticmethod
    async def fetch(method: str, url: str, headers: dict = None, params: dict = None, data: dict = None) -> str:
        if headers is None:
            headers = {}
        async with aiohttp.ClientSession(headers=headers) as session:
            timeout = aiohttp.ClientTimeout(total=3)
            if method.upper() == 'GET':
                async with session.get(url, headers=headers, params=params, timeout=timeout) as response:
                    return await response.text()
            elif method.upper() == 'POST':
                if data:
                    data = json.dumps(data)
                async with session.post(url, headers=headers, data=data, timeout=timeout) as response:
                    return await response.text()
            else:
                raise ValueError("Invalid method. Only GET and POST are supported.")

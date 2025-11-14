import asyncio

import random


class StrUtil:

    @staticmethod
    async def split_message(message: str):
        """
        模拟分词生成效果

        Args:
            message (str): 输入文本

        Returns:
            分词
        """
        length = len(message)
        i = 0
        while i < length:
            step = random.randint(1, 4)
            ask_out = message[i:i + step]
            await asyncio.sleep(0.03)
            yield ask_out
            i = i + step

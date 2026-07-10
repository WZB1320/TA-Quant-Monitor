"""LLM 客户端 — 封装 DeepSeek (OpenAI SDK 兼容) 调用

配置 (.env 文件):
  DEEPSEEK_API_KEY=sk-...
  DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
  DEEPSEEK_MODEL=deepseek-chat

用法:
  from src.ai import LLMClient
  client = LLMClient()
  if client.is_available:
      reply = client.chat("你是...", "分析这组数据...")

  # 流式调用 (SSE 场景):
  for chunk in client.stream_chat("你是...", "分析这组数据..."):
      print(chunk, end="", flush=True)
"""
import os
import logging
from typing import Generator, Optional

from dotenv import load_dotenv

load_dotenv()  # 自动加载项目根目录的 .env

logger = logging.getLogger(__name__)


class LLMClient:
    """DeepSeek LLM 客户端 (同步 + 流式调用)

    使用 OpenAI SDK 兼容接口, 可通过修改 .env 切换到豆包/千问等兼容服务.
    """

    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self._client = None

        if self.api_key and self.api_key != "sk-your-api-key-here":
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
                logger.info("LLM 客户端已初始化: model=%s, base_url=%s", self.model, self.base_url)
            except Exception as e:
                logger.warning("LLM 客户端初始化失败: %s", e)
        else:
            logger.warning("DEEPSEEK_API_KEY 未配置, LLM 功能不可用. 请参考 .env.example")

    @property
    def is_available(self) -> bool:
        """LLM 是否可用 (已配置 API key 且客户端初始化成功)"""
        return self._client is not None

    def chat(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.3, max_tokens: int = 4096) -> str:
        """同步调用 LLM, 返回完整文本回复

        Args:
            system_prompt: 系统提示词 (角色/规则)
            user_prompt: 用户消息 (数据/问题)
            temperature: 温度 (0=确定, 1=随机), 分析报告建议 0.3
            max_tokens: 最大输出 token 数

        Returns:
            LLM 回复文本

        Raises:
            RuntimeError: LLM 未配置或调用失败
        """
        if not self.is_available:
            raise RuntimeError(
                "LLM 未配置. 请在项目根目录创建 .env 文件并设置 DEEPSEEK_API_KEY.\n"
                "参考 .env.example 获取配置模板."
            )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.exception("LLM 调用失败")
            raise RuntimeError(f"LLM 调用失败: {e}") from e

    def stream_chat(self, system_prompt: str, user_prompt: str,
                    temperature: float = 0.3, max_tokens: int = 4096
                    ) -> Generator[str, None, None]:
        """流式调用 LLM, 逐块 yield 文本

        用于 SSE 场景: 前端可实时显示生成过程, 提升用户体验.

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户消息
            temperature: 温度
            max_tokens: 最大输出 token 数

        Yields:
            文本块 (delta content)

        Raises:
            RuntimeError: LLM 未配置或调用失败
        """
        if not self.is_available:
            raise RuntimeError(
                "LLM 未配置. 请在项目根目录创建 .env 文件并设置 DEEPSEEK_API_KEY.\n"
                "参考 .env.example 获取配置模板."
            )

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.exception("LLM 流式调用失败")
            raise RuntimeError(f"LLM 流式调用失败: {e}") from e

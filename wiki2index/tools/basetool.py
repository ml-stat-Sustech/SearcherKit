import abc
from typing import List, Dict, Any

class BaseTool(abc.ABC):
    """
    Agent 工具的抽象基类。 """
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """工具的名称。"""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """工具的描述，用于帮助 LLM 理解其功能。"""
        pass

    @property
    @abc.abstractmethod
    def parameters(self) -> List[Dict[str, Any]]:
        """描述工具输入参数的 schema。"""
        pass

    @abc.abstractmethod
    def call(self, **kwargs) -> str:
        """执行工具并返回一个字符串结果。"""
        pass




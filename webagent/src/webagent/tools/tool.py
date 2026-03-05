"""Base Tool

TODO
- [ ] Support client connection reuse for mcp tools
"""
import abc

class Tool:
    name: str
    description: str | None
    arguments_schema: str | None
    
    async def init(self, *args, **kwargs) -> None:
        """Initialize the tool."""

    @abc.abstractmethod
    async def run(self, *args, **kwargs) -> str:
        """Execute the tool with the provided arguments."""
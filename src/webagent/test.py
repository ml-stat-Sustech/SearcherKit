import asyncio

from tools.BrowseCompP import SearchTool,VisitTool
from tools.base import ToolCall

tool = SearchTool()

asyncio.run(tool.init("http://192.168.77.12:8100/mcp"))

print(tool.name)
print(tool.description)
print(tool.arguments_schema)

print(asyncio.run(tool.run(ToolCall("visit", {"docid": "11451", "goal": "Find out how many complains are there in Kansas City"}), None)))
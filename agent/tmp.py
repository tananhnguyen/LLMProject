# import requests

# # Replace with your actual API key if required, else leave as ''
# API_KEY = "a"
# BASE_URL = "http://0.0.0.0:8001/v1/models"

# headers = {
#     "Authorization": f"Bearer {API_KEY}"
# }

# response = requests.get(BASE_URL, headers=headers)

# # Print the result
# print("Status code:", response.status_code)
# print("Response:", response.json())

import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession

async def list_tools_sse(url: str):
    # Open SSE connection to MCP server
    async with sse_client(url=url) as streams:
        async with ClientSession(*streams) as session:
            # Initialize communication
            await session.initialize()
            # List available tools
            resp = await session.list_tools()
            print("Available tools:")
            for tool in resp.tools:
                print(f"- {tool.name}: {tool.description}")
            
if __name__ == "__main__":
    asyncio.run(list_tools_sse("http://localhost:8000/sse"))

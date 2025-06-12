import asyncio
import os
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
import shutil
import subprocess
import time
from typing import Any
from openai import AsyncOpenAI
from agents import set_default_openai_client

custom_client = AsyncOpenAI(base_url="http://0.0.0.0:8001/v1", api_key="a")
set_default_openai_client(custom_client)

from agents import set_default_openai_api

set_default_openai_api("chat_completions")

from agents import Agent, Runner#, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerSse
from agents.model_settings import ModelSettings

model= "llama-base"

async def run(mcp_server: MCPServer):
    agent = Agent(
        name="Assistant",
        model=model,
        instructions="Use the tools to answer the questions.",
        mcp_servers=[mcp_server],
        model_settings=ModelSettings(tool_choice="auto"),
    )

    # Run the `get_weather` tool
    # message = "What is the Transformer architect? from research papers on arxiv."
    message = "Translate to english arxiv의 연구 논문을 바탕으로 한 sLLM이란 무엇인가?"

    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=[{"role": "user", "content": message}], max_turns=10)#, tracing_disbale = True
    print(result)
    print(result.raw_responses)

    # Final turn
    new_input = result.to_input_list() + [{"role": "user", "content": message}]
    result = await Runner.run(agent, new_input)
    print("Final = ",result.final_output)

async def main():
    async with MCPServerSse(
        name="SSE Python Server",
        params={
            "url": "http://localhost:8000/sse",
        },
    ) as server:
        await run(server)


if __name__ == "__main__":
    # Let's make sure the user has uv installed
    if not shutil.which("uv"):
        raise RuntimeError(
            "uv is not installed. Please install it: https://docs.astral.sh/uv/getting-started/installation/"
        )

    asyncio.run(main())
import asyncio
import json
import os
from typing import Any, Dict, List

from contextlib import AsyncExitStack, asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import AsyncGroq

# # Load environment variables
# load_dotenv()

# # Validate environment variables
# if not os.environ.get("GROQ_API_KEY"):
#     raise ValueError("GROQ_API_KEY not found in environment variables")

import streamlit as st

if "GROQ_API_KEY" not in st.secrets:
    raise ValueError("GROQ_API_KEY not found in Streamlit secrets")



# --- Global Shared State ---
exit_stack = AsyncExitStack()
session: ClientSession | None = None
stdio = None
write = None
groq_client = AsyncGroq(api_key=st.secrets["GROQ_API_KEY"])
model = "gemma2-9b-it"

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    tool_used: bool

# --- MCP Connection Logic ---
async def connect_to_mcp_server(server_script: str = "server.py"):
    global session, stdio, write, exit_stack

    server_params = StdioServerParameters(command="python", args=[server_script])
    stdio, write = await exit_stack.enter_async_context(stdio_client(server_params))
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))
    await session.initialize()

    tools = await session.list_tools()
    print("\n✅ Connected to server with tools:")
    for tool in tools.tools:
        print(f" - {tool.name}: {tool.description}")

# --- Tool Schema Conversion ---
async def get_mcp_tools() -> List[Dict[str, Any]]:
    if not session:
        raise RuntimeError("Session not initialized")

    tools_result = await session.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools_result.tools
    ]

# --- Query Processing Logic ---
async def handle_query(query: str) -> ChatResponse:
    tools = await get_mcp_tools()

    response = await groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        tools=tools,  # type: ignore
        tool_choice="auto",
    )

    assistant_msg = response.choices[0].message
    clean_msg = {
        "role": "assistant",
        "content": assistant_msg.content,
    }

    if assistant_msg.tool_calls:
        clean_msg["tool_calls"] = assistant_msg.tool_calls

    messages = [{"role": "user", "content": query}, clean_msg]

    if assistant_msg.tool_calls:
        for tool_call in assistant_msg.tool_calls:
            tool_response = await session.call_tool( # type: ignore
                tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
            )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_response.content[0].text,  # type: ignore
            })

        messages.append({
            "role": "user",
            "content": "Please explain the answer in more detail using the sources.",
        })

        final_response = await groq_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools, # type: ignore
            tool_choice="none",  # Prevent further tool calls
        )

        return ChatResponse(
            response=final_response.choices[0].message.content or "",
            tool_used=True,
        )

    return ChatResponse(
        response=clean_msg["content"] or "",
        tool_used=False,
    )

# --- Lifespan Context ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting FastAPI MCP client...")
    await connect_to_mcp_server("server.py")
    print("✅ MCP client connected!")
    yield
    print("🧹 Cleaning up MCP session...")
    await exit_stack.aclose()
    print("🛑 Server shut down.")

# --- FastAPI App ---
app = FastAPI(title="MCP AI Agent API", lifespan=lifespan)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: QueryRequest):
    try:
        return await handle_query(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

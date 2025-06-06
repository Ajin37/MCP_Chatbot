import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import AsyncGroq

# Load environment variables
load_dotenv()

# Validate environment variables
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Global setup
session = None
exit_stack = AsyncExitStack()
groq_client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
model = "gemma2-9b-it"  # Better for tool calling
stdio = None
write = None

async def connect_to_server(server_script_path: str = "server.py"):
    """Start server.py and establish MCP session."""
    global session, stdio, write, exit_stack
    
    try:
        server_params = StdioServerParameters(command="python", args=[server_script_path])
        stdio, write = await exit_stack.enter_async_context(stdio_client(server_params))
        session = await exit_stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()
        
        tools = await session.list_tools()
        print("\n✅ Connected to server with tool(s):")
        for tool in tools.tools:
            print(f"   - {tool.name}: {tool.description}")
            
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        raise

async def get_mcp_tools() -> List[Dict[str, Any]]:
    """Fetch available MCP tools in OpenAI-compatible format."""
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

async def process_query(query: str) -> str:
    """Process a user query through Groq LLM with tool support."""
    try:
        tools = await get_mcp_tools()
        
        # Initial LLM call
        response = await groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            tools=tools, # type: ignore
            tool_choice="auto",
        )
        
        assistant_message = response.choices[0].message
        
        # Clean the assistant message to remove unsupported fields
        clean_assistant_message = {
            "role": "assistant",
            "content": assistant_message.content,
        }
        
        # Add tool_calls if they exist
        if assistant_message.tool_calls:
            clean_assistant_message["tool_calls"] = assistant_message.tool_calls
        
        messages = [{"role": "user", "content": query}, clean_assistant_message]
        
        # If tool is called
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                if not session:
                    raise RuntimeError("Session not initialized")
                    
                tool_response = await session.call_tool(
                    tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_response.content[0].text, # type: ignore
                })


            # Add final prompt to encourage elaboration and include sources
            messages.append({
                "role": "user",
                "content": "Please explain the answer in more detail using the sources, and include citations where possible.",
            })

            final_response = await groq_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,  # type: ignore
            tool_choice="none",
            
            )
            
            # # Final response from LLM using tool output
            # final_response = await groq_client.chat.completions.create(
            #     model=model,
            #     messages=messages,
            #     tools=tools, # type: ignore
            #     tool_choice="none",  # Prevent further tool calls
            # )
            
            return f"(✅ Used tool)\n\n{final_response.choices[0].message.content}"
        
        # If tool not used
        return f"(❌ No tool used)\n\n{clean_assistant_message['content']}"
        
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

async def main():
    try:
        print("🚀 Starting client")
        await connect_to_server("server.py")
        print("✅ Connected to server")

        
        while True:
            query = input("\n🗣️ Ask something (or type 'exit' to quit): ").strip()
            if query.lower() in ("exit", "quit", "q"):
                print("👋 Goodbye!")
                break
            
            response = await process_query(query)
            print(f"\n🤖 Response:\n{response}")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        try:
            await exit_stack.aclose()
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup warning: {cleanup_error}")


if __name__ == "__main__":
    import asyncio
    print("🧪 Client starting...")  # <-- Add this
    asyncio.run(main())
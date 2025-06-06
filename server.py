import os
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# # Load API key from .env file
# load_dotenv()
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

import streamlit as st
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]


# Initialize MCP server
mcp = FastMCP(
    name="Tavily Web Search Server",
    host="0.0.0.0",
    port=8050,
)

@mcp.tool()
def web_search(query: str) -> str:
    """Use Tavily to perform a web search and return top 3 results with sources."""
    try:
        url = "https://api.tavily.com/search"
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "max_results": 4,
        }

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json().get("results", [])

        if not results:
            return "No results found."
        
        # Build rich response with source references
        summary = "🔎 Here's a summary based on the top 3 search results:\n\n"

        for i, item in enumerate(results, 1):
            title = item.get("title", "No Title")
            link = item.get("url", "No URL")
            snippet = item.get("content", "No Description")
            summary += f"{i}. **{title}**\n{snippet}\n🔗 [Source]({link})\n\n"

        summary += "\n📚 _This answer is based on real-time Tavily search results._"
        return summary

    except Exception as e:
        return f"Error during search: {str(e)}"

# Start the server
if __name__ == "__main__":
    mcp.run(transport="sse")

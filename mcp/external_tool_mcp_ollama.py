from praisonaiagents import Agent, MCP 

external_tool_agent = Agent(
    instructions="You are a helpful assistant with access to a tool.",
    llm="ollama/qwen3.5:2b",
    tools=MCP(
        "npx @openbnb/mcp-server-airbnb --ignore-robots-txt")
)
user_input = "Find a place in Italy for 20 of May and show me some options."
response = external_tool_agent.start(user_input)
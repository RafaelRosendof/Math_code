from praisonaiagents import Agent, MCP

quant_agent = Agent(
    name="AnalistaQuanti",
    instructions="""Você é um analista financeiro direto e objetivo. 
    SEMPRE use a ferramenta disponível para buscar os dados reais antes de responder sobre um ativo. 
    Nunca invente cotações ou múltiplos da sua própria cabeça.""",
    llm="ollama/qwen2.5-coder:7b", # MODEL HERE 
    tools=[MCP("python yfinance_mcp.py")]
)

print("Terminal is running, type 'exit' to quit.")
print("--------------------------------------------------------------")

while True:

    try:
        
        user_input = input("Você: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        resp = quant_agent.run(user_input)
        print(f"\n\n AnalistaQuanti: {resp} \n\n")
        print("-"*60)

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
from mcp.server.fastmcp import FastMCP
import yfinance as yf
import json 
import signal


mcp = FastMCP(host="127.0.0.1", port=5000, name="YahooMCP")

@mcp.resource("portigolio://ativos")
def get_portfolio() -> str:
    """Retorna a composição atual da carteira de investimentos. """
    carteira = {
        "FIIs": [],
        "Ações": [],
        "Perfil": "Foco em dividendos e valorização de longo prazo"
    }

    return json.dumps(carteira, indent=2)

@mcp.tool()
def get_asset_summary(ticker: str) -> str:
    """
    Busca o preço atual, P/VP, e Dividend Yield de um ativo brasileiro (use o sufixo .SA).
    Exemplo de ticker: "PETR4.SA", "XPML11.SA".
    """
    
    try:
        asset = yf.Ticker(ticker)
        info = asset.info
        
        price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        pvp = info.get('priceToBook', 'N/A')
        dy = info.get('dividendYield', 0)
        if dy != 0:
            dy = f"{dy * 100:.2f}%"
            
        name = info.get('shortName', ticker)
        
        return f"Ativo: {name} ({ticker})\nPreço Atual: R$ {price}\nP/VP: {pvp}\nDividend Yield: {dy}"

    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}. Please ensure the .SA suffix is included for Brazilian stocks."
    
@mcp.prompt()
def analise_fundamentalista(ticker: str) -> str:
    """Prompt estruturado para iniciar uma análise quantitativa de um ativo."""
    return f"""
    Você é um analista quantitativo focado na B3. 
    1. Utilize a ferramenta 'get_asset_summary' para buscar os dados de {ticker}.
    2. Analise os múltiplos retornados (Preço, P/VP, DY).
    3. Responda se, com base nesses dados básicos, o ativo parece descontado ou esticado.
    Seja direto e objetivo na sua avaliação.
    """
    
if __name__ == "__main__":
    mcp.run()
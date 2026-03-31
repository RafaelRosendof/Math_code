import random

import playwright 
import time 
#import beautifulsoup4 as bs4
from playwright.sync_api import sync_playwright


def type_text(page, selector, text):
    page.click(selector)
    # Limpa o campo caso já tenha algo (como o ticker anterior)
    page.keyboard.press("Control+A")
    page.keyboard.press("Backspace")
    for char in text:
        page.type(selector, char, delay=random.randint(70, 250))
    page.wait_for_timeout(500)
    page.keyboard.press("Enter")
    
def collect_stock_data(page):
    """Extrai os dados principais da página atual."""
    data = {}
    try:
        # Pega o preço (o seletor .YMlS1d é comum, mas o wait_for ajuda)
        # price_element = page.wait_for_selector('div[class*="YMlS1d"]', timeout=5000)
        price_element = page.locator('div:has-text("R$")').filter(has_not_text="fechamento").first
        data['price'] = price_element.inner_text()
        
        # Pega a tabela de estatísticas (P/E, Market Cap, etc)
        # O Google usa uma estrutura de chave/valor com classes específicas
        stats = page.query_selector_all('.gyFHrc')
        for item in stats:
            key = item.query_selector('.mfs7Fc').inner_text()
            value = item.query_selector('.P6K39c').inner_text()
            data[key] = value
            
        return data
    except Exception as e:
        print(f"Erro ao coletar dados: {e}")
        return None
    
def collect_stock_data_pro(page):
    try:
        # Em vez de classe, usamos o seletor de atributo que o script sugeriu
        # O Playwright consegue ler esses atributos 'data-' facilmente
        price_element = page.wait_for_selector('div[data-last-price]', timeout=5000)
        print(type(price_element))
        return {
            "ticker": page.url.split("/")[-1].split(":")[0],
            "price": float(price_element.get_attribute("data-last-price")),
            "currency": price_element.get_attribute("data-currency-code"),
            "timestamp": int(price_element.get_attribute("data-last-normal-market-timestamp")),
        }
    except Exception as e:
        print(f"Erro na captura: {e}")
        return None

def test_google_finance(stock, next_stock):
    with sync_playwright() as p:
        # slow_mo ajuda a não ser bloqueado e a simular tempo de reação
        browser = p.firefox.launch(headless=False, slow_mo=400)
        page = browser.new_page()
        
        print(f"--- Iniciando extração de {stock} ---")
        page.goto(f"https://www.google.com/finance/quote/{stock}:BVMF?hl=pt", wait_until="networkidle")

        try:
            # 1. Movimento Humano: Scroll leve
            page.mouse.wheel(0, random.randint(200, 500))
            page.wait_for_timeout(2000)

            # 2. Coleta dados da primeira stock
            #results_1 = collect_stock_data(page)
            results_1 = collect_stock_data_pro(page)
            print(f"Dados de {stock}: {results_1}")

            # 3. Busca a próxima stock (Caminho Humano)
            # O seletor de busca mais seguro é o input dentro do header ou pelo placeholder
            search_input = 'header input[type="text"]'
            page.wait_for_selector(search_input)
            
            print(f"Pesquisando {next_stock}...")
            type_text(page, search_input, next_stock)
            
            # Espera o carregamento da nova página
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(3000)

            # 4. Coleta dados da segunda stock
            #results_2 = collect_stock_data(page)
            results_2 = collect_stock_data_pro(page)
            print(f"Dados de {next_stock}: {results_2}")

        except Exception as e:
            print(f"Erro na interação: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    test_google_finance("PETR4", "VALE3")
    
'''

def get_price_information(ticker: str, exchange: str) -> dict[str, Any]:
    """Gets price information for a given ticker and exchange."""
    url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"
    html_content = fetch_html(url)
    price_div = parse_html_for_price_info(html_content)

    if price_div is None:
        raise ValueError(f"No price information found for {ticker}:{exchange}")

    currency: str = price_div["data-currency-code"]

    return {
        "ticker": ticker,
        "exchange": exchange,
        "price": float(price_div["data-last-price"]),
        "timestamp": int(price_div["data-last-normal-market-timestamp"]),
        "currency": currency,
        "to_USD": get_currency_conversion(currency, "USD")
        if currency != "USD"
        else float(price_div["data-last-price"]),
    }
'''
import random

import playwright 
import time 
#import beautifulsoup4 as bs4
from playwright.sync_api import sync_playwright


def type_text(page, selector, text):
    page.click(selector)
    for c in text:
        page.type(selector, c, delay=random.randint(50, 200))
    page.keyboard.press("Enter")

def human_scroll(page, distance):
    steps = 10
    for _ in range(steps):
        page.mouse.wheel(0, distance / steps)
        page.wait_for_timeout(random.randint(100, 300))

def test_firefox_google(stock, next_stock):
    
    with sync_playwright() as p:
        
        browser = p.firefox.launch(headless=False, slow_mo=500)
        context = browser.new_context(viewport={'width': 1280, 'height': 720})
        page = context.new_page()
        
        url = f"https://www.google.com/finance/quote/{stock}:BVMF?hl=pt"
        page.goto(url, wait_until="networkidle")

        try:
            page.wait_for_timeout(random.randint(1000, 2000))
            
            human_scroll(page, 400)
            page.wait_for_timeout(2000)
            
            
            page.click("button:has-text('6 meses')", timeout=5000)
            print("Clicou em 6 meses")
            page.wait_for_timeout(5000)
            # 2. Clicar em '5 anos'
            page.click("button:has-text('5 anos')", timeout=5000)
            print("Clicou em 5 anos")
            page.wait_for_timeout(2000)
            
            page.wait_for_timeout(2000)
            
            page.wait_for_timeout(2000)
            
            search_selector = 'input[aria-label="Pesquisar ativos, índices e muito mais"]'
            
            page.locator(search_selector).hover()
            
            type_text(page, search_selector, next_stock)
            
            page.wait_for_load_state("networkidle")
            print(f"Agora em {next_stock}")

            page.click("button:has-text('5 anos')")
            page.wait_for_timeout(3000)
            
        except Exception as e:
            print(f"Erro durante a interação: {e}")
        
        finally:
            browser.close()

def main():
    test_firefox_google("PETR4", "VALE3")
    
if __name__ == "__main__":
    main()
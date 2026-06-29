from mcp.server.fastmcp import FastMCP
import requests as req
from bs4 import BeautifulSoup as bs
import signal 
import sys 

mcp = FastMCP(name="StockNews", host="127.0.0.1", port=5000)

def signal_handler(sig, frame):
    print('Exiting gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

@mcp.tool()
def get_stock_news(ticker: str) -> str:
    """
    This function scrapes the latest news headlines (up to 5) from the Finviz stock quote page
    and returns them in a human-readable format, each with a timestamp, headline, and URL.

    Parameters:
        ticker (str): The stock ticker symbol (e.g., "AAPL" for Apple Inc.).

    Returns:
        str: A newline-separated string of the latest headlines in the format:
             "Timestamp - Headline (URL)".
             If an error occurs during the scraping process, returns an error message.

    Raises:
        This function handles its own exceptions and returns an error message as a string
        instead of propagating exceptions.

    Example:
        >>> get_stock_news("GOOGL")
    """
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = req.get(url, headers=headers)
        soup = bs(resp.content, "html.parser")
        
        news_table = soup.find('table', class_='fullview-news-outer')
        rows = news_table.findAll('tr')
        
        news = []
        
        for row in rows[:5]:
            time_tag = row.td.text.strip()
            headline_tag = row.a.text.strip()
            link = row.a['href']
            news.append(f"{time_tag} - {headline_tag} ({link})")
        return "\n".join(news)
        
    except Exception as e:
        return f"An error occurred while fetching news: {str(e)}"
    
if __name__ == "__main__":
    print("StockNews tool is running. Press Ctrl+C to exit.")
    mcp.run()
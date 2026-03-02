package scraping

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type YahooResponse struct {
	Chart struct {
		Result []struct {
			Meta struct {
				Symbol string `json:"symbol"`
			} `json:"meta"`
			Timestamp  []int64 `json:"timestamp"`
			Indicators struct {
				Quote []struct {
					Close []float64 `json:"close"`
				} `json:"quote"`
			} `json:"indicators"`
		} `json:"result"`
	} `json:"chart"`
}

func ScrapingTest(ticker string) ([]float64, error) {
	fmt.Println("Ticker : ", ticker, "Testing the scraping")

	url := fmt.Sprintf("https://query1.finance.yahoo.com/v8/finance/chart/%s?interval=1d&range=100d", ticker)
	req, _ := http.NewRequest("GET", url, nil)

	req.Header.Set("User-Agent", "Mozilla/5.0")

	client := &http.Client{}
	resp, err := client.Do(req)

	if err != nil {
		return nil, err
	}

	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("status code ruim: %d", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)

	//fmt.Println(string(body))

	var data YahooResponse
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, err
	}

	if len(data.Chart.Result) == 0 {
		return nil, fmt.Errorf("nenhum dado encontrado")
	}

	quotes := data.Chart.Result[0].Indicators.Quote[0].Close

	var cleanPrices []float64

	for _, p := range quotes {
		if p != 0 {
			cleanPrices = append(cleanPrices, p)
		}
	}

	return cleanPrices, nil
}

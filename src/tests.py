from src.load import get_market_data


def test_yahoo_finance_api():
    """
    Test that we can fetch S&P 500 (^GSPC) and VIX (^VIX) data from Yahoo Finance
    using the yfinance API.
    """
    sp500, vix = get_market_data()

    assert sp500 is not None, "S&P 500 data is None"
    assert vix is not None, "VIX data is None"
    assert not sp500.empty, "S&P 500 DataFrame is empty"
    assert not vix.empty, "VIX DataFrame is empty"

    print("API test passed: successfully fetched S&P 500 and VIX data.")


if __name__ == "__main__":
    # Run this file directly to execute the test
    test_yahoo_finance_api()

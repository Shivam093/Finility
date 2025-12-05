"""
tests.py – basic tests for Finility project.

You can run this file directly:
    python -m src.tests
"""

from src.load import get_market_data
from src.config import START_DATE, END_DATE


def test_get_market_data():
    """
    Test that we can fetch S&P 500 (^GSPC) and VIX (^VIX) data
    from Yahoo Finance using the yfinance API.
    """
    sp500, vix = get_market_data(start_date=START_DATE, end_date=END_DATE, save_csv=False)

    assert sp500 is not None, "S&P 500 DataFrame is None"
    assert vix is not None, "VIX DataFrame is None"
    assert not sp500.empty, "S&P 500 DataFrame is empty"
    assert not vix.empty, "VIX DataFrame is empty"

    assert sp500.index.is_monotonic_increasing, "S&P 500 index is not sorted by date"
    assert vix.index.is_monotonic_increasing, "VIX index is not sorted by date"

    print("API test passed: successfully fetched S&P 500 and VIX data.")


if __name__ == "__main__":
    test_get_market_data()
    print("All tests passed.")

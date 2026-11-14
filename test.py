from datetime import date

import pandas as pd
import torch
import yfinance as yf

from train import (
    FEATURE_COLUMNS,
    MODEL_VERSION,
    StockReturnModel,
    TickerNotFoundError,
    make_feature_frame,
    model_path_for,
    read_ticker_history,
)

BUY_THRESHOLD = 0.003
SELL_THRESHOLD = -0.003


def resolve_ticker(ticker_or_company: str) -> tuple[str, pd.DataFrame]:
    try:
        return read_ticker_history(ticker_or_company)
    except TickerNotFoundError:
        search = yf.Search(
            ticker_or_company,
            max_results=10,
            news_count=0,
            lists_count=0,
            timeout=8,
        )
        for result in search.quotes:
            if result.get("quoteType") != "EQUITY":
                continue
            candidate = result.get("symbol")
            if not candidate:
                continue
            try:
                return read_ticker_history(candidate)
            except TickerNotFoundError:
                continue

    raise TickerNotFoundError(
        f"No matching stock was found for {ticker_or_company!r} in the historical "
        "dataset. Try its ticker symbol instead."
    )


def suggest_action(
    ticker_or_company: str,
) -> tuple[str, str, float, float, float, date, date]:
    symbol, history = resolve_ticker(ticker_or_company)
    if len(history) < 2:
        raise ValueError(f"Not enough historical data to analyze {symbol}.")

    model_path = model_path_for(symbol)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No trained model found for {symbol}. Run train.py and train this "
            "ticker first."
        )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint["ticker"] != symbol:
        raise ValueError(
            f"The saved model is for {checkpoint['ticker']}, not {symbol}."
        )
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(
            f"The saved model for {symbol} uses an older format. Run train.py "
            "again for this ticker."
        )
    if checkpoint.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError(
            f"The saved model for {symbol} has incompatible features. "
            "Run train.py again for this ticker."
        )

    model = StockReturnModel(len(FEATURE_COLUMNS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    latest_features = make_feature_frame(history).iloc[-1].loc[list(FEATURE_COLUMNS)]
    feature = torch.tensor(latest_features.to_numpy(), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        normalized_return = model(
            (feature - checkpoint["feature_mean"]) / checkpoint["feature_std"]
        )
        predicted_return = (
            normalized_return * checkpoint["target_std"] + checkpoint["target_mean"]
        ).item()

    current_close = float(history["Close"].iloc[-1])
    estimated_close = current_close * (1 + predicted_return)
    last_analyzed_day = history["Date"].iloc[-1].date()
    predicted_day = (pd.Timestamp(last_analyzed_day) + pd.offsets.BDay(1)).date()
    if predicted_return >= BUY_THRESHOLD:
        action = "BUY"
    elif predicted_return <= SELL_THRESHOLD:
        action = "SELL"
    else:
        action = "DO NOTHING"

    return (
        symbol,
        action,
        predicted_return,
        current_close,
        estimated_close,
        last_analyzed_day,
        predicted_day,
    )


def main() -> None:
    ticker = input(
        "Enter a stock ticker or company name to analyze (for example, AAPL): "
    )
    (
        symbol,
        action,
        predicted_return,
        current_close,
        estimated_close,
        last_analyzed_day,
        predicted_day,
    ) = suggest_action(ticker)
    print(f"\n{symbol.upper()} historical-data signal: {action}")
    print(f"Last historical day analyzed: {last_analyzed_day}")
    print(
        f"Predicted day: {predicted_day} "
        "(next weekday; market holidays not accounted for)"
    )
    print(f"Most recent close: {current_close:.2f}")
    print(f"Estimated next close: {estimated_close:.2f}")
    print(f"Estimated next-day change: {predicted_return:+.2%}")
    print(
        "This is an experimental model based only on historical data, "
        "not financial advice."
    )


if __name__ == "__main__":
    main()

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATASET_ROOT = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "bijoybhadra"
    / "stock-market-datasets"
)
FEATURE_COLUMNS = (
    "OpenToClose",
    "HighToClose",
    "LowToClose",
    "DailyReturn",
    "LogVolume",
    "PreviousReturn1",
    "PreviousReturn2",
    "PreviousReturn3",
    "PreviousReturn5",
    "PreviousReturn10",
    "PreviousReturn20",
    "MovingAverageDistance5",
    "MovingAverageDistance10",
    "MovingAverageDistance20",
    "MovingAverageDistance50",
    "Volatility5",
    "Volatility10",
    "Volatility20",
    "Volatility50",
    "VolumeChange",
    "VolumeDistance20",
)
MODEL_DIRECTORY = Path(__file__).parent / "stock_models"
MODEL_VERSION = 2
HIDDEN_SIZES = (64, 32)
MAX_EPOCHS = 400
PATIENCE = 50
MIN_EPOCHS = 40


class TickerNotFoundError(ValueError):
    pass


def find_dataset() -> Path:
    matches = sorted(DATASET_ROOT.glob("versions/*/**/7000Stocks_all.csv"))
    if not matches:
        raise FileNotFoundError(
            f"7000Stocks_all.csv was not found under {DATASET_ROOT}"
        )
    return matches[-1]


def read_ticker_history(
    ticker: str, csv_path: Path | None = None
) -> tuple[str, pd.DataFrame]:
    requested = ticker.strip().lower()
    if not requested:
        raise ValueError("Ticker symbol cannot be empty.")

    source = csv_path or find_dataset()
    columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Stock_Name"]
    exact_rows: list[pd.DataFrame] = []
    suffix_rows: list[pd.DataFrame] = []
    suffix_symbols: set[str] = set()

    for chunk in pd.read_csv(
        source,
        usecols=columns,
        dtype={"Stock_Name": "string"},
        chunksize=500_000,
    ):
        symbols = chunk["Stock_Name"].str.strip().str.lower()
        exact = symbols == requested
        if exact.any():
            exact_rows.append(chunk.loc[exact])

        if "." not in requested:
            suffix_match = symbols.str.split(".", n=1).str[0] == requested
            suffix_match &= ~exact
            if suffix_match.any():
                suffix_rows.append(chunk.loc[suffix_match])
                suffix_symbols.update(symbols.loc[suffix_match].dropna().unique())

    if exact_rows:
        selected_symbol = requested
        rows = exact_rows
    elif len(suffix_symbols) == 1:
        selected_symbol = suffix_symbols.pop()
        rows = suffix_rows
    elif suffix_symbols:
        available = ", ".join(sorted(suffix_symbols))
        raise ValueError(
            f"Ticker {ticker!r} is ambiguous. Use one of these dataset symbols: "
            f"{available}"
        )
    else:
        raise TickerNotFoundError(f"No historical data found for ticker {ticker!r}.")

    history = pd.concat(rows, ignore_index=True)
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = (
        history.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
        .loc[lambda frame: frame["Close"] > 0]
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )
    return selected_symbol, history


def make_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    close = history["Close"]
    daily_return = close.pct_change()
    volume = history["Volume"].clip(lower=0)
    features = pd.DataFrame(
        {
            "OpenToClose": history["Open"] / close - 1,
            "HighToClose": history["High"] / close - 1,
            "LowToClose": history["Low"] / close - 1,
            "DailyReturn": daily_return,
            "LogVolume": np.log1p(volume),
            **{
                f"PreviousReturn{lag}": daily_return.shift(lag)
                for lag in (1, 2, 3, 5, 10, 20)
            },
            **{
                f"MovingAverageDistance{window}": close / close.rolling(window).mean()
                - 1
                for window in (5, 10, 20, 50)
            },
            **{
                f"Volatility{window}": daily_return.rolling(window).std()
                for window in (5, 10, 20, 50)
            },
            "VolumeChange": volume.pct_change(),
            "VolumeDistance20": volume / volume.rolling(20).mean() - 1,
        },
        index=history.index,
    )
    return features.replace([np.inf, -np.inf], np.nan)


class StockReturnModel(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZES[0]),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZES[1], 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def model_path_for(ticker: str) -> Path:
    safe_ticker = re.sub(r"[^a-z0-9._-]", "_", ticker.lower())
    return MODEL_DIRECTORY / f"{safe_ticker}.pt"


def train_ticker(ticker: str, csv_path: Path | None = None) -> Path:
    symbol, history = read_ticker_history(ticker, csv_path)
    features = make_feature_frame(history)
    next_day_return = history["Close"].shift(-1) / history["Close"] - 1
    samples = features.assign(Target=next_day_return).dropna()
    if len(samples) < 100:
        raise ValueError(
            f"Ticker {symbol} has only {len(samples)} usable days; at least 100 "
            "are required."
        )

    x_values = samples.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    y_values = samples["Target"].to_numpy(dtype=np.float32).reshape(-1, 1)

    split_index = int(len(samples) * 0.8)
    train_x_values = x_values[:split_index]
    train_y_values = y_values[:split_index]
    validation_x_values = x_values[split_index:]
    validation_y_values = y_values[split_index:]

    validation_x_mean = train_x_values.mean(axis=0, keepdims=True)
    validation_x_std = train_x_values.std(axis=0, keepdims=True)
    validation_x_std[validation_x_std < 1e-6] = 1
    validation_y_mean = train_y_values.mean(axis=0, keepdims=True)
    validation_y_std = train_y_values.std(axis=0, keepdims=True)
    validation_y_std[validation_y_std < 1e-6] = 1

    train_x = torch.tensor(
        (train_x_values - validation_x_mean) / validation_x_std,
        dtype=torch.float32,
    )
    train_y = torch.tensor(
        (train_y_values - validation_y_mean) / validation_y_std,
        dtype=torch.float32,
    )
    validation_x = torch.tensor(
        (validation_x_values - validation_x_mean) / validation_x_std,
        dtype=torch.float32,
    )
    validation_y = torch.tensor(
        (validation_y_values - validation_y_mean) / validation_y_std,
        dtype=torch.float32,
    )

    torch.manual_seed(42)
    validation_model = StockReturnModel(len(FEATURE_COLUMNS))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(validation_model.parameters(), lr=0.001)
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        validation_model.train()
        prediction = validation_model(train_x)
        loss = loss_fn(prediction, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        validation_model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(validation_model(validation_x), validation_y)

        if validation_loss.item() < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss.item()
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in validation_model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch >= MIN_EPOCHS and epochs_without_improvement >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"Training failed to produce a model for {symbol}.")

    validation_model.load_state_dict(best_state)
    validation_model.eval()
    with torch.no_grad():
        training_prediction = (
            validation_model(train_x).numpy() * validation_y_std + validation_y_mean
        )
        validation_prediction = (
            validation_model(validation_x).numpy() * validation_y_std
            + validation_y_mean
        )
    training_mae = float(np.mean(np.abs(training_prediction - train_y_values)))
    training_baseline_mae = float(np.mean(np.abs(train_y_values - validation_y_mean)))
    validation_mae = float(np.mean(np.abs(validation_prediction - validation_y_values)))
    baseline_mae = float(np.mean(np.abs(validation_y_values - validation_y_mean)))

    x_mean = x_values.mean(axis=0, keepdims=True)
    x_std = x_values.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1
    y_mean = y_values.mean(axis=0, keepdims=True)
    y_std = y_values.std(axis=0, keepdims=True)
    y_std[y_std < 1e-6] = 1
    x = torch.tensor((x_values - x_mean) / x_std, dtype=torch.float32)
    y = torch.tensor((y_values - y_mean) / y_std, dtype=torch.float32)

    torch.manual_seed(42)
    model = StockReturnModel(len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(best_epoch):
        model.train()
        prediction = model(x)
        loss = loss_fn(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_path = model_path_for(symbol)
    torch.save(
        {
            "model_version": MODEL_VERSION,
            "ticker": symbol,
            "feature_columns": list(FEATURE_COLUMNS),
            "model_state_dict": model.state_dict(),
            "feature_mean": torch.tensor(x_mean, dtype=torch.float32),
            "feature_std": torch.tensor(x_std, dtype=torch.float32),
            "target_mean": torch.tensor(y_mean, dtype=torch.float32),
            "target_std": torch.tensor(y_std, dtype=torch.float32),
        },
        model_path,
    )
    print(f"Trained {symbol} using {len(samples)} historical days.")
    print(
        f"Training MAE: {training_mae:.4%} "
        f"(constant-mean baseline: {training_baseline_mae:.4%})."
    )
    print(
        f"Best chronological validation MAE: {validation_mae:.4%} "
        f"(constant-mean baseline: {baseline_mae:.4%}; epoch {best_epoch})."
    )
    print(f"Saved model to {model_path}")
    return model_path


def main() -> None:
    ticker = input("Enter the ticker to train (for example, AAPL): ")
    train_ticker(ticker)


if __name__ == "__main__":
    main()

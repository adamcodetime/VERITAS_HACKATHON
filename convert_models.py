import argparse
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from train import FEATURE_COLUMNS, MODEL_VERSION, StockReturnModel


ROOT = Path(__file__).resolve().parent


class ExportableStockModel(nn.Module):
    def __init__(self, checkpoint: dict) -> None:
        super().__init__()
        self.model = StockReturnModel(len(FEATURE_COLUMNS))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.register_buffer("feature_mean", checkpoint["feature_mean"])
        self.register_buffer("feature_std", checkpoint["feature_std"])
        self.register_buffer("target_mean", checkpoint["target_mean"])
        self.register_buffer("target_std", checkpoint["target_std"])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized_features = (features - self.feature_mean) / self.feature_std
        normalized_return = self.model(normalized_features)
        return normalized_return * self.target_std + self.target_mean


def convert_model(model_path: Path, output_directory: Path) -> Path:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(f"{model_path} does not use model version {MODEL_VERSION}.")
    if checkpoint.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError(f"{model_path} has incompatible feature columns.")

    model_name = model_path.stem.split(".", maxsplit=1)[0].lower()
    onnx_path = output_directory / f"{model_name}.onnx"
    data_path = output_directory / f"{model_name}.onnx.data"
    model = ExportableStockModel(checkpoint).eval()
    example_features = torch.zeros((1, len(FEATURE_COLUMNS)), dtype=torch.float32)

    torch.onnx.export(
        model,
        (example_features,),
        onnx_path,
        input_names=["features"],
        output_names=["predicted_return"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "predicted_return": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
        external_data=False,
    )

    exported_model = onnx.load(onnx_path)
    onnx.save_model(
        exported_model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=0,
    )
    onnx.checker.check_model(onnx_path)
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert stock_models/*.pt checkpoints to ONNX models."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "stock_models",
        help="Directory containing .pt checkpoints (default: stock_models).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output",
        help="Directory for .onnx models and their .onnx.data files (default: output).",
    )
    args = parser.parse_args()

    model_paths = sorted(args.source.glob("*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No .pt checkpoints found in {args.source}.")

    names = [path.stem.split(".", maxsplit=1)[0].lower() for path in model_paths]
    if len(names) != len(set(names)):
        raise ValueError("Multiple checkpoints map to the same output model name.")

    args.output.mkdir(parents=True, exist_ok=True)
    for model_path in model_paths:
        onnx_path = convert_model(model_path, args.output)
        print(f"Converted {model_path.name} -> {onnx_path.name}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sklearn IncrementalPCA pickle to portable NumPy .npz")
    parser.add_argument("--input", required=True, help="Path to .pkl file")
    parser.add_argument("--output", required=True, help="Path to output .npz file")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    with inp.open("rb") as f:
        model = pickle.load(f)

    payload: dict[str, np.ndarray] = {
        "components": np.asarray(model.components_, dtype=np.float32),
        "mean": np.asarray(model.mean_, dtype=np.float32),
        "explained_variance": np.asarray(model.explained_variance_, dtype=np.float32),
        "explained_variance_ratio": np.asarray(model.explained_variance_ratio_, dtype=np.float32),
        "singular_values": np.asarray(model.singular_values_, dtype=np.float32),
        "n_components": np.asarray([int(model.n_components_)], dtype=np.int32),
        "n_features_in": np.asarray([int(model.n_features_in_)], dtype=np.int32),
        "batch_size": np.asarray([int(getattr(model, "batch_size", 0) or 0)], dtype=np.int32),
        "whiten": np.asarray([int(bool(getattr(model, "whiten", False)))], dtype=np.int32),
    }

    # Optional attrs (store if present)
    noise_variance = getattr(model, "noise_variance_", None)
    if noise_variance is not None:
        payload["noise_variance"] = np.asarray([float(noise_variance)], dtype=np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)

    print(f"Exported: {out}")
    print(f"components shape: {payload['components'].shape}")
    print(f"mean shape: {payload['mean'].shape}")


if __name__ == "__main__":
    main()

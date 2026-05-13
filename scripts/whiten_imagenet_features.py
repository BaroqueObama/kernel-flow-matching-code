"""Whiten DINOv2+PCA features by rescaling each component by 1/sqrt(EVR_i).

Equalizes signal across PCA components, removing the eigenvalue ordering
that concentrates information in the leading dimensions.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Whiten DINOv2+PCA features")
    parser.add_argument("--input-dir", type=str, default="data/imagenet_dino_pca")
    parser.add_argument("--output-dir", type=str, default="data/imagenet_dino_pca_whitened")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    assert in_dir.exists(), f"Input directory not found: {in_dir}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pca_path = in_dir / "pca_model.joblib"
    assert pca_path.exists(), f"PCA model not found: {pca_path}"
    pca = joblib.load(pca_path)

    evr = pca.explained_variance_ratio_
    assert evr is not None and len(evr) > 0, "PCA model has no explained_variance_ratio_"
    print(f"PCA explained variance ratio (first 8): {evr[:8]}")
    print(f"Total EVR: {evr.sum():.4f}")

    with open(in_dir / "metadata.json") as f:
        metadata = json.load(f)

    target_dims = metadata["target_dims"]
    print(f"Target dims: {target_dims}")

    for d in target_dims:
        evr_d = evr[:d]
        scale = 1.0 / np.sqrt(np.clip(evr_d, 1e-8, None))
        assert scale.max() < 200, (
            f"Extreme amplification {scale.max():.0f}x at d={d} — check PCA model integrity"
        )
        scale_tensor = torch.from_numpy(scale.astype(np.float32))

        print(f"\nd={d}: EVR range [{evr_d.min():.4f}, {evr_d.max():.4f}]")
        print(f"  Scale range [{scale.min():.2f}, {scale.max():.2f}]")

        for split in ["train", "val", "test"]:
            in_path = in_dir / f"{split}_d{d}.pt"
            if not in_path.exists():
                print(f"  Skipping {in_path} (not found)")
                continue

            data = torch.load(in_path, map_location="cpu", weights_only=True)

            whitened = {}
            n_images = 0
            for cid, feats in data.items():
                assert feats.shape[1] == d, f"Feature dim mismatch: {feats.shape[1]} != {d}"
                w = feats * scale_tensor.unsqueeze(0)
                assert torch.isfinite(w).all(), f"Non-finite values in class {cid} after whitening"
                whitened[cid] = w
                n_images += feats.shape[0]

            out_path = out_dir / f"{split}_d{d}.pt"
            torch.save(whitened, out_path)
            print(f"  {split}_d{d}: {len(whitened)} classes, {n_images} images")

    metadata_out = dict(metadata)
    metadata_out["whitening"] = {
        "method": "inverse_sqrt_explained_variance_ratio",
        "description": "Each component scaled by 1/sqrt(EVR_i), equalizing PCA signal",
        "evr_first_8": evr[:8].tolist(),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata_out, f, indent=2)

    shutil.copy(in_dir / "pca_model.joblib", out_dir / "pca_model.joblib")
    print(f"\nWhitened features saved to {out_dir}")


if __name__ == "__main__":
    main()

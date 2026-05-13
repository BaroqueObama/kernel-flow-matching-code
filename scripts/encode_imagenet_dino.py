"""Encode ImageNet with DINOv2 CLS tokens and PCA-project to target dimensions."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Encode ImageNet with DINOv2+PCA")
    parser.add_argument("--output-dir", type=str, default="data/imagenet_dino_pca")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=None,
        help="Limit images per class (for debugging)",
    )
    parser.add_argument(
        "--min-class-size", type=int, default=600, help="Drop classes with fewer images than this"
    )
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load ImageNet via HuggingFace
    print("Loading ImageNet from HuggingFace...")
    from datasets import load_dataset
    from dotenv import load_dotenv

    load_dotenv()
    dataset = load_dataset("ILSVRC/imagenet-1k", split="train", trust_remote_code=True, num_proc=4, token=os.environ.get("HF_TOKEN"))
    print(f"Loaded {len(dataset)} images")

    # Step 2: Load DINOv2
    DINOV2_REPO = "facebookresearch/dinov2"
    DINOV2_MODEL = "dinov2_vits14"
    print(f"Loading {DINOV2_MODEL} from {DINOV2_REPO}...")
    device = torch.device(args.device)
    model = torch.hub.load(DINOV2_REPO, DINOV2_MODEL)
    model = model.to(device).eval()

    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Step 3: Encode all images, group by class
    print("Encoding images with DINOv2...")
    checkpoint_path = out_dir / "_encoding_checkpoint.pt"
    checkpoint_interval = 50000
    class_features: dict[int, list[torch.Tensor]] = {}
    start_idx = 0
    n_images = len(dataset)

    if checkpoint_path.exists():
        print("Resuming from encoding checkpoint...")
        ckpt = torch.load(checkpoint_path, weights_only=True)
        start_idx = ckpt["next_index"]
        for cid, stacked in ckpt["class_features"].items():
            class_features[cid] = list(stacked.unbind(0))
        n_resumed = sum(len(v) for v in class_features.values())
        print(
            f"  Loaded {n_resumed} features from {len(class_features)} classes, "
            f"resuming from index {start_idx}/{n_images}"
        )

    for start in tqdm(range(start_idx, n_images, args.batch_size), desc="Encoding"):
        end = min(start + args.batch_size, n_images)
        batch_items = dataset[start:end]
        images = batch_items["image"]
        labels = batch_items["label"]

        tensors = []
        valid_labels = []
        for img, label in zip(images, labels):
            if args.max_images_per_class is not None:
                existing = len(class_features.get(label, []))
                if existing >= args.max_images_per_class:
                    continue
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensors.append(transform(img))
            valid_labels.append(label)

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            features = model(batch).float().cpu()

        for feat, label in zip(features, valid_labels):
            if label not in class_features:
                class_features[label] = []
            class_features[label].append(feat)

        if (end - start_idx) >= checkpoint_interval and end % checkpoint_interval < args.batch_size:
            stacked_ckpt = {cid: torch.stack(fs) for cid, fs in class_features.items()}
            tmp_path = checkpoint_path.with_suffix(".pt.tmp")
            torch.save({"class_features": stacked_ckpt, "next_index": end}, tmp_path)
            os.replace(tmp_path, checkpoint_path)
            n_so_far = sum(v.shape[0] for v in stacked_ckpt.values())
            tqdm.write(f"  Checkpoint saved: {n_so_far} features, next_index={end}")

    for cid in class_features:
        class_features[cid] = torch.stack(class_features[cid])

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Encoding complete, removed checkpoint file.")

    n_classes = len(class_features)
    total_images = sum(v.shape[0] for v in class_features.values())
    feat_dim = next(iter(class_features.values())).shape[1]
    print(f"Encoded {total_images} images across {n_classes} classes, feature dim={feat_dim}")

    # Step 4: Class split (800/100/100)
    all_class_ids = sorted(class_features.keys())

    if args.min_class_size > 0:
        before = len(all_class_ids)
        min_sz = args.min_class_size
        all_class_ids = [c for c in all_class_ids if class_features[c].shape[0] >= min_sz]
        dropped = before - len(all_class_ids)
        if dropped > 0:
            print(f"Dropped {dropped} classes with < {args.min_class_size} images")

    rng = np.random.RandomState(42)
    shuffled = list(all_class_ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_classes = shuffled[:n_train]
    val_classes = shuffled[n_train : n_train + n_val]
    test_classes = shuffled[n_train + n_val :]
    print(f"Split: {len(train_classes)} train, {len(val_classes)} val, {len(test_classes)} test")

    splits = {"train": train_classes, "val": val_classes, "test": test_classes}

    # Step 5: Fit PCA on train split
    print("Fitting PCA on train split features...")
    from sklearn.decomposition import PCA

    train_features = torch.cat([class_features[c] for c in train_classes]).numpy()
    print(f"PCA input: {train_features.shape}")

    max_d = 64
    pca = PCA(n_components=max_d)
    pca.fit(train_features)
    variance_explained = pca.explained_variance_ratio_.cumsum().tolist()
    print(
        f"PCA variance explained (cumulative): d=8: {variance_explained[7]:.3f}, "
        f"d=16: {variance_explained[15]:.3f}, d=32: {variance_explained[31]:.3f}, "
        f"d=64: {variance_explained[63]:.3f}"
    )

    import joblib

    joblib.dump(pca, out_dir / "pca_model.joblib")
    print(f"Saved PCA model to {out_dir / 'pca_model.joblib'}")

    # Step 6: Project and standardize
    all_features_pca = {}
    for cid, feats in class_features.items():
        projected = pca.transform(feats.numpy())
        all_features_pca[cid] = projected

    train_projected = np.concatenate([all_features_pca[c] for c in train_classes])
    mean = train_projected.mean(axis=0)
    std = train_projected.std(axis=0)
    std[std < 1e-8] = 1.0

    for cid in all_features_pca:
        all_features_pca[cid] = (all_features_pca[cid] - mean) / std

    # Step 7: Save per (split, d)
    target_dims = [8, 16, 32, 64]

    for split_name, split_classes in splits.items():
        for d in target_dims:
            split_data = {}
            for cid in split_classes:
                split_data[cid] = torch.from_numpy(all_features_pca[cid][:, :d].astype(np.float32))
            out_path = out_dir / f"{split_name}_d{d}.pt"
            torch.save(split_data, out_path)
            n_imgs = sum(v.shape[0] for v in split_data.values())
            print(f"Saved {out_path}: {len(split_data)} classes, {n_imgs} images, d={d}")

    # Step 8: Save metadata
    class_sizes = {str(c): class_features[c].shape[0] for c in all_class_ids}
    metadata = {
        "encoder": DINOV2_MODEL,
        "encoder_repo": DINOV2_REPO,
        "feature_dim": feat_dim,
        "pca_max_components": max_d,
        "pca_variance_explained_cumulative": variance_explained,
        "standardization_mean": mean.tolist(),
        "standardization_std": std.tolist(),
        "target_dims": target_dims,
        "train_classes": train_classes,
        "val_classes": val_classes,
        "test_classes": test_classes,
        "class_sizes": class_sizes,
        "min_class_size_filter": args.min_class_size,
        "max_images_per_class": args.max_images_per_class,
        "total_images_encoded": total_images,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")
    print("Done!")


if __name__ == "__main__":
    main()

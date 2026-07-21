import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]
CLS_TO_ID = {name: i for i, name in enumerate(ATTRIBUTES)}


def load_classifier(classifier_ckpt: Path, device: torch.device):
    classifier_ckpt = Path(classifier_ckpt).resolve()

    if not classifier_ckpt.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_ckpt}")

    ckpt = torch.load(classifier_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    def get_tensor(*keys):
        for key in keys:
            if key in state_dict:
                return state_dict[key]
        return None

    weight = get_tensor("ema_classifier.weight", "classifier.weight")
    bias = get_tensor("ema_classifier.bias", "classifier.bias")
    conds_mean = get_tensor("conds_mean")
    conds_std = get_tensor("conds_std")

    if weight is None or bias is None:
        keys = [k for k in state_dict.keys() if "classifier" in k]
        raise KeyError(f"Could not find classifier weights. Classifier keys found: {keys[:50]}")

    if conds_mean is None or conds_std is None:
        keys = [k for k in state_dict.keys() if "cond" in k or "classifier" in k]
        raise KeyError(
            "This script expects conds_mean and conds_std inside the classifier checkpoint. "
            f"Relevant keys found: {keys[:50]}"
        )

    classifier = torch.nn.Linear(weight.shape[1], weight.shape[0])
    classifier.weight.data.copy_(weight.float())
    classifier.bias.data.copy_(bias.float())
    classifier = classifier.to(device)
    classifier.eval()

    conds_mean = torch.as_tensor(conds_mean).float().reshape(1, -1).to(device)
    conds_std = torch.as_tensor(conds_std).float().reshape(1, -1).to(device)

    return classifier, conds_mean, conds_std


def load_metadata(metadata_path: Path):
    metadata_path = Path(metadata_path).resolve()

    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--classifier-ckpt",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc_cls/last.ckpt",
    )
    parser.add_argument(
        "--zsem-train",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_train.npy",
    )
    parser.add_argument(
        "--train-metadata",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_train_metadata.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/smiling_train_probs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    zsem_train_path = Path(args.zsem_train).resolve()
    metadata_path = Path(args.train_metadata).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not zsem_train_path.exists():
        raise FileNotFoundError(f"Training z_sem file not found: {zsem_train_path}")

    zsem_train = np.load(zsem_train_path).astype(np.float32)

    if zsem_train.ndim != 2:
        raise ValueError(f"Expected train z_sem shape [N, D], got {zsem_train.shape}")

    num_embeddings, z_dim = zsem_train.shape

    print(f"Loaded training z_sem embeddings: {zsem_train_path}")
    print(f"Shape: {zsem_train.shape}")

    metadata_rows = load_metadata(metadata_path)

    if metadata_rows is not None:
        if len(metadata_rows) != num_embeddings:
            raise ValueError(
                f"Metadata row count does not match embeddings. "
                f"metadata rows={len(metadata_rows)}, embeddings={num_embeddings}"
            )
        print(f"Loaded training metadata: {metadata_path}")
    else:
        print(f"No metadata file found at {metadata_path}. Saving row indices only.")

    classifier, conds_mean, conds_std = load_classifier(
        classifier_ckpt=Path(args.classifier_ckpt),
        device=device,
    )

    if classifier.in_features != z_dim:
        raise ValueError(
            f"Classifier expects z_sem dimension {classifier.in_features}, "
            f"but train embeddings have dimension {z_dim}"
        )

    smiling_idx = CLS_TO_ID["Smiling"]

    all_probs = []

    with torch.no_grad():
        for start in range(0, num_embeddings, args.batch_size):
            end = min(start + args.batch_size, num_embeddings)

            batch = torch.from_numpy(zsem_train[start:end]).float().to(device)
            batch_norm = (batch - conds_mean) / conds_std

            logits = classifier(batch_norm)
            probs = torch.sigmoid(logits[:, smiling_idx])

            all_probs.append(probs.detach().cpu())

            print(f"Processed {end}/{num_embeddings}", end="\r")

    smiling_probs = torch.cat(all_probs, dim=0).numpy().astype(np.float32)
    smiling_pred = smiling_probs >= args.threshold

    print(f"\nComputed Smiling probabilities for {len(smiling_probs)} embeddings.")

    npy_path = out_dir / "train_smiling_probabilities.npy"
    np.save(npy_path, smiling_probs)

    npz_path = out_dir / "train_smiling_probabilities_with_indices.npz"

    if metadata_rows is not None:
        embedding_indices = np.array(
            [int(row["embedding_index"]) for row in metadata_rows],
            dtype=np.int64,
        )
    else:
        embedding_indices = np.arange(num_embeddings, dtype=np.int64)

    np.savez(
        npz_path,
        row_index=np.arange(num_embeddings, dtype=np.int64),
        embedding_index=embedding_indices,
        smiling_probability=smiling_probs,
        smiling_prediction=smiling_pred.astype(np.bool_),
    )

    csv_path = out_dir / "train_smiling_probabilities.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "row_index_in_train_npy",
            "embedding_index_original_dataset",
            "smiling_probability",
            "smiling_percentage",
            "smiling_prediction",
            "filename",
            "image_path",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_embeddings):
            if metadata_rows is not None:
                row = metadata_rows[i]
                embedding_index = row.get("embedding_index", i)
                filename = row.get("filename", "")
                image_path = row.get("image_path", "")
            else:
                embedding_index = i
                filename = ""
                image_path = ""

            writer.writerow({
                "row_index_in_train_npy": i,
                "embedding_index_original_dataset": embedding_index,
                "smiling_probability": f"{float(smiling_probs[i]):.8f}",
                "smiling_percentage": f"{100.0 * float(smiling_probs[i]):.4f}",
                "smiling_prediction": int(bool(smiling_pred[i])),
                "filename": filename,
                "image_path": image_path,
            })

    sorted_csv_path = out_dir / "train_smiling_probabilities_sorted_desc.csv"

    sorted_indices = np.argsort(-smiling_probs)

    with open(sorted_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rank",
            "row_index_in_train_npy",
            "embedding_index_original_dataset",
            "smiling_probability",
            "smiling_percentage",
            "smiling_prediction",
            "filename",
            "image_path",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, i in enumerate(sorted_indices, start=1):
            if metadata_rows is not None:
                row = metadata_rows[int(i)]
                embedding_index = row.get("embedding_index", int(i))
                filename = row.get("filename", "")
                image_path = row.get("image_path", "")
            else:
                embedding_index = int(i)
                filename = ""
                image_path = ""

            writer.writerow({
                "rank": rank,
                "row_index_in_train_npy": int(i),
                "embedding_index_original_dataset": embedding_index,
                "smiling_probability": f"{float(smiling_probs[i]):.8f}",
                "smiling_percentage": f"{100.0 * float(smiling_probs[i]):.4f}",
                "smiling_prediction": int(bool(smiling_pred[i])),
                "filename": filename,
                "image_path": image_path,
            })

    summary = {
        "zsem_train_path": str(zsem_train_path),
        "classifier_ckpt": str(Path(args.classifier_ckpt).resolve()),
        "metadata_path": str(metadata_path) if metadata_rows is not None else None,
        "num_embeddings": int(num_embeddings),
        "z_dim": int(z_dim),
        "attribute": "Smiling",
        "attribute_index": int(smiling_idx),
        "threshold": float(args.threshold),
        "mean_probability": float(smiling_probs.mean()),
        "std_probability": float(smiling_probs.std()),
        "min_probability": float(smiling_probs.min()),
        "max_probability": float(smiling_probs.max()),
        "num_predicted_smiling": int(smiling_pred.sum()),
        "num_predicted_not_smiling": int((~smiling_pred).sum()),
        "outputs": {
            "aligned_npy": str(npy_path),
            "aligned_npz_with_indices": str(npz_path),
            "aligned_csv": str(csv_path),
            "sorted_csv_desc": str(sorted_csv_path),
        },
    }

    summary_path = out_dir / "train_smiling_probabilities_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(f"  Aligned probabilities .npy: {npy_path}")
    print(f"  Aligned probabilities + indices .npz: {npz_path}")
    print(f"  Aligned CSV: {csv_path}")
    print(f"  Sorted CSV: {sorted_csv_path}")
    print(f"  Summary JSON: {summary_path}")

    print("\nImportant:")
    print("  train_smiling_probabilities.npy is aligned with ffhq256_diffae_zsem_train.npy.")
    print("  That means probability[i] corresponds to zsem_train[i].")
    print("  The CSV also stores row_index_in_train_npy, original embedding_index, filename, and image_path.")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont


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


def load_diffae_ffhq256_autoencoder(diffae_root: Path, checkpoint_path: Path, device: torch.device):
    diffae_root = Path(diffae_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

    if not diffae_root.exists():
        raise FileNotFoundError(f"DiffAE repo not found: {diffae_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint_path}")

    sys.path.insert(0, str(diffae_root))

    old_cwd = os.getcwd()
    os.chdir(diffae_root)

    try:
        from templates import ffhq256_autoenc
        from config import PretrainConfig
        from experiment import LitModel

        conf = ffhq256_autoenc()
        conf.pretrain = PretrainConfig(
            name="ffhq256_autoenc",
            path=str(checkpoint_path),
        )
        conf.latent_infer_path = None

        model = LitModel(conf)
        model = model.to(device)
        model.eval()

        if hasattr(model, "ema_model"):
            model.ema_model.eval()

        for p in model.parameters():
            p.requires_grad_(False)

    finally:
        os.chdir(old_cwd)

    return model


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

    if weight is None or bias is None:
        keys = list(state_dict.keys())[:50]
        raise KeyError(f"Could not find classifier weights. First checkpoint keys: {keys}")

    conds_mean = get_tensor("conds_mean")
    conds_std = get_tensor("conds_std")

    if conds_mean is None or conds_std is None:
        keys = [k for k in state_dict.keys() if "cond" in k or "classifier" in k]
        raise KeyError(
            "Your classifier checkpoint does not contain conds_mean and conds_std. "
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


def tensor_to_pil(x):
    x = x.detach().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def load_original_image(path, size=256):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def draw_multiline(draw, text, x, y, font, max_width, line_h=14, max_lines=8):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        trial = word if current == "" else current + " " + word
        width = draw.textbbox((0, 0), trial, font=font)[2]

        if width <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    for line in lines[:max_lines]:
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += line_h


def make_grid(originals, reconstructions, captions_original, captions_recon, out_path):
    cell = 256
    pad = 18
    title_h = 42
    caption_h = 110
    cols = 2
    rows = 2

    width = cols * cell + (cols + 1) * pad
    height = title_h + rows * cell + rows * caption_h + (rows + 1) * pad

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title = "Pair 0: classifier scores vs original images and DiffAE reconstructions"
    draw.text((pad, 12), title, fill=(0, 0, 0), font=font)

    for col in range(2):
        x = pad + col * (cell + pad)

        y_original = title_h + pad

        if originals[col] is not None:
            canvas.paste(originals[col], (x, y_original))
        else:
            draw.rectangle(
                [x, y_original, x + cell, y_original + cell],
                outline=(160, 160, 160),
                width=2,
            )
            draw.text(
                (x + 70, y_original + 120),
                "Original not found",
                fill=(80, 80, 80),
                font=font,
            )

        draw_multiline(
            draw,
            captions_original[col],
            x,
            y_original + cell + 8,
            font,
            max_width=cell,
        )

        y_recon = title_h + pad + cell + caption_h + pad
        canvas.paste(reconstructions[col], (x, y_recon))

        draw_multiline(
            draw,
            captions_recon[col],
            x,
            y_recon + cell + 8,
            font,
            max_width=cell,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--diffae-root",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/diffae/",
    )
    parser.add_argument(
        "--autoenc-ckpt",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt",
    )
    parser.add_argument(
        "--classifier-ckpt",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc_cls/last.ckpt",
    )
    parser.add_argument(
        "--zsem-pairs",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_test_pairs.npy",
    )
    parser.add_argument(
        "--xt-pairs",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_xt_test_pairs.npy",
    )
    parser.add_argument(
        "--pair-metadata",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_test_pairs_metadata.csv",
    )
    parser.add_argument(
        "--pair-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--T",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--out-grid",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/pair0_young_smiling_grid.png",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/pair0_young_smiling_scores.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    zsem_pairs = np.load(args.zsem_pairs).astype(np.float32)
    xt_pairs = np.load(args.xt_pairs).astype(np.float32)

    if zsem_pairs.ndim != 3 or zsem_pairs.shape[1] != 2:
        raise ValueError(f"Expected zsem pairs with shape [N, 2, D], got {zsem_pairs.shape}")

    if xt_pairs.ndim != 5 or xt_pairs.shape[1] != 2:
        raise ValueError(f"Expected xT pairs with shape [N, 2, 3, H, W], got {xt_pairs.shape}")

    pair_index = args.pair_index

    z_pair = torch.from_numpy(zsem_pairs[pair_index]).float().to(device)
    xt_pair = torch.from_numpy(xt_pairs[pair_index]).float().to(device)

    metadata_row = None
    pair_metadata_path = Path(args.pair_metadata)

    if pair_metadata_path.exists():
        with open(pair_metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["pair_index"]) == pair_index:
                    metadata_row = row
                    break

    classifier, conds_mean, conds_std = load_classifier(
        classifier_ckpt=Path(args.classifier_ckpt),
        device=device,
    )

    idx_young = CLS_TO_ID["Young"]
    idx_smiling = CLS_TO_ID["Smiling"]

    with torch.no_grad():
        z_pair_norm = (z_pair - conds_mean) / conds_std
        logits = classifier(z_pair_norm)
        probs = torch.sigmoid(logits)

    scores = []

    for i, side in enumerate(["A", "B"]):
        young_prob = float(probs[i, idx_young].item())
        smiling_prob = float(probs[i, idx_smiling].item())

        scores.append({
            "side": side,
            "Young_probability": young_prob,
            "Young_percentage": 100.0 * young_prob,
            "Young_prediction": young_prob >= args.threshold,
            "Smiling_probability": smiling_prob,
            "Smiling_percentage": 100.0 * smiling_prob,
            "Smiling_prediction": smiling_prob >= args.threshold,
        })

    print("\nClassifier results for pair", pair_index)
    print("-" * 70)

    for item in scores:
        print(
            f"Side {item['side']} | "
            f"Young: {'YES' if item['Young_prediction'] else 'NO'} "
            f"({item['Young_percentage']:.2f}%) | "
            f"Smiling: {'YES' if item['Smiling_prediction'] else 'NO'} "
            f"({item['Smiling_percentage']:.2f}%)"
        )

    print("\nLoading DiffAE autoencoder and rendering reconstructions...")

    model = load_diffae_ffhq256_autoencoder(
        diffae_root=Path(args.diffae_root),
        checkpoint_path=Path(args.autoenc_ckpt),
        device=device,
    )

    with torch.no_grad():
        decoded = model.render(xt_pair, cond=z_pair, T=args.T)

    reconstructions = [
        tensor_to_pil(decoded[0]),
        tensor_to_pil(decoded[1]),
    ]

    originals = [None, None]
    captions_original = []
    captions_recon = []

    for i, side in enumerate(["A", "B"]):
        filename = f"Pair {pair_index} side {side}"
        image_path = ""

        if metadata_row is not None:
            filename = metadata_row.get(f"filename_{side}", filename)
            image_path = metadata_row.get(f"image_path_{side}", "")

        if image_path and Path(image_path).exists():
            originals[i] = load_original_image(image_path, size=256)

        young_label = "YES" if scores[i]["Young_prediction"] else "NO"
        smiling_label = "YES" if scores[i]["Smiling_prediction"] else "NO"

        score_text = (
            f"Young: {young_label} ({scores[i]['Young_percentage']:.1f}%) | "
            f"Smiling: {smiling_label} ({scores[i]['Smiling_percentage']:.1f}%)"
        )

        captions_original.append(
            f"Original {side} | {filename} | {score_text}"
        )
        captions_recon.append(
            f"Decoded {side} from stored z_sem + x_T | {score_text}"
        )

    make_grid(
        originals=originals,
        reconstructions=reconstructions,
        captions_original=captions_original,
        captions_recon=captions_recon,
        out_path=args.out_grid,
    )

    output = {
        "pair_index": pair_index,
        "threshold": args.threshold,
        "T": args.T,
        "scores": scores,
        "metadata": metadata_row,
        "grid_path": str(Path(args.out_grid).resolve()),
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved grid to: {args.out_grid}")
    print(f"Saved JSON scores to: {args.out_json}")


if __name__ == "__main__":
    main()

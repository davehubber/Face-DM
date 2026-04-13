import argparse
import csv
import os

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid


def load_diffae_autoencoder(checkpoint_path: str, device: torch.device):
    from templates import ffhq256_autoenc
    from config import PretrainConfig
    from experiment import LitModel

    conf = ffhq256_autoenc()
    conf.pretrain = PretrainConfig(name="ffhq256_autoenc", path=checkpoint_path)
    conf.latent_infer_path = None

    model = LitModel(conf).to(device)
    model.eval()
    model.ema_model.eval()
    return model


@torch.no_grad()
def decode_saved_pair(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DiffAE Autoencoder from {args.diffae_checkpoint}...")
    diffae = load_diffae_autoencoder(args.diffae_checkpoint, device)

    # 1. Locate and load the saved embeddings
    base_dir = os.path.join("experiments", args.run_name)
    data_path = os.path.join(base_dir, "results", "decode_pair_data.pt")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find saved pair data at {data_path}. Ensure training has completed evaluating at least once.")
    
    result_item = torch.load(data_path, map_location="cpu")
    print(f"Successfully loaded embeddings for run: {args.run_name}")

    # 2. Prepare images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def load_image(path: str):
        pil = Image.open(path).convert("RGB")
        tensor = transform(pil).unsqueeze(0).to(device)
        return pil, tensor

    dominant_pil, dominant_tensor = load_image(result_item["dominant_source_path"])
    recessive_pil, recessive_tensor = load_image(result_item["recessive_source_path"])

    # 3. DiffAE Inversion
    print(f"Stochastic encoding images for {args.decode_t_invert} timesteps...")
    z_dom = diffae.encode(dominant_tensor)
    z_rec = diffae.encode(recessive_tensor)
    xT_dom = diffae.encode_stochastic(dominant_tensor, z_dom, T=args.decode_t_invert)
    xT_rec = diffae.encode_stochastic(recessive_tensor, z_rec, T=args.decode_t_invert)

    pred_dom = result_item["predicted_dominant"].unsqueeze(0).to(device)
    pred_rec = result_item["predicted_recessive"].unsqueeze(0).to(device)

    # 4. DiffAE Rendering
    print(f"Rendering generation over {args.decode_t_decode} timesteps...")
    decoded_dom = diffae.render(xT_dom, cond=pred_dom, T=args.decode_t_decode).detach().cpu().clamp(0, 1)
    decoded_rec = diffae.render(xT_rec, cond=pred_rec, T=args.decode_t_decode).detach().cpu().clamp(0, 1)

    original_dom = ((dominant_tensor.detach().cpu() + 1.0) / 2.0).clamp(0, 1)
    original_rec = ((recessive_tensor.detach().cpu() + 1.0) / 2.0).clamp(0, 1)

    # 5. Build and Save Grid
    grid = make_grid(
        torch.cat([original_dom, original_rec, decoded_dom, decoded_rec], dim=0),
        nrow=4,
        padding=8,
        pad_value=1.0,
    )

    os.makedirs(os.path.join(base_dir, "samples", "decode"), exist_ok=True)
    out_path = os.path.join(base_dir, "samples", "decode", "single_pair_decode.png")
    transforms.ToPILImage()(grid).save(out_path)

    # 6. Build and Save CSV Meta
    caption_path = os.path.join(base_dir, "samples", "decode", "single_pair_decode.csv")
    with open(caption_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "description", "sample_id", "source_path"])
        writer.writerow([0, "original dominant image", result_item["dominant_sample_id"], result_item["dominant_source_path"]])
        writer.writerow([1, "original recessive image", result_item["recessive_sample_id"], result_item["recessive_source_path"]])
        writer.writerow([2, "decoded predicted dominant embedding", result_item["dominant_sample_id"], result_item["dominant_source_path"]])
        writer.writerow([3, "decoded mathematically extracted recessive embedding", result_item["recessive_sample_id"], result_item["recessive_source_path"]])

    print(f"\nDecoding complete! Files saved to:")
    print(f" - Image Grid: {out_path}")
    print(f" - Metadata:   {caption_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode saved latent predictions back into images using DiffAE.")
    parser.add_argument("--run_name", required=True, help="Name of the experiment folder to load predictions from")
    parser.add_argument("--diffae_checkpoint", default="ffhq256_autoenc/last.ckpt", help="FFHQ256 DiffAE autoencoder checkpoint")
    parser.add_argument("--decode_t_invert", default=100, type=int, help="DDIM inversion steps for decoding")
    parser.add_argument("--decode_t_decode", default=100, type=int, help="Render steps for decoding")
    
    args = parser.parse_args()
    decode_saved_pair(args)
import argparse
import math
import os
from pathlib import Path

import torch
from diffusers import UNet2DModel
from PIL import Image, ImageOps

from utils import get_data


class ColdDiffusion:
    def __init__(self, max_timesteps=300, alpha_max=0.5, device="cuda"):
        self.max_timesteps = max_timesteps
        self.device = device
        self.alteration_per_t = alpha_max / max_timesteps

    def mix_images(self, bright_image, dark_image, t):
        weight = (self.alteration_per_t * t)[:, None, None, None]
        return bright_image * torch.sqrt(1.0 - weight) + dark_image * torch.sqrt(weight)


def get_unet(image_size):
    resolution = image_size[0] if isinstance(image_size, tuple) else image_size
    return UNet2DModel(
        sample_size=resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


def tensor_to_bordered_pil(image_tensor, border_px=1):
    """
    Converts one image tensor in [-1, 1] to a PIL image and adds a black border.

    Accepts either [C, H, W] or [1, C, H, W].
    The tensor is clamped before conversion, matching the usual visualization logic.
    """
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]

    image_tensor = image_tensor.detach().cpu().float().clamp(-1.0, 1.0)
    image_uint8 = ((image_tensor + 1.0) * 127.5).round().byte()
    image_uint8 = image_uint8.permute(1, 2, 0).numpy()

    image = Image.fromarray(image_uint8)
    return ImageOps.expand(image, border=border_px, fill=(0, 0, 0))


def save_tensor_image(image_tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_bordered_pil(image_tensor, border_px=1).save(path)


def load_state_dict(path, device):
    """
    Loads a checkpoint while remaining compatible with older and newer PyTorch versions.
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def sample_and_capture_xt(model, diffusion, mixed_image, alpha_init, capture_timesteps):
    """
    Reproduces ColdDiffusion.sample(), but stores the x_t tensor seen by the model
    before the denoising update at selected timesteps.
    """
    n = len(mixed_image)
    init_timestep = math.ceil(alpha_init / diffusion.alteration_per_t)

    x_t = mixed_image.to(diffusion.device)
    captured_xt = {}

    for i in reversed(range(1, init_timestep + 1)):
        if i in capture_timesteps:
            captured_xt[i] = x_t.detach().cpu()

        t = torch.full((n,), i, device=diffusion.device, dtype=torch.long)

        model_out = model(x_t, t).sample
        predicted_bright = model_out

        predicted_dark = (
            mixed_image - math.sqrt(1.0 - alpha_init) * predicted_bright
        ) / math.sqrt(alpha_init)

        # Same clamping logic as the training script.
        predicted_bright = torch.clamp(predicted_bright, -1.0, 1.0)
        predicted_dark = torch.clamp(predicted_dark, -1.0, 1.0)

        x_t = x_t - diffusion.mix_images(predicted_bright, predicted_dark, t)
        x_t = x_t + diffusion.mix_images(predicted_bright, predicted_dark, t - 1)

    final_bright = x_t
    final_dark = (
        mixed_image - math.sqrt(1.0 - alpha_init) * final_bright
    ) / math.sqrt(alpha_init)

    return final_bright.detach().cpu(), final_dark.detach().cpu(), captured_xt


def main(args):
    if args.max_timesteps < 300:
        raise ValueError(
            "This script saves forward degradation images at t=300. "
            "Please use --max_timesteps 300 or higher."
        )

    device = torch.device(args.device)
    args.image_size = (args.image_size, args.image_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This mirrors eval(): use the validation dataloader and take the first sample.
    val_dataloader = get_data(args, "val")
    bright_batch, dark_batch = next(iter(val_dataloader))

    bright_image = bright_batch[:1].to(device)
    dark_image = dark_batch[:1].to(device)

    model = get_unet(args.image_size).to(device)
    state_dict = load_state_dict(args.checkpoint, device)
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = ColdDiffusion(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        device=device,
    )

    # Same initial mixture used in eval().
    mixed_image = (
        bright_image * math.sqrt(1.0 - args.alpha_init)
        + dark_image * math.sqrt(args.alpha_init)
    )

    # 1-2: Ground-truth images.
    save_tensor_image(bright_image, output_dir / "01_ground_truth_bright.png")
    save_tensor_image(dark_image, output_dir / "02_ground_truth_dark.png")

    # 3-6: Forward degradation images between bright and dark.
    forward_timesteps = [1, 100, 200, 300]
    for index, timestep in enumerate(forward_timesteps, start=3):
        t = torch.full((1,), timestep, device=device, dtype=torch.long)
        degraded = diffusion.mix_images(bright_image, dark_image, t)

        suffix = "_final" if timestep == args.max_timesteps else ""
        save_tensor_image(
            degraded,
            output_dir / f"{index:02d}_forward_degradation_t{timestep:03d}{suffix}.png",
        )

    # 7-9: x_t tensors seen by the model during iterative sampling.
    capture_timesteps = {200, 100, 1}
    final_bright, final_dark, captured_xt = sample_and_capture_xt(
        model=model,
        diffusion=diffusion,
        mixed_image=mixed_image,
        alpha_init=args.alpha_init,
        capture_timesteps=capture_timesteps,
    )

    for index, timestep in zip([7, 8, 9], [200, 100, 1]):
        if timestep not in captured_xt:
            raise RuntimeError(
                f"Could not capture sampling x_t at timestep {timestep}. "
                "Check alpha_init, alpha_max, and max_timesteps."
            )

        save_tensor_image(
            captured_xt[timestep],
            output_dir / f"{index:02d}_sampling_xt_seen_by_model_t{timestep:03d}.png",
        )

    # 10-11: Final sampled outputs.
    save_tensor_image(final_bright, output_dir / "10_final_sampled_bright.png")
    save_tensor_image(final_dark, output_dir / "11_final_sampled_dark.png")

    print(f"Saved 11 bordered images to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Save the first validation pair used in the eval grid as separate images, "
            "including forward degradation states, sampling x_t states, and final outputs."
        )
    )

    parser.add_argument("--dataset_path", required=True, help="Path to the image dataset")
    parser.add_argument("--run_name", required=True, help="Name of the experiment folder")

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to the model checkpoint. "
            "Defaults to experiments/<run_name>/checkpoints/unet_ema.pt"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Folder where the separate images will be saved. "
            "Defaults to experiments/<run_name>/samples/eval_first_row_separate"
        ),
    )

    # Arguments kept compatible with get_data(args, 'val').
    parser.add_argument("--test_images", default=1000, type=int)
    parser.add_argument("--train_samples_per_epoch", default=50000, type=int)
    parser.add_argument("--num_workers", default=None, type=int)

    # Must match the training/evaluation configuration.
    parser.add_argument("--alpha_max", default=0.5, type=float)
    parser.add_argument("--alpha_init", default=0.5, type=float)
    parser.add_argument("--max_timesteps", default=300, type=int)
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--batch_size", default=1, type=int)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )

    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = os.path.join(
            "experiments", args.run_name, "checkpoints", "unet_ema.pt"
        )

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "experiments", args.run_name, "samples", "eval_first_row_separate"
        )

    return args


if __name__ == "__main__":
    main(parse_args())

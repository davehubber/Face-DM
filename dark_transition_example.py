import argparse
import math
import os
from pathlib import Path
import torch
from PIL import Image

# Import existing components directly from your training pipeline
from train import get_unet, ColdDiffusion
from utils import get_data, to_uint8


@torch.no_grad()
def capture_sampling_trajectory(model, diffusion, mixed_image, alpha_init=0.5, stride=10):
    """
    Runs the iterative cold diffusion inverse process on a single mixed image 
    and collects intermediate states at regular intervals.
    """
    model.eval()
    n = mixed_image.shape[0]
    init_timestep = math.ceil(alpha_init / diffusion.alteration_per_t)
    device = diffusion.device

    x_t = mixed_image.to(device)
    
    # Storage for trajectory frames: tuples of (timestep, x_t, pred_bright, pred_dark)
    trajectory_records = []

    print(f"Starting trajectory tracking from step {init_timestep} down to 1...")
    
    for i in reversed(range(1, init_timestep + 1)):
        t = torch.full((n,), i, device=device, dtype=torch.long)

        # Forward pass through UNet (predicts 6 channels: bright, dark)
        model_out = model(x_t, t).sample
        predicted_dark = model_out[:, 3:]
        predicted_bright = (mixed_image - math.sqrt(1.0 - alpha_init) * predicted_dark) / math.sqrt(alpha_init)

        # Enforce physical pixel bounds symmetrically
        predicted_dark = torch.clamp(predicted_dark, -1.0, 1.0)
        predicted_bright = torch.clamp(predicted_bright, -1.0, 1.0)

        # Record frames based on stride, or at critical boundary steps
        if i == init_timestep or i == 1 or (i % stride == 0):
            trajectory_records.append({
                "step": i,
                "x_t": to_uint8(x_t.clone())[0],
                "pred_bright": to_uint8(predicted_bright.clone())[0],
                "pred_dark": to_uint8(predicted_dark.clone())[0]
            })

        # Update transition state equation
        x_t = x_t - diffusion.mix_images(predicted_dark, predicted_bright, t) + diffusion.mix_images(
            predicted_dark, predicted_bright, t - 1
        )

    # Final mathematical alignment block extraction at step 0
    predicted_bright_final = (mixed_image - math.sqrt(1.0 - alpha_init) * x_t) / math.sqrt(alpha_init)
    trajectory_records.append({
        "step": 0,
        "x_t": to_uint8(x_t.clone())[0],
        "pred_bright": to_uint8(predicted_bright_final)[0],
        "pred_dark": to_uint8(x_t)[0]
    })

    return trajectory_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/nas-ctm01/datasets/public/ffhq256/", help="Path to validation dataset")
    parser.add_argument("--run_name", type=str, default="faces_dual_iso", help="Experiment folder containing the checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="unet_ema_best.pt", help="Target checkpoint file name")
    parser.add_argument("--image_size", type=int, default=256, help="Resolution matching training setup")
    parser.add_argument("--max_timesteps", type=int, default=300, help="Total diffusion timesteps setup")
    parser.add_argument("--alpha_max", type=float, default=0.5)
    parser.add_argument("--alpha_init", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=15, help="Step interval saving frequency for the grid rows")
    
    # Dummy parameters to satisfy utility get_data validation requirements
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_samples_per_epoch", type=int, default=1)
    parser.add_argument("--test_images", type=int, default=1000)

    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path("experiments") / args.run_name
    checkpoint_path = base_dir / "checkpoints" / args.checkpoint_name
    output_grid_path = base_dir / "samples" / "sampling_path_trajectory.png"
    output_grid_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Model and Weights
    print(f"Loading UNet architecture using checkpoint: {checkpoint_path}")
    model = get_unet(args.image_size).to(device)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Target checkpoint weights not found at: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 2. Fetch a Random Sample from Validation Stream
    print("Sampling a validation test image pair...")
    val_dataloader = get_data(args, "val")
    bright_img, dark_img = next(iter(val_dataloader))
    
    # Isolate first single index sample pair from the mini-batch array
    bright_img = bright_img[0:1].to(device)
    dark_img = dark_img[0:1].to(device)

    # 3. Initialize Cold Diffusion Instance
    diffusion = ColdDiffusion(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        device=device,
    )

    # Form the initial blended composite mixture
    mixed_image = bright_img * math.sqrt(1.0 - args.alpha_init) + dark_img * math.sqrt(args.alpha_init)

    # 4. Extract Trajectory History
    trajectory = capture_sampling_trajectory(
        model=model,
        diffusion=diffusion,
        mixed_image=mixed_image,
        alpha_init=args.alpha_init,
        stride=args.stride
    )

    # 5. Build and Stitch the Visual Trajectory Grid
    # Columns layout: [ Moving Blend State (x_t) | Predicted Bright | Predicted Dark ]
    num_rows = len(trajectory)
    img_h, img_w = args.image_size[0], args.image_size[1]
    
    grid_canvas = Image.new("RGB", (img_w * 3, img_h * num_rows))

    print(f"Assembling grid visualization ({num_rows} interval steps)...")
    for row_idx, frame in enumerate(trajectory):
        # Convert PyTorch CHW uint8 tensors back to standard PIL images for pasting
        img_xt = Image.fromarray(frame["x_t"].permute(1, 2, 0).cpu().numpy())
        img_pb = Image.fromarray(frame["pred_bright"].permute(1, 2, 0).cpu().numpy())
        img_pd = Image.fromarray(frame["pred_dark"].permute(1, 2, 0).cpu().numpy())

        y_offset = row_idx * img_h
        grid_canvas.paste(img_xt, (0, y_offset))
        grid_canvas.paste(img_pb, (img_w, y_offset))
        grid_canvas.paste(img_pd, (img_w * 2, y_offset))
        
        print(f"  -> Pasted row {row_idx + 1}/{num_rows} (Diffusion Step: {frame['step']})")

    grid_canvas.save(output_grid_path)
    print(f"\n[SUCCESS] Sampling path trajectory grid saved to: {output_grid_path}")


if __name__ == "__main__":
    main()
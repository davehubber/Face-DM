import argparse
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        default="experiments/faces_dual_iso/samples/sampling_path_trajectory.png",
        help="Path to the already generated trajectory grid image",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="experiments/faces_dual_iso/samples/sampling_path_trajectory_edited.png",
        help="Output path for the cropped/edited grid image",
    )
    parser.add_argument(
        "--row-height",
        type=int,
        default=256,
        help="The pixel height of a single row/image in your pipeline (default: 256)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    out_path = Path(args.out_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find the generated grid at: {input_path}\n"
            f"Make sure the path is correct or pass it via --input-path"
        )

    # 1. Load the original stitched grid
    grid_img = Image.open(input_path).convert("RGB")
    W, H = grid_img.size
    img_h = args.row_height

    # 2. Compute total rows currently inside the image
    num_rows = H // img_h
    if num_rows < 3:
        print(f"Error: The image only has {num_rows} row(s). Cannot extract first, middle, and second-to-last.")
        return

    # 3. Determine the vertical pixel boundaries for our targets
    first_idx = 0
    mid_idx = num_rows // 2
    prev_last_idx = num_rows - 2

    print(f"Opened grid image ({W}x{H} px). Detected {num_rows} total rows.")
    print(f"Extracting rows at indices: First ({first_idx}), Middle ({mid_idx}), Before Last ({prev_last_idx})")

    # Crop out the full width horizontal strips for each target row
    row_first = grid_img.crop((0, first_idx * img_h, W, (first_idx + 1) * img_h))
    row_mid = grid_img.crop((0, mid_idx * img_h, W, (mid_idx + 1) * img_h))
    row_prev_last = grid_img.crop((0, prev_last_idx * img_h, W, (prev_last_idx + 1) * img_h))

    # 4. Create a new canvas to stack the 3 rows vertically
    # Keeps the exact same width (3 columns wide) but changes the total height to 3 rows
    new_canvas = Image.new("RGB", (W, img_h * 3))
    
    new_canvas.paste(row_first, (0, 0))
    new_canvas.paste(row_mid, (0, img_h))
    new_canvas.paste(row_prev_last, (0, img_h * 2))

    # 5. Export result
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_canvas.save(out_path)
    print(f"\n[SUCCESS] Slice complete. Edited grid saved to: {out_path}")


if __name__ == "__main__":
    main()
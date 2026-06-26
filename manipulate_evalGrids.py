import argparse
import os
from PIL import Image


def manipulate_grid_range(image_path, img_size, start_row_idx, end_row_idx, target_col_idx, padding, output_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at: {image_path}")
        
    grid_img = Image.open(image_path).convert("RGB")
    grid_w, grid_h = grid_img.size
    
    # Calculate grid dimensions accounting for outer border padding
    num_cols = (grid_w - padding) // (img_size + padding)
    num_rows = (grid_h - padding) // (img_size + padding)
    
    # Validation checks
    if start_row_idx < 0 or start_row_idx >= num_rows:
        raise ValueError(f"Start row index {start_row_idx} is out of bounds for a grid with {num_rows} rows.")
    if end_row_idx < 0 or end_row_idx >= num_rows:
        raise ValueError(f"End row index {end_row_idx} is out of bounds for a grid with {num_rows} rows.")
    if start_row_idx > end_row_idx:
        raise ValueError(f"Start row index ({start_row_idx}) cannot be greater than end row index ({end_row_idx}).")
    if target_col_idx >= num_cols or target_col_idx < 0:
        raise ValueError(f"Column index {target_col_idx} is out of bounds for a grid with {num_cols} columns.")
        
    num_rows_to_keep = end_row_idx - start_row_idx + 1
    
    # Define compact, padding-free target dimensions
    new_grid_w = num_cols * img_size
    new_grid_h = num_rows_to_keep * img_size
    new_grid_img = Image.new("RGB", (new_grid_w, new_grid_h))
    
    # Map input rows to continuous output rows
    for out_r, in_r in enumerate(range(start_row_idx, end_row_idx + 1)):
        row_images = []
        
        for c in range(num_cols):
            # Crop from the padded source using the source row index (in_r)
            left = padding + c * (img_size + padding)
            top = padding + in_r * (img_size + padding)
            right = left + img_size
            bottom = top + img_size
            
            sub_img = grid_img.crop((left, top, right, bottom))
            row_images.append(sub_img)
        
        # Pull target column to index 0
        target_img = row_images[target_col_idx]
        remaining_images = row_images[:target_col_idx] + row_images[target_col_idx + 1:]
        reordered_row = [target_img] + remaining_images
        
        # Paste into target canvas using the sequential output row index (out_r)
        for c, img in enumerate(reordered_row):
            new_left = c * img_size
            new_top = out_r * img_size
            new_grid_img.paste(img, (new_left, new_top))
            
    new_grid_img.save(output_path)
    print(f"Success: Sliced and compact grid saved to '{output_path}'")
    print(f"Extracted: Rows {start_row_idx} to {end_row_idx} ({num_rows_to_keep} rows total) | Reordered column {target_col_idx} to front.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a specific row range from a padded grid and rearrange columns.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input grid image file")
    parser.add_argument("--size", type=int, required=True, help="Dimension of the square sub-images")
    parser.add_argument("--start-row", type=int, default=0, help="Starting row index to keep (0-indexed, inclusive)")
    parser.add_argument("--end-row", type=int, required=True, help="Ending row index to keep (0-indexed, inclusive)")
    parser.add_argument("--col", type=int, required=True, help="Column index to shift to the front (0-indexed)")
    parser.add_argument("--padding", type=int, default=2, help="Padding value used during grid generation")
    parser.add_argument("--output", type=str, default="sliced_compact_grid.png", help="Path to save the resulting image")
    
    args = parser.parse_args()
    
    manipulate_grid_range(
        image_path=args.image,
        img_size=args.size,
        start_row_idx=args.start_row,
        end_row_idx=args.end_row,
        target_col_idx=args.col,
        padding=args.padding,
        output_path=args.output
    )
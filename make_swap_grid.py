import os
from PIL import Image

# --- CONFIGURATION ---
swaps_dir = "experiments/flowers_baseline/samples/severe_swaps"
output_path = "stacked_swaps_swapped_cols.jpg"

swap_files = [
    "rank_1_severity_7768.jpg",
    "rank_2_severity_7250.jpg",
    "rank_3_severity_6948.jpg",
    "rank_4_severity_6614.jpg",
    "rank_5_severity_6577.jpg"
]

IMG_SIZE = 64  # Size of each individual square image inside the grids
PADDING = 2   # PyTorch make_grid default outer/inner padding
# ---------------------


def remove_grid_padding_and_swap(img, img_size, padding):
    """Extracts sub-images from a padded grid, swaps the 3rd and 4th columns, 

    and returns a completely flush image.
    """
    num_cols = (img.width - padding) // (img_size + padding)
    num_rows = (img.height - padding) // (img_size + padding)
    
    # Create a new canvas with zero padding
    flush_w = num_cols * img_size
    flush_h = num_rows * img_size
    flush_img = Image.new("RGB", (flush_w, flush_h))
    
    for r in range(num_rows):
        row_cells = []
        
        # 1. Crop all cells in the current row
        for c in range(num_cols):
            left = padding + c * (img_size + padding)
            top = padding + r * (img_size + padding)
            right = left + img_size
            bottom = top + img_size
            
            row_cells.append(img.crop((left, top, right, bottom)))
        
        # 2. Perform the column swap (3rd column is index 2, 4th column is index 3)
        if len(row_cells) > 3:
            row_cells[2], row_cells[3] = row_cells[3], row_cells[2]
        
        # 3. Paste them sequentially without padding
        for c, cell in enumerate(row_cells):
            flush_img.paste(cell, (c * img_size, r * img_size))
            
    return flush_img


def stack_swaps_vertically(directory, files, img_size, padding, output_path):
    processed_images = []
    
    # Load, un-pad, and swap columns for each grid image
    for file in files:
        full_path = os.path.join(directory, file)
        if os.path.exists(full_path):
            raw_grid = Image.open(full_path).convert("RGB")
            clean_grid = remove_grid_padding_and_swap(raw_grid, img_size, padding)
            processed_images.append(clean_grid)
        else:
            print(f"Error: File not found at {full_path}")
            return

    if not processed_images:
        return

    base_width = processed_images[0].width
    total_height = sum(img.height for img in processed_images)
    
    # Create the final vertical stack canvas
    stacked_image = Image.new("RGB", (base_width, total_height))
    
    # Stack the modified grids sequentially
    current_y = 0
    for img in processed_images:
        if img.width != base_width:
            scale_factor = base_width / img.width
            new_height = int(img.height * scale_factor)
            img = img.resize((base_width, new_height), Image.Resampling.LANCZOS)
        
        stacked_image.paste(img, (0, current_y))
        current_y += img.height
    
    # Export result
    stacked_image.save(output_path, quality=95)
    print(f"Success! Vertically stacked grid with swapped columns saved to: {output_path}")


if __name__ == "__main__":
    stack_swaps_vertically(
        directory=swaps_dir, 
        files=swap_files, 
        img_size=IMG_SIZE, 
        padding=PADDING, 
        output_path=output_path
    )
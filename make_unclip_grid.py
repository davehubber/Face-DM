import os
from PIL import Image

# Define paths to your two grid images
img_faces_path = "unclip_embeddings_faces/comparacao_unclip_flat_batched.jpg"
img_flowers_path = "unclip_embeddings_flowers/comparacao_unclip_flat_batched.jpg"

# Configuration
ORIG_SUB_SIZE = 256    # Original size of each face/flower image
TARGET_SUB_SIZE = 64   # Desired final size of each face/flower image
ROWS_TO_TAKE = 5       # Number of rows to copy from each grid

def combine_grids_horizontally(path1, path2, output_path="combined_grid_horizontal.jpg"):
    # Open both grid images
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    
    # Calculate the pixel height of the first 4 rows (4 * 256 = 1024 pixels)
    crop_height = ROWS_TO_TAKE * ORIG_SUB_SIZE
    
    # Crop the first 4 rows from both grids
    grid1_cropped = img1.crop((0, 0, img1.width, crop_height))
    grid2_cropped = img2.crop((0, 0, img2.width, crop_height))
    
    # Calculate new scaled dimensions (scaled down by 75%)
    scale_factor = TARGET_SUB_SIZE / ORIG_SUB_SIZE
    new_width1 = int(img1.width * scale_factor)
    new_width2 = int(img2.width * scale_factor)
    new_height = int(crop_height * scale_factor)
    
    # Resize the cropped blocks using high-quality downscaling
    grid1_resized = grid1_cropped.resize((new_width1, new_height), Image.Resampling.LANCZOS)
    grid2_resized = grid2_cropped.resize((new_width2, new_height), Image.Resampling.LANCZOS)
    
    # Create a new blank canvas to stick them side-by-side
    # Total width is the sum of both; total height matches the cropped height
    total_width = new_width1 + new_width2
    final_image = Image.new("RGB", (total_width, new_height))
    
    # Paste the resized grids side-by-side into the canvas
    final_image.paste(grid1_resized, (0, 0))            # Left side
    final_image.paste(grid2_resized, (new_width1, 0))    # Right side
    
    # Save the output image
    final_image.save(output_path, quality=95)
    print(f"Success! Horizontally combined image saved to: {output_path}")

# Run the script
if __name__ == "__main__":
    combine_grids_horizontally(img_faces_path, img_flowers_path)
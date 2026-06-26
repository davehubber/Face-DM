import os
from PIL import Image

# 1. Define the directory and image names based on your folder structure
folder_path = "pca_analysis"
image_files = [
    "pc1_traversal_grid.png",
    "pc2_traversal_grid.png",
    "pc3_traversal_grid.png"
]

# Create full file paths
image_paths = [os.path.join(folder_path, filename) for filename in image_files]

# 2. Open all images
images = [Image.open(path) for path in image_paths]

# 3. Calculate dimensions for the stitched canvas
# Uses the maximum width among images and sums up their heights
max_width = max(img.width for img in images)
total_height = sum(img.height for img in images)

# Create a new blank canvas image (supports transparency via "RGBA")
stitched_image = Image.new("RGBA", (max_width, total_height))

# 4. Paste images sequentially from top to bottom
current_y = 0
for img in images:
    stitched_image.paste(img, (0, current_y))
    current_y += img.height

# 5. Calculate 1/4 dimensions
target_width = max_width // 4
target_height = total_height // 4

# Resize using high-quality Lanczos resampling
final_image = stitched_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

# 6. Save the final output
output_path = os.path.join(folder_path, "pca_traversal_grid.png")
final_image.save(output_path)

print(f"Process complete! Saved to: {output_path}")
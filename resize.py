import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

# Common image extensions to look for
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

def resize_single_image(args):
    """Worker function executed by individual processes."""
    file_path, output_dir = args
    try:
        with Image.open(file_path) as img:
            # LANCZOS is high quality; if speed is still an issue, 
            # you can switch to Image.Resampling.BILINEAR
            resized_img = img.resize((64, 64), Image.Resampling.LANCZOS)
            
            target_path = output_dir / file_path.name
            resized_img.save(target_path)
        return True, file_path.name
    except Exception as e:
        return False, f"{file_path.name} (Error: {e})"

def resize_images_parallel(source_dir_path):
    source_dir = Path(source_dir_path).resolve()
    
    if not source_dir.exists() or not source_dir.is_dir():
        print(f"Error: The directory '{source_dir}' does not exist.")
        return

    # Create the new directory path at the same level
    output_dir = source_dir.parent / f"{source_dir.name}-64x64"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather all valid image paths
    image_tasks = [
        (file_path, output_dir) 
        for file_path in source_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS
    ]
    
    total_images = len(image_tasks)
    if total_images == 0:
        print("No valid images found to resize.")
        return

    print(f"Found {total_images} images. Starting parallel processing...")
    start_time = time.time()

    success_count = 0
    # ProcessPoolExecutor automatically defaults to the number of CPU cores you have
    with ProcessPoolExecutor() as executor:
        # Map the worker function across all image tasks
        results = executor.map(resize_single_image, image_tasks)
        
        for success, message in results:
            if success:
                success_count += 1
                # Optional: Print progress every 100 images to avoid terminal slowdown
                if success_count % 100 == 0 or success_count == total_images:
                    print(f"Progress: {success_count}/{total_images} processed...")
            else:
                print(f"Skipped: {message}")

    end_time = time.time()
    print(f"\nDone! Successfully resized {success_count}/{total_images} images.")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")

# --- RUN THE SCRIPT ---
if __name__ == "__main__":
    # Replace this with the actual path to your image folder
    folder_to_process = "/nas-ctm01/datasets/public/BIOMETRICS/CELEBA/celebamask-hq-db/CelebA-HQ-img/" 
    
    resize_images_parallel(folder_to_process)

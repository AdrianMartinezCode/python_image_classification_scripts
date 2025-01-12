import os
import shutil
from PIL import Image
import argparse

def move_large_images(source_folder, destination_folder):
    """
    Traverses through all folders in the source directory,
    moves images to the destination folder if width or height > 800,
    while preserving the folder structure.

    :param source_folder: Path to the source directory
    :param destination_folder: Path to the destination directory
    """
    for root, _, files in os.walk(source_folder):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    if width > 800 or height > 800:
                        # Compute the relative path from the source folder
                        relative_path = os.path.relpath(root, source_folder)
                        # Create the corresponding folder in the destination
                        target_folder = os.path.join(destination_folder, relative_path)
                        if not os.path.exists(target_folder):
                            os.makedirs(target_folder)
                        # Move the file to the target folder
                        destination_path = os.path.join(target_folder, file)
                        shutil.move(file_path, destination_path)
                        print(f"Moved: {file_path} -> {destination_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move images with width or height > 800 while preserving folder structure.")
    parser.add_argument("source_folder", type=str, help="Path to the source folder")
    parser.add_argument("destination_folder", type=str, help="Path to the destination folder")
    args = parser.parse_args()

    move_large_images(args.source_folder, args.destination_folder)

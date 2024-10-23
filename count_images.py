import os

def count_images_in_folder(folder_path):
    # Define valid image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    # Initialize a counter
    image_count = 0
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Get the file extension
        ext = os.path.splitext(filename)[1].lower()
        
        # Check if the file has a valid image extension
        if ext in valid_extensions:
            image_count += 1

    return image_count

if __name__ == "__main__":
    # Provide the folder path
    folder_path = "/usr/xtmp/mxy/VLM-Poisoning/data/clean_data/cc_sbu_align-llava/image"  # Replace with the actual folder path

    # Count and print the number of images
    image_count = count_images_in_folder(folder_path)
    print(f"The folder '{folder_path}' contains {image_count} image(s).")

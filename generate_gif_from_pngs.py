import os
import re
import imageio

# Set your directory path
basePath = "/Users/gracecalkins/Local_Documents/local_code/pipag_training/data/"

LDs = [3]
NCs = [3]

for LD in LDs:
    for NC in NCs:
        # Get the folder in basePath that ends with LD{LD}_NC{NC}
        folder_path = os.path.join(basePath, [folder for folder in os.listdir(basePath) if folder.endswith(f"LD{LD}_NC{NC}")][0])

        # Regular expression to extract epoch number XXXXX from filename
        pattern = re.compile(r"latent_samples_epoch_(\d+)_.*")

        # Collect files with epoch numbers
        image_files = []
        for filename in os.listdir(folder_path):
            match = pattern.match(filename)
            if match:
                epoch = int(match.group(1))  # Extract and convert XXXXX to an integer
                image_files.append((epoch, os.path.join(folder_path, filename)))

        # Sort files by epoch number
        image_files.sort()

        # Extract just the sorted file paths
        sorted_image_paths = [file[1] for file in image_files]

        # Create a GIF
        output_gif = os.path.join(folder_path, f"latent_samples_animation_LD{LD}_NC{NC}.gif")
        imageio.mimsave(output_gif, [imageio.imread(img) for img in sorted_image_paths], duration=0.25)  # Adjust duration as needed

        print(f"GIF saved as {output_gif}")

import os
import re

def remove_duplicate_epochs(root_dir):
    """
    Scans the given root directory recursively, finds files with the same epoch 
    within each separate subdirectory, and removes duplicates, keeping only one 
    file per epoch.

    Parameters:
    root_dir (str): The root directory to scan.

    Returns:
    None
    """
    # Regular expression to extract epoch number from the filename
    epoch_pattern = re.compile(r'graph_emb_epoch_(\d+)_loss_[-\d.]+\.pt')

    for subdir, _, files in os.walk(root_dir):
        # Dictionary to store the files for each epoch within the current subdir
        epoch_files = {}

        for file in files:
            match = epoch_pattern.match(file)
            if match:
                epoch = int(match.group(1))
                if epoch in epoch_files:
                    # If the epoch is already in the dictionary, remove one of the files
                    os.remove(os.path.join(subdir, file))
                    print ('delete:',subdir, file)
                else:
                    epoch_files[epoch] = file

# Example usage
remove_duplicate_epochs('experiment_results/')

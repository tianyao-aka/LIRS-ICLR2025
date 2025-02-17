import os
import pandas as pd

def process_results(file_dir, file_name="xxx.txt"):
    """
    Processes the directory structure to extract and aggregate test values.

    Parameters:
    - file_dir (str): The root directory containing subdirectories to process.
    - file_name (str): The name of the file within seed subdirectories to parse. Defaults to "xxx.txt".

    Returns:
    - pd.DataFrame: A DataFrame with grouped statistics (mean, max, std) of test values.
    """
    # Initialize list to hold data for DataFrame
    data = []

    # List all subdirectories in the given file directory
    for subdir in os.listdir(file_dir):
        subdir_path = os.path.join(file_dir, subdir)
        
        # Check if the subdir is indeed a directory
        if os.path.isdir(subdir_path):
            for seed_dir in os.listdir(subdir_path):
                seed_path = os.path.join(subdir_path, seed_dir)
                
                # Ensure the path is a directory (e.g., seed_1, seed_2, etc.)
                if os.path.isdir(seed_path):
                    file_path = os.path.join(seed_path, file_name)
                    
                    # Parse the file for the test value
                    if os.path.isfile(file_path):
                        with open(file_path, 'r') as f:
                            content = f.read()
                            # Extract test value from the line
                            try:
                                test_value = float(content.split("test:")[1].split(",")[0].strip())
                                data.append([subdir, seed_dir, test_value])
                            except (IndexError, ValueError) as e:
                                print(f"Error parsing file {file_path}: {e}")

    # Create a DataFrame from collected data
    df = pd.DataFrame(data, columns=["Subdir", "Seed", "TestValue"])
    
    # Group by "Subdir" and aggregate statistics
    grouped_df = df.groupby("Subdir")["TestValue"].agg(['mean', 'max', 'std']).reset_index()
    
    return grouped_df

# Example usage
dset = 'ogbg-molbbbp'
domain = 'size'
path = f"OODGCL_res/{dset}/{domain}/"

result_df = process_results(path, file_name="test_perf.txt")
result_df.to_csv(f"OODGCL_res/csv_files/{dset}_{domain}_results.csv", index=False)




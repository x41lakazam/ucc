import pandas as pd
import os
import glob
import re

# --- 1. Configuration: Please edit these paths and settings ---

# Define the path to the directory where your files are stored.
folder_path = '/my/inner/logs/path'

# Define the full path for the output CSV file.
output_path = '/my/output/path/combined_latency.csv'

# Define a pattern to find your files.
file_pattern = 'matrices_exec_time_bench*_iter*.log.*'

# --- 2. Data Processing ---

# Use glob to find all files in the folder that match the pattern
file_list = glob.glob(os.path.join(folder_path, file_pattern))

if not file_list:
    print(f"Error: No files found at '{folder_path}' with pattern '{file_pattern}'.")
    print("Please check your 'folder_path' and 'file_pattern'.")
else:
    print(f"Found {len(file_list)} files to process.")
    all_data = []

    for file_path in file_list:
        filename = os.path.basename(file_path)

        try:
            # 1. Extract Rank
            rank_number = int(filename.split('.')[-1])

            # 2. Extract Matrix Identifier
            match = re.search(r'matrices_exec_time_bench(\d+)_iter(\d+)', filename)
            if not match:
                print(f"Warning: Could not extract matrix ID from filename '{filename}'. Skipping this file.")
                continue
            
            # Extract both numbers from the matrix identifier
            first_num = int(match.group(1))
            second_num = int(match.group(2))
            
            # Create a unique matrix identifier by combining both numbers
            matrix_id = f"{first_num}_{second_num}"

        except (ValueError, IndexError):
            print(f"Warning: Could not parse numeric matrix/rank from filename '{filename}'. Skipping this file.")
            continue

        # --- Read the single column of latency data ---
        try:
            temp_df = pd.read_csv(file_path, header=None, names=['latency'])
            
            # Store the matrix ID as a string in the format "X_Y"
            temp_df['matrix'] = matrix_id
            temp_df['rank'] = rank_number

            all_data.append(temp_df)

        except Exception as e:
            print(f"Error reading or processing file {filename}: {e}")

# --- 3. Finalization and Saving ---

if not all_data:
    print("No data was successfully processed.")
else:
    final_df = pd.concat(all_data, ignore_index=True)

    final_df = final_df[['matrix', 'rank', 'latency']]
    
    # Sort by matrix ID (which is now a string) and rank
    final_df = final_df.sort_values(by=['matrix', 'rank']).reset_index(drop=True)

    print("\n--- Data Processing Complete ---")
    print("Total rows processed:", len(final_df))
    print(f"Total columns in final table: {len(final_df.columns)}")
    print("\nFirst 5 rows of the combined data:")
    print(final_df.head())
    print("\nLast 5 rows of the combined data:")
    print(final_df.tail())

    final_df.to_csv(output_path, index=False)

    print(f"\nSuccessfully saved all data to '{output_path}'")
import os
import pandas as pd

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_DIR = "data/songs/rawsongdata"   # folder containing the CSV files
OUTPUT_FILE = "data/songs/merged_lyrics.csv"       # name of the final merged CSV
REMOVE_DUPLICATES = False                # set to False if you want to keep all rows
PRIMARY_KEYS = ["artist", "song"]       # columns used to detect duplicates (optional)
# ---------------------------------------------------------

def merge_csvs(input_dir, output_file, remove_duplicates=True, primary_keys=None):
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    if not csv_files:
        raise RuntimeError("No CSV files found in directory.")

    print(f"Found {len(csv_files)} CSV files. Merging...")

    df_list = []
    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        print(f"Loading {file_path} ...")
        df = pd.read_csv(file_path, low_memory=False)
        df_list.append(df)

    print("Concatenating all CSVs...")
    merged_df = pd.concat(df_list, ignore_index=True)

    if remove_duplicates:
        if primary_keys:
            print(f"Removing duplicates using keys: {primary_keys}")
            merged_df.drop_duplicates(subset=primary_keys, inplace=True)
        else:
            print("Removing exact duplicate rows...")
            merged_df.drop_duplicates(inplace=True)

    print(f"Saving merged CSV to: {output_file}")
    merged_df.to_csv(output_file, index=False)
    print("Done!")

    return merged_df


# Run the merge
merged = merge_csvs(
    INPUT_DIR,
    OUTPUT_FILE,
    remove_duplicates=REMOVE_DUPLICATES,
    primary_keys=PRIMARY_KEYS
)

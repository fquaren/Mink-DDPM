import os
import glob
import yaml
import argparse
import subprocess
import shutil
from tqdm import tqdm


def main():
    """Consolidates daily metadata using disk-based Linux tools."""
    if not shutil.which("shuf") or not shutil.which("wc"):
        raise EnvironmentError(
            "This script requires 'shuf' and 'wc' to be in the system's PATH."
        )

    parser = argparse.ArgumentParser(
        description="Consolidate and split daily metadata using shuf."
    )
    parser.add_argument("config", type=str, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    METADATA_DIR = config["METADATA_DIR"]
    SPLIT_RATIOS = config["SPLIT_RATIOS"]
    DAILY_METADATA_DIR = os.path.join(METADATA_DIR, "temp_metadata")

    consolidated_file = os.path.join(METADATA_DIR, "all_patches.tmp")
    shuffled_file = os.path.join(METADATA_DIR, "all_patches_shuffled.tmp")

    print("Consolidating daily metadata files...")
    daily_files = sorted(glob.glob(os.path.join(DAILY_METADATA_DIR, "*.txt")))
    if not daily_files:
        raise FileNotFoundError(
            f"No daily metadata files found in {DAILY_METADATA_DIR}"
        )

    with open(consolidated_file, "wb") as outfile:
        for filename in tqdm(daily_files, desc="Concatenating"):
            with open(filename, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    print("Shuffling consolidated file using 'shuf'...")
    subprocess.run(["shuf", consolidated_file, "-o", shuffled_file], check=True)

    print("Counting total patches...")
    result = subprocess.run(
        ["wc", "-l", shuffled_file], capture_output=True, text=True, check=True
    )
    total_patches = int(result.stdout.split()[0])
    print(f"Found a total of {total_patches} valid patches.")

    print("Splitting shuffled file into train, validation, and test sets...")
    train_end_idx = int(total_patches * SPLIT_RATIOS["train"])
    val_end_idx = train_end_idx + int(total_patches * SPLIT_RATIOS["validation"])

    train_path = os.path.join(METADATA_DIR, "train_patches_metadata.txt")
    val_path = os.path.join(METADATA_DIR, "val_patches_metadata.txt")
    test_path = os.path.join(METADATA_DIR, "test_patches_metadata.txt")

    with open(shuffled_file, "r") as f_in, open(train_path, "w") as f_train, open(
        val_path, "w"
    ) as f_val, open(test_path, "w") as f_test:
        for i, line in tqdm(
            enumerate(f_in), total=total_patches, desc="Writing splits"
        ):
            if i < train_end_idx:
                f_train.write(line)
            elif i < val_end_idx:
                f_val.write(line)
            else:
                f_test.write(line)

    print("\nFinal train/validation/test metadata files created successfully.")
    os.remove(consolidated_file)
    os.remove(shuffled_file)
    shutil.rmtree(DAILY_METADATA_DIR)
    print("Cleaned up temporary files and directory.")


if __name__ == "__main__":
    main()

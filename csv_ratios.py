#!/usr/bin/env python3
import argparse
import pandas as pd
import sys

def main():
    parser = argparse.ArgumentParser(description="Calculate value ratios for a CSV column.")
    parser.add_argument("filepath", help="Path to the CSV file")
    parser.add_argument("column", help="Column name to analyze")
    args = parser.parse_args()

    # Load the CSV
    try:
        df = pd.read_csv(args.filepath)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Validate column
    if args.column not in df.columns:
        print(f"Error: Column '{args.column}' not found in CSV.")
        print("Available columns:", ", ".join(df.columns))
        sys.exit(1)

    # Get value counts + ratios
    counts = df[args.column].value_counts(dropna=False)
    ratios = counts / counts.sum()

    print(f"\nValue ratios for column: {args.column}\n")
    for value, ratio in ratios.items():
        print(f"{repr(value)}: {ratio:.4f} ({ratio*100:.2f}%)")

if __name__ == "__main__":
    main()

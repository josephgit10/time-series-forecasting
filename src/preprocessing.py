import os
import pandas as pd

def load_and_merge():
    sales_path    = "/app/data/raw/sales_data.csv"
    features_path = "/app/data/raw/Features_data.csv"
    stores_path   = "/app/data/raw/stores_data.csv"

    # Parse Date with dayfirst=True
    sales_df = pd.read_csv(
        sales_path,
        parse_dates=['Date'],
        dayfirst=True
    )
    features_df = pd.read_csv(
        features_path,
        parse_dates=['Date'],
        dayfirst=True
    )
    stores_df = pd.read_csv(stores_path)

    # Merge
    df = sales_df.merge(features_df, on=["Store", "Date"], how="left")
    df = df.merge(stores_df, on="Store", how="left")

    # Ensure output folder exists
    processed_dir = "/app/data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Write merged result
    df.to_csv(os.path.join(processed_dir, "merged_data.csv"), index=False)

if __name__ == "__main__":
    load_and_merge()
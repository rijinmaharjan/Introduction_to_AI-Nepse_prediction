import pandas as pd

# Change this to the dataset you want to inspect
FILE_PATH = "Dataset_Different_Sector/ACLBSL_2000-01-01_2021-12-31.csv"

def inspect_dataset(file_path):
    df = pd.read_csv(file_path)

    print("\n" + "="*70)
    print("NEPSE DATASET INSPECTION")
    print("="*70)

    # Shape
    print("\n1️⃣ Dataset Shape (Rows, Columns):")
    print(df.shape)

    # Columns
    print("\n2️⃣ Column Names:")
    for col in df.columns:
        print("-", col)

    # First 5 rows
    print("\n3️⃣ First 5 Rows:")
    print(df.head())

    # Data types
    print("\n4️⃣ Data Types:")
    print(df.dtypes)

    # Missing values
    print("\n5️⃣ Missing Values Per Column:")
    print(df.isnull().sum())

    # Statistical summary
    print("\n6️⃣ Statistical Summary:")
    print(df.describe())

    print("\n" + "="*70)


if __name__ == "__main__":
    inspect_dataset(FILE_PATH)
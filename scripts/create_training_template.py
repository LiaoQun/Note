"""
This script generates a CSV template for BDE annotation from a list of SMILES,
based on the faithful port of `alfabet` fragmentation logic.
"""
import os
import pandas as pd
from src.data_preparation.template_generator import generate_fragment_template

# 1. 在這裡輸入您想要產生模板的分子 SMILES 列表
smiles_to_process = [
    "CC",       # Ethane
    "CCO",      # Ethanol
    "c1ccccc1", # Benzene (will be skipped as it only contains ring bonds)
    "C(C)(C)C", # Isobutane
]

# 2. 指定您想要儲存的 CSV 檔案路徑
output_dir = "data"
output_filename = "fragment_template_advanced.csv"
output_csv_path = os.path.join(output_dir, output_filename)

# 確保輸出目錄存在
os.makedirs(output_dir, exist_ok=True)

# 3. 執行生成
print(f"Generating fragment template for {len(smiles_to_process)} molecules...")
fragment_template_df = generate_fragment_template(smiles_to_process)

# 4. 儲存為 CSV
fragment_template_df.to_csv(output_csv_path, index=False)

print(f"\nTemplate successfully saved to: {output_csv_path}")
print("\nDataFrame Head:")
print(fragment_template_df.head())

print(f"\nTo use this template:")
print(f"1. Open '{output_csv_path}' in a spreadsheet editor.")
print(f"2. The file lists all non-ring single bonds and their resulting radical fragments.")
print(f"3. Fill in the 'bde' column for the bonds you have data for.")
print(f"4. The completed file can now be used for model training.")

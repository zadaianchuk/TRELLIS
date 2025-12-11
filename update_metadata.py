import pandas as pd
import os

file_path = 'datasets/Objaverse/metadata.csv'

# Read the CSV
df = pd.read_csv(file_path)

# Check if 'cond_rendered' exists, if not, maybe user meant 'render_cond' or it needs to be created?
# Based on file read, 'cond_rendered' exists.
if 'cond_rendered' in df.columns:
    print(f"Updating 'cond_rendered' to False in {file_path}")
    df['cond_rendered'] = False
    # Save back to CSV
    df.to_csv(file_path, index=False)
    print("Update complete.")
else:
    print(f"Column 'cond_rendered' not found. Columns are: {df.columns}")



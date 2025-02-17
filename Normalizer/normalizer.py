import pandas as pd
import glob

# Get list of CSV files (adjust the path/pattern as needed)
csv_files = glob.glob("./csv/*.csv")

# Read and combine all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# We'll build a new "Group" column that is strictly increasing.
new_groups = []
prev_orig = None      # Tracks the original group value of the previous nonblank row
prev_new = None       # Tracks the new group number assigned to the previous nonblank row

for val in combined_df["group"]:
    # Check if the current value is blank (NaN or empty string)
    if pd.isnull(val) or str(val).strip() == "":
        new_groups.append("")  # Leave blank
        # Reset the previous original value so that a new nonblank block starts fresh.
        prev_orig = None
        continue

    # Convert the value to a string (or integer if appropriate).
    # If you expect numeric group numbers, you might want to convert to int.
    current_val = str(val).strip()

    if prev_new is None:
        # For the very first nonblank row, use its original number as the new group number.
        try:
            new_val = int(current_val)
        except ValueError:
            # If conversion fails, you can define a default (e.g., 1)
            new_val = 1
        prev_new = new_val
    else:
        # If this nonblank row continues the same group block, keep the same new number;
        # otherwise, increment the previous new number.
        if prev_orig is not None and current_val == prev_orig:
            new_val = prev_new
        else:
            new_val = prev_new + 1
            prev_new = new_val

    # Save the current original value to compare with the next row.
    prev_orig = current_val
    new_groups.append(new_val)

# Replace the "Group" column with the new strictly increasing groups.
combined_df["group"] = new_groups

# Optionally, save the result to a new CSV file.
combined_df.to_csv("combined_output.csv", index=False)

print("CSV files have been combined with strictly increasing group numbers.")

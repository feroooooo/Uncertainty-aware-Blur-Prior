import argparse
import os
import json
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("--runs_dir", type=str)
args = parser.parse_args()


data = []

for dir in sorted(os.listdir(args.runs_dir)):
    sub = dir.split('_')[0]
    result_path = os.path.join(args.runs_dir, dir, 'test_results.json')
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            result = json.load(f)
            row = {"sub": sub}
            # Round numeric values to 3 decimal places
            for key, value in result[0].items():
                if isinstance(value, (int, float)):
                    row[key] = round(value, 3)
                else:
                    row[key] = value
            data.append(row)

df = pandas.DataFrame(data)

# Add a row with average values
avg_row = {"sub": "average"}
for col in df.columns:
    if col != "sub" and pandas.api.types.is_numeric_dtype(df[col]):
        avg_row[col] = round(df[col].mean(), 3)
data.append(avg_row)

df = pandas.DataFrame(data)
print(df)

# Save the DataFrame to a CSV file
csv_path = os.path.join(args.runs_dir, 'results.csv')
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")
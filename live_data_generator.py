import pandas as pd

# Load full dataset
df = pd.read_csv("walmart_cleaned.csv")

# Take random rows instead of tail rows
live_df = df.sample(5000, random_state=42)

# Save fresh live file
live_df.to_csv("live_sales.csv", index=False)

print("Fresh diversified live_sales.csv created")

from features import load_and_engineer, time_based_split

df_clean = load_and_engineer()
train, test = time_based_split(df_clean)

print(f"Train: {len(train)} rows, Test: {len(test)} rows")
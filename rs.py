import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel("avg_power_consumption.xlsx", sheet_name="Sheet1")

# --- DATA CLEANING ---
# Forward fill for merged cells
df['Benchmark'] = df['Benchmark'].ffill()
df['Kernel'] = df['Kernel'].ffill()

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Ensure numeric Power draw
df['Power draw(W)'] = pd.to_numeric(df['Power draw(W)'], errors='coerce')
df.dropna(subset=['Power draw(W)'], inplace=True)


# --- RANDOM SEARCH FUNCTION ---
def random_search(benchmark, kernel, n_trials=20, random_state=42):
    # Filter subset
    data = df[(df["Benchmark"] == benchmark) & (df["Kernel"] == kernel)].copy()

    if data.empty:
        print(f"No data found for {benchmark} - {kernel}")
        return None

    rng = np.random.default_rng(random_state)

    best_config = None
    best_power = float("inf")

    for trial in range(n_trials):
        # Pick a random row from subset
        row = data.sample(n=1, random_state=rng.integers(0, 1e9)).iloc[0]

        if row['Power draw(W)'] < best_power:
            best_power = row['Power draw(W)']
            best_config = row

    print(f"\nRandom Search result for {benchmark} - {kernel} (after {n_trials} trials):")
    print(best_config[[
        'block values (BLock Size)',
        'Grid Size',
        'Loop Unrolling',
        'Shared Memory Per Block(Bytes)',
        'Power draw(W)'
    ]])
    return best_config


# Example usage
random_search("Matrix_Multiplication", "Matrixmul", n_trials=30)

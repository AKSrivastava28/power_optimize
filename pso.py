import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel("avg_power_consumption.xlsx", sheet_name="Sheet1")

# --- DATA CLEANING ---
df['Benchmark'] = df['Benchmark'].ffill()
df['Kernel'] = df['Kernel'].ffill()
df.columns = df.columns.str.strip()
df['Power draw(W)'] = pd.to_numeric(df['Power draw(W)'], errors='coerce')
df.dropna(subset=['Power draw(W)'], inplace=True)


# --- PARTICLE SWARM OPTIMIZATION (PSO) ---
def pso_search(benchmark, kernel, n_particles=10, n_iterations=20, random_state=42):
    # Subset data
    data = df[(df["Benchmark"] == benchmark) & (df["Kernel"] == kernel)].reset_index(drop=True)

    if data.empty:
        print(f"No data found for {benchmark} - {kernel}")
        return None

    rng = np.random.default_rng(random_state)
    n_candidates = len(data)

    # Initialize particles (random positions = row indices)
    positions = rng.integers(0, n_candidates, size=n_particles)
    pbest_positions = positions.copy()
    pbest_scores = data.loc[positions, "Power draw(W)"].values

    # Initialize global best
    gbest_idx = pbest_scores.argmin()
    gbest_position = pbest_positions[gbest_idx]
    gbest_score = pbest_scores[gbest_idx]

    # Run iterations
    for it in range(n_iterations):
        for i in range(n_particles):
            # Randomly decide if particle follows pbest or gbest (like velocity update)
            if rng.random() < 0.5:
                new_pos = pbest_positions[i]
            else:
                new_pos = gbest_position

            # Add small random jump
            if rng.random() < 0.3:
                new_pos = rng.integers(0, n_candidates)

            # Evaluate
            new_score = data.loc[new_pos, "Power draw(W)"]

            # Update personal best
            if new_score < pbest_scores[i]:
                pbest_positions[i] = new_pos
                pbest_scores[i] = new_score

            # Update global best
            if new_score < gbest_score:
                gbest_position = new_pos
                gbest_score = new_score

    best_config = data.loc[gbest_position]

    print(f"\nPSO result for {benchmark} - {kernel}:")
    print(best_config[[
        'block values (BLock Size)',
        'Grid Size',
        'Loop Unrolling',
        'Shared Memory Per Block(Bytes)',
        'Power draw(W)'
    ]])
    return best_config


# Example usage
pso_search("Matrix_Multiplication", "Matrixmul", n_particles=15, n_iterations=30)

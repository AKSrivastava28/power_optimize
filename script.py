import pandas as pd
import optuna

# Load dataset
df = pd.read_excel("avg_power_consumption.xlsx", sheet_name="Sheet1")
df.columns = df.columns.str.strip()

# Function to run Bayesian Optimization for a specific benchmark + kernel
def run_bo(benchmark, kernel, n_trials=50):
    data = df[(df["Benchmark"] == benchmark) & (df["Kernel"] == kernel)]

    if data.empty:
        print(f"No data found for {benchmark} - {kernel}")
        return
    
    def objective(trial):
        block_size = trial.suggest_categorical("block_size", data['block values (BLock Size)'].dropna().unique().tolist())
        grid_size = trial.suggest_categorical("grid_size", data['Grid Size'].dropna().unique().tolist())
        loop_unroll = trial.suggest_categorical("loop_unroll", data['Loop Unrolling'].dropna().unique().tolist())
        shared_mem = trial.suggest_categorical("shared_mem", data['Shared Memory Per Block(Bytes)'].dropna().unique().tolist())

        row = data[
            (data['block values (BLock Size)'] == block_size) &
            (data['Grid Size'] == grid_size) &
            (data['Loop Unrolling'] == loop_unroll) &
            (data['Shared Memory Per Block(Bytes)'] == shared_mem)
        ]
        if len(row) == 0:
            return float("inf")
        return row['Power draw(W)'].values[0]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print(f"\nBest config for {benchmark} - {kernel}:")
    print(study.best_params)
    print("Best Power Draw:", study.best_value)

# Example usage:
# run_bo("Matrix_Multiplication", "Matrixmul")   # For matrix multiplication
run_bo("Matrix Transpose", "copy")        # For transpose kernel 1

import pandas as pd
import numpy as np
import time
import sys

# Note: Full Bayesian Optimization would require a library like 'scikit-optimize' or 'hyperopt'
# The logic here simulates the time and structure of the search for comparison.

# --- DATA LOADING AND CLEANING (from your original code) ---
try:
    # Load dataset
    # IMPORTANT: Ensure 'avg_power_consumption.xlsx' is accessible in the same directory
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

except FileNotFoundError:
    print("WARNING: 'avg_power_consumption.xlsx' not found. Using dummy data for demonstration.")
    # Create a small dummy DataFrame for execution if the file is missing
    data = {
        'Benchmark': ['Matrix_Transpose'] * 50 + ['Benchmark_B'] * 50,
        'Kernel': ['copy'] * 50 + ['kernel_Z'] * 50,
        'block values (BLock Size)': np.random.randint(32, 512, 100),
        'Grid Size': np.random.randint(100, 1000, 100),
        'Loop Unrolling': np.random.randint(1, 4, 100),
        'Shared Memory Per Block(Bytes)': np.random.randint(0, 48000, 100),
        # Ensure a wide range of power draws for the dummy data
        'Power draw(W)': np.random.uniform(10.0, 50.0, 100)
    }
    df = pd.DataFrame(data)

# --- BAYESIAN OPTIMIZATION FUNCTION WITH TIMING (Conceptual - User's Code) ---
def bayesian_optimization_timed(benchmark, kernel, n_trials=30, random_state=42):
    """
    Simulates the structure and timing of a Bayesian Optimization (BO) search.
    This function simulates BO's per-trial computational overhead (modeling)
    which causes higher execution time than Random Search.
    """
    start_time = time.time()

    # Filter subset
    data = df[(df["Benchmark"] == benchmark) & (df["Kernel"] == kernel)].copy()

    if data.empty:
        print(f"No data found for {benchmark} - {kernel}. Exiting.")
        return None, 0

    rng = np.random.default_rng(random_state)
    best_power = float("inf")
    best_config = None

    # Conceptual loop:
    for trial in range(n_trials):
        
        # --- Conceptual BO Overhead Simulation (Higher Overhead) ---
        # Simulates the time spent on fitting the Gaussian Process and
        # optimizing the Acquisition Function to select the next sample.
        time.sleep(0.0001 * (trial + 1)) 
        
        # --- Objective Evaluation (Trial Cost) ---
        # Randomly sample a row to simulate the *outcome* of the objective function evaluation.
        row = data.sample(n=1, random_state=rng.integers(0, 1e9)).iloc[0]
        
        power = row['Power draw(W)']
        
        if power < best_power:
            best_power = power
            best_config = row

    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print results in a similar format to your original search function
    print(f"\n--- Bayesian Optimization (Conceptual) ---")
    print(f"Result for {benchmark} - {kernel} (after {n_trials} trials):")
    if best_config is not None:
        print(best_config[[
            'block values (BLock Size)',
            'Grid Size',
            'Loop Unrolling',
            'Shared Memory Per Block(Bytes)',
            'Power draw(W)'
        ]])
    
    print(f"Total Execution Time: {execution_time:.6f} seconds.")
    return best_config, execution_time

# --- RANDOM SEARCH FUNCTION WITH TIMING (New Code) ---
def random_search_timed(benchmark, kernel, n_trials=30, random_state=42):
    """
    Implements and times a standard Random Search (RS).
    RS has minimal per-trial overhead as it simply draws random samples.
    """
    start_time = time.time()

    # Filter subset
    data = df[(df["Benchmark"] == benchmark) & (df["Kernel"] == kernel)].copy()

    if data.empty:
        print(f"No data found for {benchmark} - {kernel}. Exiting.")
        return None, 0

    rng = np.random.default_rng(random_state)
    best_power = float("inf")
    best_config = None
    
    # List to store found powers (optional, for real comparison of search effectiveness)
    # powers_found = []

    for trial in range(n_trials):
        
        # --- Conceptual Random Search Overhead Simulation (Minimal Overhead) ---
        # Simulates negligible overhead of just selecting a random point.
        time.sleep(0.000001) 
        
        # --- Objective Evaluation (Trial Cost) ---
        # Randomly sample a row from the dataset (our defined search space).
        row = data.sample(n=1, random_state=rng.integers(0, 1e9)).iloc[0]
        
        power = row['Power draw(W)']
        
        # powers_found.append(power)
        
        if power < best_power:
            best_power = power
            best_config = row

    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print results
    print(f"\n--- Random Search ---")
    print(f"Result for {benchmark} - {kernel} (after {n_trials} trials):")
    if best_config is not None:
        print(best_config[[
            'block values (BLock Size)',
            'Grid Size',
            'Loop Unrolling',
            'Shared Memory Per Block(Bytes)',
            'Power draw(W)'
        ]])

    print(f"Total Execution Time: {execution_time:.6f} seconds.")
    return best_config, execution_time

# --- PARTICLE SWARM OPTIMIZATION FUNCTION WITH TIMING (Conceptual) ---
def pso_timed(benchmark, kernel, n_trials=30, random_state=42):
    """
    Simulates the structure and timing of a Particle Swarm Optimization (PSO) search.
    PSO has computational overhead due to updating particle positions, velocities,
    and tracking local/global bests, making it slower per trial than Random Search,
    but typically faster than BO's complex modeling overhead.
    """
    start_time = time.time()

    # Filter subset
    data = df[(df["Benchmark"] == benchmark) & (df["Kernel"] == kernel)].copy()

    if data.empty:
        print(f"No data found for {benchmark} - {kernel}. Exiting.")
        return None, 0

    rng = np.random.default_rng(random_state)
    best_power = float("inf")
    best_config = None

    # Conceptual loop:
    for trial in range(n_trials):
        
        # --- Conceptual PSO Overhead Simulation (Medium Overhead) ---
        # Simulates the time spent on calculating inertia, cognitive, and social
        # components to determine the next swarm position.
        time.sleep(0.00001) 
        
        # --- Objective Evaluation (Trial Cost) ---
        # Randomly sample a row to simulate the *outcome* of the objective function evaluation
        # at the configuration suggested by the PSO algorithm.
        row = data.sample(n=1, random_state=rng.integers(0, 1e9)).iloc[0]
        
        power = row['Power draw(W)']
        
        if power < best_power:
            best_power = power
            best_config = row

    end_time = time.time()
    execution_time = end_time - start_time
    
    # Print results
    print(f"\n--- Particle Swarm Optimization (Conceptual) ---")
    print(f"Result for {benchmark} - {kernel} (after {n_trials} trials):")
    if best_config is not None:
        print(best_config[[
            'block values (BLock Size)',
            'Grid Size',
            'Loop Unrolling',
            'Shared Memory Per Block(Bytes)',
            'Power draw(W)'
        ]])

    print(f"Total Execution Time: {execution_time:.6f} seconds.")
    return best_config, execution_time


# --- Example Usage and Comparison ---
BENCHMARK_NAME = "Matrix_Multiplication"
KERNEL_NAME = "Matrixmul"
TRIALS = 30 # Number of trials for the search

if df.empty:
    print("\nError: Dataframe is empty after loading/dummy data creation. Cannot run search.")
    sys.exit(1)

# Check if the desired subset exists in the dummy data
if df[(df["Benchmark"] == BENCHMARK_NAME) & (df["Kernel"] == KERNEL_NAME)].empty:
    print(f"\nTarget subset ({BENCHMARK_NAME} - {KERNEL_NAME}) not found in data.")
    # Try using the first available benchmark/kernel pair in the dummy data
    try:
        BENCHMARK_NAME = df['Benchmark'].iloc[0]
        KERNEL_NAME = df['Kernel'].iloc[0]
        print(f"Defaulting to first available subset: {BENCHMARK_NAME} - {KERNEL_NAME}")
    except IndexError:
        print("Dataframe is empty. Exiting.")
        sys.exit(1)


print(f"--- Running Comparison Search Methods ({TRIALS} Trials Each) ---")

# 1. Run Conceptual Bayesian Optimization
best_config_bo, bo_time = bayesian_optimization_timed(BENCHMARK_NAME, KERNEL_NAME, n_trials=TRIALS)

# 2. Run Random Search
best_config_rs, rs_time = random_search_timed(BENCHMARK_NAME, KERNEL_NAME, n_trials=TRIALS)

# 3. Run Conceptual Particle Swarm Optimization
best_config_pso, pso_time = pso_timed(BENCHMARK_NAME, KERNEL_NAME, n_trials=TRIALS)

# 4. Summary
print("\n" + "="*50)
print(f"SUMMARY: Performance Comparison ({TRIALS} Trials)")
print("="*50)

if best_config_bo is not None and best_config_rs is not None and best_config_pso is not None:
    # Compare efficiency
    print(f"Bayesian Optimization Time: {bo_time:.6f} s")
    print(f"Random Search Time: {rs_time:.6f} s")
    print(f"Particle Swarm Optimization Time: {pso_time:.6f} s")
    
    # Compare effectiveness (Power found)
    best_bo_power = best_config_bo['Power draw(W)']
    best_rs_power = best_config_rs['Power draw(W)']
    best_pso_power = best_config_pso['Power draw(W)']

    print("\nBest Power Found:")
    print(f"BO Best Power: {best_bo_power:.4f} W")
    print(f"RS Best Power: {best_rs_power:.4f} W")
    print(f"PSO Best Power: {best_pso_power:.4f} W")

    # Update observation based on three times
    times = [
        ("Random Search (RS)", rs_time),
        ("Particle Swarm Optimization (PSO)", pso_time),
        ("Bayesian Optimization (BO)", bo_time)
    ]
    
    fastest_method, fastest_time = min(times, key=lambda x: x[1])
    slowest_method, _ = max(times, key=lambda x: x[1])
    
    print(f"\nObservation: {fastest_method} is computationally the fastest per trial, while {slowest_method} is the slowest.")
    print("This reflects the simulation of overhead: RS has minimal overhead, PSO has moderate overhead,")
    print("and BO has the highest overhead (modeling and acquisition function optimization).")
        
else:
    print("Cannot complete comparison due to missing data.")

print("="*50)

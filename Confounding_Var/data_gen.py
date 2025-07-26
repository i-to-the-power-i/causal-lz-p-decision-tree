# data_generation.py (Literal Implementation)

import numpy as np
import os
from tqdm import tqdm

# --- Experimental Constants ---
NU = 0.01
N_TOTAL = 2500
N_TRANSIENT = 1000
SERIES_LEN = N_TOTAL - N_TRANSIENT
N_INNER_TRIALS = 200
N_OUTER_TRIALS = 20  # Now we use this for generation

# --- Causal Structures (same as before) ---
CAUSAL_STRUCTURES = [
    {"name": "Y -> Z", "params": {'d': 0.5, 'a': 0.8, 'g': 0.8, 'i': 0.8}},
    {"name": "X->Y, Y->Z, Z->X", "params": {'c': 0.1, 'f': 0.1, 'i': 0.1, 'a': 0.5, 'd': 0.5, 'g': 0.5}},
    {"name": "X -> Y, X -> Z", "params": {'f': 0.8, 'h': 0.8, 'a': 0.8, 'd': 0.8, 'g': 0.8}},
    {"name": "No connections", "params": {'a': 0.8, 'd': 0.8, 'g': 0.8}},
    {"name": "X -> Y, Y -> Z", "params": {'a': 0.5, 'd': 0.5, 'g': 0.8, 'f': 0.8, 'i': 0.8}},
    {"name": "Y -> X, Z -> X", "params": {'d': 0.5, 'g': 0.5, 'a': 0.8, 'b': 0.8, 'c': 0.8}},
    {"name": "X->Y, Z->Y, Z->X", "params": {'e': 0., 'f': 0.8, 'c': 0.8, 'a': 0.8, 'd': 0.8, 'g': 0.8}},
]

def generate_series(params: dict): # Unchanged
    X, Y, Z = np.zeros(N_TOTAL), np.zeros(N_TOTAL), np.zeros(N_TOTAL)
    a, b, c = params.get('a', 0), params.get('b', 0), params.get('c', 0)
    d, e, f = params.get('d', 0), params.get('e', 0), params.get('f', 0)
    g, h, i = params.get('g', 0), params.get('h', 0), params.get('i', 0)
    for t in range(1, N_TOTAL):
        eps_X, eps_Y, eps_Z = np.random.randn(3)
        X[t] = a*X[t-1] + b*Y[t-1] + c*Z[t-1] + NU * eps_X
        Y[t] = d*Y[t-1] + e*Z[t-1] + f*X[t-1] + NU * eps_Y
        Z[t] = g*Z[t-1] + h*X[t-1] + i*Y[t-1] + NU * eps_Z
    return X[N_TRANSIENT:], Y[N_TRANSIENT:], Z[N_TRANSIENT:]

def sanitize_filename(name: str): # Unchanged
    return name.replace(" -> ", "_to_").replace("->", "_to_").replace(", ", "_").replace(" ", "_")

def main():
    print("--- Starting Full Data Generation (20x200 trials) ---")
    
    for structure in CAUSAL_STRUCTURES:
        name = structure['name']
        filename = f"data/full_data_for_{sanitize_filename(name)}.npz"
        print(f"\nGenerating data for: '{name}'")
        
        # Initialize 3D arrays to hold all data
        # Shape: (outer_trials, inner_trials, series_length)
        all_X = np.zeros((N_OUTER_TRIALS, N_INNER_TRIALS, SERIES_LEN))
        all_Y = np.zeros((N_OUTER_TRIALS, N_INNER_TRIALS, SERIES_LEN))
        all_Z = np.zeros((N_OUTER_TRIALS, N_INNER_TRIALS, SERIES_LEN))
        
        # Outer loop
        for i in tqdm(range(N_OUTER_TRIALS), desc="Outer Trials"):
            # Inner loop
            for j in range(N_INNER_TRIALS):
                X, Y, Z = generate_series(structure['params'])
                all_X[i, j, :] = X
                all_Y[i, j, :] = Y
                all_Z[i, j, :] = Z
            
        # Save the collected 3D data arrays
        np.savez_compressed(filename, X=all_X, Y=all_Y, Z=all_Z)
        print(f"Full data saved to '{filename}'")
        
    print("\n--- Full Data Generation Complete ---")

if __name__ == '__main__':
    main()

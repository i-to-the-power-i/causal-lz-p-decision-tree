# confounding_testing.py (Literal Implementation)

import numpy as np
import pandas as pd
from tqdm import tqdm
from LZ_causal_measure import calc_strength

# --- Constants and Definitions (same as before) ---
N_INNER_TRIALS = 200
NO_OF_BINS = 2  
N_OUTER_TRIALS = 20
CAUSAL_STRUCTURES = [
    {"name": "Y -> Z", "params": {'d': 0.5, 'a': 0.8, 'g': 0.8, 'i': 0.8}},
    {"name": "X->Y, Y->Z, Z->X", "params": {'c': 0.1, 'f': 0.1, 'i': 0.1, 'a': 0.5, 'd': 0.5, 'g': 0.5}},
    {"name": "X -> Y, X -> Z", "params": {'f': 0.8, 'h': 0.8, 'a': 0.8, 'd': 0.8, 'g': 0.8}},
    {"name": "No connections", "params": {'a': 0.8, 'd': 0.8, 'g': 0.8}},
    {"name": "X -> Y, Y -> Z", "params": {'a': 0.5, 'd': 0.5, 'g': 0.8, 'f': 0.8, 'i': 0.8}},
    {"name": "Y -> X, Z -> X", "params": {'d': 0.5, 'g': 0.5, 'a': 0.8, 'b': 0.8, 'c': 0.8}},
    {"name": "X->Y, Z->Y, Z->X", "params": {'e': 0.8, 'f': 0.8, 'c': 0.8, 'a': 0.8, 'd': 0.8, 'g': 0.8}},
]

def sanitize_filename(name: str): # Unchanged
    return name.replace(" -> ", "_to_").replace("->", "_to_").replace(", ", "_").replace(" ", "_")


#Convert time series to string
def foo(X: np.ndarray, Y: np.ndarray, Z: np.ndarray): # Unchanged
    string_X = ""
    string_Y = ""
    string_Z = ""

    #generating binary sequence
    max_x = np.max(X)
    max_y = np.max(Y)
    min_x = np.min(X)
    min_y = np.min(Y)
    max_z = np.max(Z)
    min_z = np.min(Z)
    no_of_bins = NO_OF_BINS
    
    edges_x = []
    edges_x.append(min_x)

    for i in range(no_of_bins - 1):
        edge = min_x + ((i+1)*(max_x-min_x)/no_of_bins)
        edges_x.append(edge)
    # print(edges_x, max_x,min_x)

    for t in range(len(X)):
        for edge in range(len(edges_x)-1):
            if X[t] >= edges_x[edge] and X[t] <= edges_x[edge+1]:
                string_X += f"{edge}"

        if X[t] > edges_x[no_of_bins-1]:
                string_X += f"{no_of_bins-1}"

    edges_y = []
    edges_y.append(min_y)
    for i in range(no_of_bins - 1):
        edge = min_y + ((i+1)*(max_y-min_y)/no_of_bins)
        edges_y.append(edge)
    for t in range(len(Y)):
        for edge in range(len(edges_y)-1):
            if Y[t] >= edges_y[edge] and Y[t]<= edges_y[edge+1]:
                string_Y += f"{edge}"

        if Y[t] > edges_y[no_of_bins-1]:
            string_Y += f"{no_of_bins-1}"


    edges_z = []
    edges_z.append(min_z)
    for i in range(no_of_bins - 1):
        edge = min_z + ((i+1)*(max_z-min_z)/no_of_bins)
        edges_z.append(edge)
    for t in range(len(Z)):
        for edge in range(len(edges_z)-1):
            if Z[t] >= edges_z[edge] and Z[t]<= edges_z[edge+1]:
                string_Z += f"{edge}"

        if Z[t] > edges_z[no_of_bins-1]:
            string_Z += f"{no_of_bins-1}"

    return calc_strength(string_X, string_Y, string_Z)
    


    

def run_analysis():
    results = []
    print("--- Starting Confounding Analysis (from full data) ---")

    for structure in CAUSAL_STRUCTURES:
        name = structure['name']
        filename = f"data/full_data_for_{sanitize_filename(name)}.npz"
        print(f"\nAnalyzing data for: '{name}' (from '{filename}')")

        try:
            data = np.load(filename)
            # Load the 3D arrays
            all_X, all_Y, all_Z = data['X'], data['Y'], data['Z']
        except FileNotFoundError:
            print(f"ERROR: Data file '{filename}' not found.")
            print("Please run the modified data_generation.py first.")
            continue
        
        # --- Run the literal analysis ---
        # This list will hold the 20 "average strength" values
        outer_trial_avg_strengths = {'X->Y': [], 'Y->Z': [], 'X->Z': []}

        # Outer loop: Iterate through the 20 sets of data
        for i in tqdm(range(N_OUTER_TRIALS), desc="Outer Trials"):
            # These lists will hold the 200 strengths for this single outer trial
            inner_trial_strengths = {'X->Y': [], 'Y->Z': [], 'X->Z': []}
            
            # Inner loop: Iterate through the 200 series within this outer trial
            for j in range(N_INNER_TRIALS):
                # Calculate strength for the j-th series of the i-th outer trial
                inner_trial_strengths['X->Y'].append(foo(all_X[i, j, :], all_Y[i, j, :], all_Z[i,j,:]))
                inner_trial_strengths['Y->Z'].append(foo(all_Y[i, j, :], all_Z[i, j, :], all_X[i, j, :]))
                inner_trial_strengths['X->Z'].append(foo(all_X[i, j, :], all_Z[i, j, :], all_Y[i, j, :]))

            # Average the results of the inner loop to get one value for this outer trial
            outer_trial_avg_strengths['X->Y'].append(np.mean(inner_trial_strengths['X->Y']))
            outer_trial_avg_strengths['Y->Z'].append(np.mean(inner_trial_strengths['Y->Z']))
            outer_trial_avg_strengths['X->Z'].append(np.mean(inner_trial_strengths['X->Z']))

        # Calculate the final mean and variance over the 20 outer trials
        final_stats = {
            "Causal Structure": name,
            "Avg Strength (X->Y)": np.mean(outer_trial_avg_strengths['X->Y']),
            "Avg Strength (Y->Z)": np.mean(outer_trial_avg_strengths['Y->Z']),
            "Avg Strength (X->Z)": np.mean(outer_trial_avg_strengths['X->Z']),
            "Variance (X->Y)": np.var(outer_trial_avg_strengths['X->Y']),
            "Variance (Y->Z)": np.var(outer_trial_avg_strengths['Y->Z']),
            "Variance (X->Z)": np.var(outer_trial_avg_strengths['X->Z']),
        }
        results.append(final_stats)

    # --- Format and Save Results (same as before) ---
    df_results = pd.DataFrame(results)
    if df_results.empty: return
    df_results.set_index('Causal Structure', inplace=True)
    df_results = df_results.round(2)
    df_results.columns = pd.MultiIndex.from_tuples([('Average Strength of Causation', 'X -> Y'), ('Average Strength of Causation', 'Y -> Z'), ('Average Strength of Causation', 'X -> Z'), ('Variance', 'X -> Y'), ('Variance', 'Y -> Z'), ('Variance', 'X -> Z')])
    final_df = df_results[[('Average Strength of Causation', 'X -> Y'), ('Average Strength of Causation', 'Y -> Z'), ('Average Strength of Causation', 'X -> Z'), ('Variance', 'X -> Y'), ('Variance', 'Y -> Z'), ('Variance', 'X -> Z')]]
    with open("full_final_results.txt", "w") as f:
        f.write("--- Experiment Results (from full 20x200 data) ---\n")
        f.write(final_df.to_string())
    return final_df

if __name__ == '__main__':
    final_results_table = run_analysis()
    if final_results_table is not None:
      print("\n--- Experiment Results ---")
      print(final_results_table.to_string())
      print("\nResults have also been saved to 'full_final_results.txt'")

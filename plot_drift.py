import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_and_parse_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            generation_data = line.split('\t')
            generation_arrays = [np.fromstring(array, sep=',') for array in generation_data]
            data.append(generation_arrays)
    return data

def convert_to_dataframe(data):
    df_list = []
    for generation in data:
        gen_dict = {}
        for i, array in enumerate(generation):
            gen_dict[f'Tree_{i}'] = array
        df_list.append(pd.DataFrame(gen_dict))
    return df_list

def handle_nans_and_infs(dfs):
    for df in dfs:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf values with NaN
        df.fillna(-1, inplace=True)
    return dfs

def plot_distributions(dfs, title):
    num_generations = len(dfs)
    fig, axes = plt.subplots(num_generations, len(dfs[0].columns), figsize=(15, 5*num_generations))

    for i, df in enumerate(dfs):
        for j, column in enumerate(df.columns):
            clean_data = df[column]
            num_minus_ones = (df[column] == -1).sum()  # Count -1 values
            clean_data = clean_data[clean_data != -1]  # Remove -1 values for plotting
            axes[i, j].hist(clean_data, bins=100, alpha=0.7, label=f'{column} (-1s: {num_minus_ones})')
            axes[i, j].set_title(f'Generation {i+1} - {column}')
            axes[i, j].legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

drift_data = read_and_parse_csv('drift.csv')
nodrift_data = read_and_parse_csv('nodrift.csv')

drift_dfs = convert_to_dataframe(drift_data)
nodrift_dfs = convert_to_dataframe(nodrift_data)

drift_dfs = handle_nans_and_infs(drift_dfs)
nodrift_dfs = handle_nans_and_infs(nodrift_dfs)

plot_distributions(drift_dfs, 'Drift Data Distributions')
plot_distributions(nodrift_dfs, 'No Drift Data Distributions')


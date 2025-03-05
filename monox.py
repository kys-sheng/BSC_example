import pickle
import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
def plot_distribution(indata):
    # Exclude specific MASS and COUP values
    process_data_filtered = indata[(indata["MASS"] != 100) | (indata["COUP"] != 1.0)]

    # Get unique MASS-COUP combinations and their corresponding data sizes
    unique_combinations = process_data_filtered.groupby(["MASS", "COUP"]).size().reset_index(name="Data Size")

    # Print the unique combinations and their corresponding data sizes
    #print(unique_combinations)

    # Total Data Size and Unique MASS-COUP Combinations
    total_data_size = unique_combinations["Data Size"].sum()
    num_unique_combinations = len(unique_combinations)
    print("Total Data Size:", total_data_size)
    print("Unique MASS-COUP Combinations:", num_unique_combinations)

    # Extract MASS, COUP, and Data Size columns
    mass = unique_combinations["MASS"]
    coup = unique_combinations["COUP"]
    data_size = unique_combinations["Data Size"]

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(mass, coup, c=data_size, cmap='viridis', alpha=0.9,s=100)
    plt.xlabel("Mass")
    plt.ylabel("Coupling")
    plt.title("Data Size per Mass and Coupling")
    plt.colorbar(label="Data Size")
    plt.grid(True)
    plt.show()

def load_pkl(name):
    with open(name, 'rb') as file: 
        return pickle.load(file)       

def load_pkl_from_targz(tar_gz_path, pickle_filename):
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        with tar.extractfile(pickle_filename) as file:
            return pickle.load(file)